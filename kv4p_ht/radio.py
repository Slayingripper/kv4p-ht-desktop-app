"""
Serial I/O and Audio I/O workers as QThreads.
"""
from __future__ import annotations

import queue
import time

import serial
from PyQt6.QtCore import QThread, pyqtSignal
from serial.tools import list_ports

from .protocol import (
    EspCmd,
    FrameParser,
    FrameSender,
    unpack_rssi,
    unpack_version,
    unpack_window_update,
)

KV4P_VID_PID = [(0x10C4, 0xEA60), (0x1A86, 0x7523)]
_CACHED_PORT: str | None = None
_CACHED_PORT_TTL = 0


def find_kv4p_port() -> str | None:
    global _CACHED_PORT, _CACHED_PORT_TTL
    now = time.monotonic()
    if _CACHED_PORT and now < _CACHED_PORT_TTL:
        return _CACHED_PORT
    for port in list_ports.comports():
        if port.vid is not None and port.pid is not None and (port.vid, port.pid) in KV4P_VID_PID:
            _CACHED_PORT = port.device
            _CACHED_PORT_TTL = now + 5.0
            return port.device
    _CACHED_PORT = None
    _CACHED_PORT_TTL = now + 2.0
    return None


class SerialWorker(QThread):
    version_received = pyqtSignal(object)
    hello_received = pyqtSignal()
    smeter = pyqtSignal(int)
    ax25_packet = pyqtSignal(int, bytes)
    debug_msg = pyqtSignal(int, str)
    connected = pyqtSignal(bool)
    window_update = pyqtSignal(int)
    tx_cmd = pyqtSignal(int)  # emitted when a command is written to serial
    phys_ptt_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._port: serial.Serial | None = None
        self._sender: FrameSender | None = None
        self._running = True
        self.cmd_queue: queue.SimpleQueue = queue.SimpleQueue()
        self.rx_opus_queue: queue.SimpleQueue | None = None
        self._window = 2048

    def stop(self):
        self._running = False
        self.cmd_queue.put(None)
        self.wait(3000)

    def send_raw(self, data: bytes):
        if self._sender:
            self._sender._write(data)

    def run(self):
        while self._running:
            port_path = find_kv4p_port()
            if not port_path:
                self.connected.emit(False)
                time.sleep(2)
                continue

            try:
                ser = serial.Serial(port_path, 115200, timeout=0.1)
                self._port = ser
                self._sender = FrameSender(ser.write)
                self.connected.emit(True)
                self._sender.send_config(is_high=True)
                parser = FrameParser(self._on_esp_command)
                tx_count = 0

                while self._running:
                    # Use short timeout when commands are queued so we drain
                    # the queue promptly; otherwise idle with longer timeout.
                    pending = not self.cmd_queue.empty()
                    try:
                        data = ser.read(4096 if not pending else 0)
                    except serial.SerialException:
                        break
                    if data:
                        parser.feed(data)

                    while not self.cmd_queue.empty():
                        try:
                            cmd = self.cmd_queue.get_nowait()
                        except queue.Empty:
                            break
                        if cmd is None:
                            break
                        try:
                            ser.write(cmd)
                            tx_count += 1
                            if len(cmd) >= 5:
                                self.tx_cmd.emit(cmd[4])
                        except serial.SerialException as e:
                            self.debug_msg.emit(2, f"Write error: {e}")
                            break

                    if not data and not pending:
                        time.sleep(0.001)

                ser.close()
            except serial.SerialException as e:
                self.debug_msg.emit(2, f"Serial error: {e}")
                self.connected.emit(False)
                time.sleep(3)
            except Exception as e:
                self.debug_msg.emit(2, f"Serial thread: {e}")
                self.connected.emit(False)
                time.sleep(3)
            finally:
                self._port = None
                self._sender = None

    def _on_esp_command(self, cmd: int, payload: bytes):
        if cmd == EspCmd.HELLO:
            self.hello_received.emit()
        elif cmd == EspCmd.VERSION:
            if len(payload) >= 12:
                self.version_received.emit(unpack_version(payload))
        elif cmd == EspCmd.RX_AUDIO:
            # Write directly to audio queue, bypassing main thread Qt signal
            if self.rx_opus_queue is not None:
                self.rx_opus_queue.put(payload)
        elif cmd == EspCmd.SMETER_REPORT:
            self.smeter.emit(unpack_rssi(payload))
        elif cmd == EspCmd.RX_AX25_PACKET:
            if payload:
                self.ax25_packet.emit(payload[0], payload[1:])
        elif cmd == EspCmd.WINDOW_UPDATE:
            self._window += unpack_window_update(payload)
            self.window_update.emit(self._window)
        elif cmd == EspCmd.PHYS_PTT_DOWN:
            self.debug_msg.emit(0, "Physical PTT pressed")
            self.phys_ptt_changed.emit(True)
        elif cmd == EspCmd.PHYS_PTT_UP:
            self.debug_msg.emit(0, "Physical PTT released")
            self.phys_ptt_changed.emit(False)
        elif EspCmd.DEBUG_INFO <= cmd <= EspCmd.DEBUG_TRACE:
            text = payload.decode('utf-8', errors='replace')
            self.debug_msg.emit(cmd, text)


class AudioWorker(QThread):
    opus_for_tx = pyqtSignal(bytes)
    aprs_packet = pyqtSignal(object)
    morse_samples = pyqtSignal(object)
    debug_msg = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.rx_opus_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._pcm_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._demod_pcm_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._inject_queue: queue.SimpleQueue = queue.SimpleQueue()
        self.spectrum_pcm_queue: queue.SimpleQueue | None = None
        self.morse_decoder_active = False
        self.sstv_decoder_active = False
        self.mic_gain: float = 1.0
        self.speaker_volume: float = 1.0
        self._sr = 48000
        self._frame = 1920

    def stop(self):
        self._running = False
        self._pcm_queue.put(None)
        self.rx_opus_queue.put(None)
        self._demod_pcm_queue.put(None)
        self._inject_queue.put(None)
        self.wait(3000)

    def run(self):
        try:
            import numpy as np
            import opuslib
            import sounddevice as sd
        except ImportError as e:
            print(f"[Audio] Deps missing: {e}")
            return

        try:
            enc = opuslib.Encoder(self._sr, 1, opuslib.APPLICATION_AUDIO)
            dec = opuslib.Decoder(self._sr, 1)
        except Exception as e:
            print(f"[Audio] Opus init: {e}")
            return

        def mic_cb(indata, frames, time_info, status):
            if status:
                print(f"[Audio] Mic: {status}")
            if self._running:
                scaled = indata * self.mic_gain
                self._pcm_queue.put(scaled.copy())

        try:
            instream = sd.InputStream(
                samplerate=self._sr, channels=1,
                blocksize=self._frame, callback=mic_cb,
            )
            instream.start()
        except Exception as e:
            print(f"[Audio] Mic stream: {e}")
            return

        _rx_pcm_queue: queue.SimpleQueue = queue.SimpleQueue()
        _rx_pcm_buf: list[np.ndarray] = []
        _rx_prebuf_target = 5  # frames to buffer before playback
        _rx_prebuffering = True

        def spk_cb(outdata, frames, time_info, status):
            nonlocal _rx_pcm_buf, _rx_prebuffering
            if status and not status.output_underflow:
                print(f"[Audio] Speaker: {status}")
            # Drain any available decoded frames
            while True:
                try:
                    _rx_pcm_buf.append(_rx_pcm_queue.get_nowait())
                except queue.Empty:
                    break
            if _rx_prebuffering and len(_rx_pcm_buf) < _rx_prebuf_target:
                outdata[:, 0] = 0.0
                return
            _rx_prebuffering = False
            n = len(outdata)
            written = 0
            while written < n:
                if _rx_pcm_buf:
                    chunk = _rx_pcm_buf[0]
                    take = min(len(chunk), n - written)
                    outdata[written:written + take, 0] = chunk[:take]
                    written += take
                    if take < len(chunk):
                        _rx_pcm_buf[0] = chunk[take:]
                    else:
                        _rx_pcm_buf.pop(0)
                else:
                    outdata[written:, 0] = 0.0
                    break

        try:
            outstream = sd.OutputStream(
                samplerate=self._sr, channels=1,
                blocksize=self._frame, callback=spk_cb,
            )
            outstream.start()
        except Exception as e:
            print(f"[Audio] Speaker stream: {e}")
            instream.stop()
            return

        print("[Audio] Audio started")
        acc = bytearray()
        _demod_loop_count = 0
        _demod_total_samples = 0
        PREBUF_FRAMES = 5

        import queue as _q

        from .afsk_demod import AfskDemodulator
        _pkt_queue: _q.SimpleQueue = _q.SimpleQueue()
        _demod = AfskDemodulator(callback=lambda p: _pkt_queue.put(p))

        while self._running:
            try:
                # Drain microphone PCM → encode as Opus
                try:
                    samples = self._pcm_queue.get(timeout=0.1)
                    if samples is None:
                        break
                    mono = samples.squeeze()
                    i16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
                    acc.extend(i16.tobytes())

                    frame_bytes = self._frame * 2
                    while len(acc) >= frame_bytes:
                        frame = bytes(acc[:frame_bytes])
                        del acc[:frame_bytes]
                        try:
                            opus = enc.encode(frame, self._frame)
                            self.opus_for_tx.emit(opus)
                        except Exception as e:
                            print(f"[Audio] Opus encode: {e}")
                except queue.Empty:
                    pass

                # Decode RX Opus → PCM for speaker + demod + spectrum + morse
                for _ in range(10):
                    try:
                        raw = self.rx_opus_queue.get_nowait()
                        if raw is None:
                            break
                        try:
                            pcm_bytes = dec.decode(raw, self._frame)
                            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            _rx_pcm_queue.put(arr * self.speaker_volume)
                            _demod_pcm_queue.put(arr.copy())
                        except Exception as e:
                            print(f"[Audio] Opus decode: {e}")
                    except _q.Empty:
                        break

                # Drain decoded RX PCM → feed to AFSK demodulator + spectrum + morse
                for _ in range(10):
                    try:
                        pcm = self._demod_pcm_queue.get_nowait()
                        if pcm is None:
                            break
                        _demod.process(pcm.ravel().tolist())
                        _demod_total_samples += len(pcm)
                        # Feed to spectrum analyzer
                        if self.spectrum_pcm_queue is not None:
                            try:
                                self.spectrum_pcm_queue.put_nowait(pcm.copy())
                            except queue.Full:
                                pass
                        # Emit PCM for morse/SSTV decoders (only if listeners exist)
                        if self.morse_decoder_active or self.sstv_decoder_active:
                            self.morse_samples.emit(pcm.copy())
                    except _q.Empty:
                        break

                # Drain self-test injection → feed to AFSK demodulator
                for _ in range(10):
                    try:
                        wav = self._inject_queue.get_nowait()
                        if wav is None:
                            break
                        _demod.process(wav.ravel().tolist())
                        self.debug_msg.emit(f"Self-test: fed {len(wav)} samples to demod")
                    except _q.Empty:
                        break

                # Emit any decoded APRS packets (from demod callback)
                for _ in range(10):
                    try:
                        pkt = _pkt_queue.get_nowait()
                        self.aprs_packet.emit(pkt)
                    except _q.Empty:
                        break

                _demod_loop_count += 1
                if _demod_loop_count % 500 == 0:
                    self.debug_msg.emit(
                        f"Demod: {_demod_loop_count} loops, "
                        f"{_demod_total_samples} samples"
                    )

            except Exception as e:
                print(f"[Audio] Loop error: {e}")
                time.sleep(0.5)

        instream.stop()
        outstream.stop()
        self.debug_msg.emit("Audio thread stopped")

    def inject_audio(self, waveform):
        """Inject a waveform into the AFSK demodulator (for loopback testing)."""
        self._inject_queue.put(waveform)

    def inject_selftest(self):
        """Inject a synthetic AFSK waveform into the demodulator to test the decode path.
        Tests both direct (bypassing Opus) and Opus round-trip paths."""
        import numpy as np

        from .afsk import build_tx_waveform_from_body
        from .aprs import encode_ax25_ui, format_message
        body = encode_ax25_ui('SELFTST', 'TESTDST', [],
                              format_message('TESTDST', 'Hello from self-test').encode())
        wav = build_tx_waveform_from_body(body)
        self._inject_queue.put(wav)
        self.debug_msg.emit(f"Self-test direct: queued {len(wav)} samples")

        # Also test through Opus round-trip to simulate real RX path
        frame_size = 1920
        total = len(wav)
        pad = (frame_size - (total % frame_size)) % frame_size
        wav_padded = np.pad(wav, (0, pad))
        import opuslib
        enc = opuslib.Encoder(48000, 1, opuslib.APPLICATION_AUDIO)
        dec = opuslib.Decoder(48000, 1)
        opus_out = np.empty(total + pad, dtype=np.float32)
        pos = 0
        for start in range(0, total + pad, frame_size):
            frame = wav_padded[start:start + frame_size]
            i16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
            opus_data = enc.encode(i16.tobytes(), frame_size)
            pcm = dec.decode(opus_data, frame_size)
            decoded = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            take = min(len(decoded), total - pos)
            opus_out[pos:pos + take] = decoded[:take]
            pos += take
        self._inject_queue.put(opus_out[:total])
        self.debug_msg.emit(f"Self-test opus: queued {total} samples (round-trip)")
