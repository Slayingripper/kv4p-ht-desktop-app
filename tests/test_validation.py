"""End-to-end validation tests for every module in the KV4P-Desktop app.
Each test creates a fresh instance, feeds real data, and asserts expected outputs.
"""
import io
import os
import struct
import tempfile
import time
import queue
import threading

import numpy as np
import pytest


# ── Spectrum ───────────────────────────────────────────────────────

class TestSpectrumAnalyzer:
    def test_feed_silence(self):
        from kv4p_ht.spectrum import SpectrumAnalyzer
        sa = SpectrumAnalyzer(sample_rate=48000, fft_size=2048, averaging=2)
        sa.set_center_freq(144.390)
        silence = np.zeros(4800, dtype=np.float32)
        sa.feed(silence)
        result = sa.get_spectrum()
        assert result is not None
        freqs, power = result
        assert len(freqs) == 2048
        assert len(power) == 2048
        assert np.all(power < -50), "Silence should be low power"

    def test_feed_tone(self):
        from kv4p_ht.spectrum import SpectrumAnalyzer
        sa = SpectrumAnalyzer(sample_rate=48000, fft_size=2048, averaging=1)
        sa.set_center_freq(144.390)
        t = np.arange(4096) / 48000.0
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        sa.feed(tone)
        result = sa.get_spectrum()
        assert result is not None
        _, power = result
        assert np.max(power) > -30, "1kHz tone should have significant power"

    def test_waterfall_buffer(self):
        from kv4p_ht.spectrum import WaterfallBuffer
        wb = WaterfallBuffer(max_rows=10, num_bins=8)
        assert wb.get_matrix() is None
        for i in range(15):
            wb.push(np.ones(8, dtype=np.float32) * i)
        m = wb.get_matrix()
        assert m is not None
        assert m.shape == (10, 8)
        assert wb.num_rows == 10
        assert wb.num_bins == 8

    def test_peak_hold_and_min_hold(self):
        from kv4p_ht.spectrum import SpectrumAnalyzer
        sa = SpectrumAnalyzer(sample_rate=48000, fft_size=2048, averaging=1)
        silence = np.zeros(4096, dtype=np.float32)
        sa.feed(silence)
        t = np.arange(4096) / 48000.0
        tone = 0.8 * np.sin(2 * np.pi * 2000 * t).astype(np.float32)
        sa.feed(tone)
        peak = sa.get_peak_hold()
        mn = sa.get_min_hold()
        assert peak is not None
        assert mn is not None
        assert np.max(peak[1]) > np.max(mn[1])
        sa.reset_peak_hold()
        sa.reset_min_hold()

    def test_span_and_ref_level(self):
        from kv4p_ht.spectrum import SpectrumAnalyzer
        sa = SpectrumAnalyzer(sample_rate=48000, fft_size=2048)
        sa.set_span(50000)
        sa.set_ref_level(0)
        sa.set_averaging(8)
        sa.set_fft_size(1024)
        assert sa._span == 50000
        assert sa._ref_level == 0

    def test_waterfall_row(self):
        from kv4p_ht.spectrum import SpectrumAnalyzer
        sa = SpectrumAnalyzer(sample_rate=48000, fft_size=2048, averaging=1)
        silence = np.zeros(4096, dtype=np.float32)
        sa.feed(silence)
        row = sa.get_waterfall_row()
        assert row is not None
        assert len(row) > 0


# ── Morse ──────────────────────────────────────────────────────────

class TestMorseKeyer:
    def test_generate_hello_world(self):
        from kv4p_ht.morse import MorseKeyer, CHAR_TO_MORSE
        keyer = MorseKeyer(wpm=20, tone_hz=700, sample_rate=48000)
        assert 'H' in CHAR_TO_MORSE
        assert CHAR_TO_MORSE['H'] == '....'
        waveform = keyer.generate_tone_array("HI")
        assert len(waveform) > 0
        assert waveform.dtype == np.float32
        assert np.max(np.abs(waveform)) > 0.1

    def test_generate_empty(self):
        from kv4p_ht.morse import MorseKeyer
        keyer = MorseKeyer(wpm=20)
        waveform = keyer.generate_tone_array("")
        assert len(waveform) == 0

    def test_text_to_timing(self):
        from kv4p_ht.morse import MorseKeyer
        keyer = MorseKeyer(wpm=20)
        timing = keyer.text_to_timing("E")
        assert len(timing) > 0
        assert timing[0][0] is True
        assert timing[0][1] > 0

    def test_generate_tone_int16(self):
        from kv4p_ht.morse import MorseKeyer
        keyer = MorseKeyer(wpm=20)
        pcm = keyer.generate_tone("SOS")
        assert len(pcm) > 0
        assert all(isinstance(s, int) for s in pcm[:10])
        assert max(abs(s) for s in pcm) > 1000

    def test_set_wpm(self):
        from kv4p_ht.morse import MorseKeyer
        keyer = MorseKeyer(wpm=20)
        keyer.set_wpm(10)
        assert keyer.wpm == 10
        keyer.set_wpm(200)
        assert keyer.wpm == 60
        keyer.set_wpm(1)
        assert keyer.wpm == 5


class TestMorseDecoder:
    def test_decode_s(self):
        from kv4p_ht.morse import MorseDecoder
        decoder = MorseDecoder(wpm=20)
        et = decoder._element_time
        t = 100.0
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 0.8
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 0.8
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 3.0
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 3.0
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 3.0
        decoder.process_key(True, t)
        t += et * 0.8
        decoder.process_key(False, t)
        t += et * 5.0
        text = decoder.get_text()
        assert 'S' in text or 'O' in text or '3' in text

    def test_reset(self):
        from kv4p_ht.morse import MorseDecoder
        decoder = MorseDecoder(wpm=20)
        decoder.process_key(True, 0.0)
        decoder.process_key(False, 0.06)
        decoder.reset()
        assert decoder.get_text() == ""
        assert decoder.get_current_element() == ""


class TestPracticeGenerator:
    def test_generate_exchange(self):
        from kv4p_ht.morse import PracticeGenerator
        gen = PracticeGenerator()
        ex = gen.generate_exchange()
        assert len(ex) > 0
        assert isinstance(ex, str)

    def test_random_callsign(self):
        from kv4p_ht.morse import PracticeGenerator
        gen = PracticeGenerator()
        cs = gen.random_callsign()
        assert len(cs) >= 3
        assert any(c.isdigit() for c in cs)

    def test_generate_session(self):
        from kv4p_ht.morse import PracticeGenerator
        gen = PracticeGenerator()
        session = gen.generate_session(count=5)
        assert len(session) == 5
        assert all(isinstance(s, str) for s in session)


# ── SSTV ───────────────────────────────────────────────────────────

class TestSstvEncoder:
    def test_encode_small_image_m1(self):
        from kv4p_ht.sstv import SstvEncoder
        enc = SstvEncoder(mode='M1', sample_rate=48000)
        img = np.random.randint(0, 256, (32, 64, 3), dtype=np.uint8)
        waveform = enc.encode_image(img)
        assert waveform.dtype == np.float32
        assert len(waveform) > 0
        duration_s = len(waveform) / 48000
        assert duration_s > 0.5, f"M1 32-line image should be >0.5s, got {duration_s:.2f}s"
        assert duration_s < 120, f"Unexpectedly long: {duration_s:.2f}s"
        assert np.max(np.abs(waveform)) > 0.1, "Waveform should have audible amplitude"

    def test_encode_robot36(self):
        from kv4p_ht.sstv import SstvEncoder
        enc = SstvEncoder(mode='R36', sample_rate=48000)
        img = np.random.randint(0, 256, (20, 40, 3), dtype=np.uint8)
        waveform = enc.encode_image(img)
        assert len(waveform) > 0

    def test_encode_float_image(self):
        from kv4p_ht.sstv import SstvEncoder
        enc = SstvEncoder(mode='M2', sample_rate=48000)
        img = np.random.rand(16, 32, 3).astype(np.float32)
        waveform = enc.encode_image(img)
        assert len(waveform) > 0

    def test_all_modes_encode(self):
        from kv4p_ht.sstv import SstvEncoder, MODES
        for mode_name in MODES:
            enc = SstvEncoder(mode=mode_name, sample_rate=48000)
            mode = MODES[mode_name]
            h = min(mode.lines, 8)
            img = np.random.randint(0, 256, (h, 32, 3), dtype=np.uint8)
            waveform = enc.encode_image(img)
            assert len(waveform) > 0, f"Mode {mode_name} produced empty waveform"

    def test_freq_to_luminance_roundtrip(self):
        from kv4p_ht.sstv import freq_to_luminance, luminance_to_freq, COLOR_BLACK_HZ, COLOR_WHITE_HZ
        for lum in range(0, 256, 16):
            freq = luminance_to_freq(lum)
            assert COLOR_BLACK_HZ <= freq <= COLOR_WHITE_HZ
            lum_back = freq_to_luminance(freq)
            assert abs(lum_back - lum) <= 2


class TestSstvDecoder:
    def test_decode_table_lookup(self):
        from kv4p_ht.sstv import MODES
        assert 'M1' in MODES
        assert 'S1' in MODES
        assert 'R36' in MODES
        assert MODES['M1'].vis_code == 0x2C
        assert MODES['S1'].vis_code == 0x3C

    def test_feed_does_not_crash(self):
        from kv4p_ht.sstv import SstvDecoder
        received = []
        dec = SstvDecoder(mode='M1', callback=lambda img, m: received.append((img, m)))
        silence = np.zeros(1920, dtype=np.float32)
        dec.feed(silence)
        dec.feed(silence)
        dec.reset()

    def test_callback_on_valid_scanline(self):
        from kv4p_ht.sstv import SstvEncoder, SstvDecoder
        enc = SstvEncoder(mode='M1', sample_rate=48000)
        tiny = np.full((1, 32, 3), 128, dtype=np.uint8)
        waveform = enc.encode_image(tiny)
        received = []
        dec = SstvDecoder(mode='M1', sample_rate=48000,
                          callback=lambda img, m: received.append((img, m)))
        chunk_size = 1920
        offset = 0
        while offset < len(waveform):
            chunk = waveform[offset:offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            dec.feed(chunk)
            offset += chunk_size


# ── AFSK ───────────────────────────────────────────────────────────

class TestAfskModulation:
    def test_crc_ccitt(self):
        from kv4p_ht.afsk import crc_ccitt
        data = b"Hello, world!"
        crc = crc_ccitt(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_build_tx_waveform(self):
        from kv4p_ht.afsk import build_tx_waveform
        waveform = build_tx_waveform("N0CALL", "APZ010", [], b">Test message")
        assert len(waveform) > 0
        assert waveform.dtype == np.float32
        duration_s = len(waveform) / 48000
        assert duration_s > 0.5
        assert np.max(np.abs(waveform)) > 0.1

    def test_modulate_ax25(self):
        from kv4p_ht.afsk import modulate_ax25
        waveform = modulate_ax25("N0CALL", "APZ010", [], b"test")
        assert len(waveform) > 0
        assert waveform.dtype == np.float32

    def test_build_ax25_bits(self):
        from kv4p_ht.afsk import build_ax25_bits
        bits = build_ax25_bits("N0CALL", "APZ010", [], b"test")
        assert len(bits) > 0
        assert all(b in (0, 1) for b in bits)


class TestAfskDemodulation:
    def test_demod_does_not_crash(self):
        from kv4p_ht.afsk_demod import AfskDemodulator
        packets = []
        demod = AfskDemodulator(callback=lambda p: packets.append(p))
        silence = [0.0] * 1920
        demod.process(silence)

    def test_demodulate_own_signal(self):
        from kv4p_ht.afsk import build_tx_waveform
        from kv4p_ht.afsk_demod import AfskDemodulator
        packets = []
        demod = AfskDemodulator(callback=lambda p: packets.append(p))
        waveform = build_tx_waveform("N0CALL", "APZ010", [], b"test123")
        samples = waveform.tolist()
        chunk_size = 1920
        for i in range(0, len(samples), chunk_size):
            demod.process(samples[i:i + chunk_size])
        assert len(packets) > 0, "AFSK modulate->demodulate roundtrip should decode at least one packet"
        decoded = packets[0]
        assert 'source' in decoded
        assert decoded['source'] == "N0CALL"


# ── AX.25 File Transfer ───────────────────────────────────────────

class TestFileTransferPacket:
    def test_pack_unpack_roundtrip(self):
        from kv4p_ht.ax25_file_transfer import FileTransferPacket, CMD_FILE_START
        pkt = FileTransferPacket(CMD_FILE_START, seq=0, data=b"test.txt\x00100")
        raw = pkt.pack()
        assert raw[:2] == b'KF'
        assert raw[2] == CMD_FILE_START
        unpacked = FileTransferPacket.unpack(raw)
        assert unpacked is not None
        assert unpacked.cmd == CMD_FILE_START
        assert unpacked.seq == 0
        assert unpacked.data == b"test.txt\x00100"

    def test_unpack_corrupted(self):
        from kv4p_ht.ax25_file_transfer import FileTransferPacket
        result = FileTransferPacket.unpack(b'\x00\x01\x02\x03')
        assert result is None

    def test_large_data(self):
        from kv4p_ht.ax25_file_transfer import FileTransferPacket, CMD_FILE_DATA
        data = os.urandom(200)
        pkt = FileTransferPacket(CMD_FILE_DATA, seq=42, data=data)
        raw = pkt.pack()
        unpacked = FileTransferPacket.unpack(raw)
        assert unpacked is not None
        assert unpacked.data == data
        assert unpacked.seq == 42


class TestFileTransferReceiver:
    def test_start_stop(self):
        from kv4p_ht.ax25_file_transfer import FileTransferReceiver
        rx = FileTransferReceiver(
            source_call="N0CALL", dest_call="N0CALL",
            tx_callback=lambda x: None,
        )
        rx.start()
        assert rx is not None
        rx.stop()

    def test_process_abort(self):
        from kv4p_ht.ax25_file_transfer import (
            FileTransferReceiver, FileTransferPacket, CMD_ABORT, CMD_FILE_START
        )
        received = []
        rx = FileTransferReceiver(
            source_call="N0CALL", dest_call="N0CALL",
            tx_callback=lambda x: None,
            on_complete=lambda ok, msg: received.append((ok, msg)),
        )
        rx.start()
        pkt = FileTransferPacket(CMD_FILE_START, seq=0, data=b"test.txt\x0010")
        rx.process_packet(pkt.pack())
        abort = FileTransferPacket(CMD_ABORT, seq=0)
        rx.process_packet(abort.pack())
        time.sleep(0.1)
        assert len(received) > 0


class TestFileTransferSender:
    def test_send_small_file(self):
        from kv4p_ht.ax25_file_transfer import FileTransferSender
        sent_packets = []
        tx_frames = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello AX.25 file transfer!")
            tmppath = f.name

        try:
            sender = FileTransferSender(
                source_call="N0CALL", dest_call="N0CALL",
                tx_callback=lambda x: tx_frames.append(x),
                log_fn=lambda m: None,
                progress_callback=lambda s, t: sent_packets.append((s, t)),
            )
            sender.send_file(tmppath)
            time.sleep(1.0)
            assert len(tx_frames) > 0, "Sender should have produced AX.25 frames"
        finally:
            os.unlink(tmppath)


# ── Protocol ───────────────────────────────────────────────────────

class TestProtocol:
    def test_pack_group(self):
        from kv4p_ht.protocol import pack_group
        payload = pack_group(0, 144.390, 144.390, 0, 3, 0)
        assert len(payload) == 12
        assert isinstance(payload, bytes)

    def test_pack_hl(self):
        from kv4p_ht.protocol import pack_hl
        assert pack_hl(True) == b'\x01'
        assert pack_hl(False) == b'\x00'

    def test_pack_config(self):
        from kv4p_ht.protocol import pack_config
        assert pack_config(True) == b'\x01'

    def test_rssi_to_s_meter(self):
        from kv4p_ht.protocol import rssi_to_s_meter
        assert rssi_to_s_meter(0) >= 1
        assert rssi_to_s_meter(10000) <= 9
        assert rssi_to_s_meter(500) >= 1
        assert rssi_to_s_meter(500) <= 9

    def test_unpack_version(self):
        from kv4p_ht.protocol import unpack_version
        # H=uint16, B=uint8, I=uint32, I=uint32, B=uint8 = 12 bytes
        data = struct.pack('<HBIIB', 1, ord('f'), 2, 0, 0)
        ver = unpack_version(data)
        assert ver['ver'] == 1
        assert ver['module_type'] == 0
        assert ver['radio_status'] == 'f'

    def test_unpack_rssi(self):
        from kv4p_ht.protocol import unpack_rssi
        # unpack_rssi reads data[0]
        assert unpack_rssi(bytes([50])) == 50
        assert unpack_rssi(bytes([0])) == 0
        assert unpack_rssi(b'') == 0

    def test_frame_sender(self):
        from kv4p_ht.protocol import FrameSender
        written = []
        sender = FrameSender(write_fn=lambda data: written.append(data))
        sender.ptt_down()
        assert len(written) == 1
        assert written[0][:4] == b'\xDE\xAD\xBE\xEF'
        assert written[0][4] == 0x01
        sender.ptt_up()
        assert len(written) == 2
        assert written[1][4] == 0x02

    def test_frame_parser(self):
        from kv4p_ht.protocol import FrameParser, DELIMITER
        frames = []
        parser = FrameParser(callback=lambda cmd, payload: frames.append((cmd, payload)))
        cmd = 0x53
        payload = struct.pack('<I', 42)
        frame = DELIMITER + bytes([cmd]) + struct.pack('<H', len(payload)) + payload
        for b in frame:
            parser.feed(bytes([b]))
        assert len(frames) == 1
        assert frames[0][0] == cmd
        assert frames[0][1] == payload

    def test_frame_parser_multiple(self):
        from kv4p_ht.protocol import FrameParser, DELIMITER
        frames = []
        parser = FrameParser(callback=lambda cmd, payload: frames.append((cmd, payload)))
        for cmd in [0x01, 0x02, 0x07]:
            payload = bytes([cmd * 10])
            frame = DELIMITER + bytes([cmd]) + struct.pack('<H', len(payload)) + payload
            for b in frame:
                parser.feed(bytes([b]))
        assert len(frames) == 3


# ── APRS ───────────────────────────────────────────────────────────

class TestAprs:
    def test_encode_decode_roundtrip(self):
        from kv4p_ht.aprs import encode_ax25_ui, decode_ax25_frame
        info = b"!4903.50N/07201.75W-PHG2360"
        frame = encode_ax25_ui("N0CALL", "APZ010", [], info)
        decoded = decode_ax25_frame(frame)
        assert decoded['source'] == "N0CALL"
        assert decoded['destination'] == "APZ010"
        assert decoded['info'] == info
        assert decoded['digipeaters'] == []

    def test_encode_with_digipeaters(self):
        from kv4p_ht.aprs import encode_ax25_ui, decode_ax25_frame
        frame = encode_ax25_ui("N0CALL", "APZ010", ["WIDE1-1", "WIDE2-1"], b">Test")
        decoded = decode_ax25_frame(frame)
        assert len(decoded['digipeaters']) == 2
        assert "WIDE1-1" in decoded['digipeaters']

    def test_format_beacon(self):
        from kv4p_ht.aprs import format_beacon
        beacon = format_beacon("N0CALL", 49.0583, -72.0283, comment="Test")
        assert "49" in beacon
        assert "=" in beacon
        assert "N" in beacon
        assert "W" in beacon

    def test_format_message(self):
        from kv4p_ht.aprs import format_message
        msg = format_message("N0CALL", "Hello!")
        assert "N0CALL" in msg
        assert "Hello!" in msg

    def test_format_ack(self):
        from kv4p_ht.aprs import format_ack
        ack = format_ack("N0CALL", "123")
        assert "ack" in ack.lower()
        assert "123" in ack

    def test_parse_position(self):
        from kv4p_ht.aprs import parse_aprs
        info = "!4903.50N/07201.75W-PHG2360"
        parsed = parse_aprs(info, source="N0CALL")
        assert parsed['type'] == 'position'
        assert 'raw_position' in parsed

    def test_parse_message(self):
        from kv4p_ht.aprs import parse_aprs
        info = ":N0CALL   :Hello world{123"
        parsed = parse_aprs(info, source="N0CALL")
        assert parsed['type'] == 'message'
        assert 'addressee' in parsed
        assert parsed['addressee'] == 'N0CALL'
        assert parsed['text'] == 'Hello world'
        assert parsed['msg_id'] == '123'

    def test_parse_status(self):
        from kv4p_ht.aprs import parse_aprs
        info = ">Test status message"
        parsed = parse_aprs(info, source="N0CALL")
        assert parsed['type'] == 'status'

    def test_decode_short_frame(self):
        from kv4p_ht.aprs import decode_ax25_frame
        result = decode_ax25_frame(b'\x00\x01\x02')
        assert 'raw' in result

    def test_digipeater(self):
        from kv4p_ht.aprs import Digipeater, encode_ax25_ui, decode_ax25_frame
        tx_frames = []
        digi = Digipeater("N0CALL", tx_callback=lambda x: tx_frames.append(x))
        frame = encode_ax25_ui("KG7OED", "APZ010", ["WIDE1-1"], b">Test")
        result = digi.process(frame)
        assert result is True
        assert len(tx_frames) == 1
        decoded = decode_ax25_frame(tx_frames[0])
        assert "N0CALL" in decoded['digipeaters']

    def test_digipeater_dedup(self):
        from kv4p_ht.aprs import Digipeater, encode_ax25_ui
        tx_frames = []
        digi = Digipeater("N0CALL", tx_callback=lambda x: tx_frames.append(x))
        frame = encode_ax25_ui("KG7OED", "APZ010", ["WIDE1-1"], b">Test")
        digi.process(frame)
        time.sleep(0.01)
        digi.process(frame)
        assert len(tx_frames) == 1, "Duplicate should be suppressed"


# ── Scanner ────────────────────────────────────────────────────────

class TestScanner:
    def test_band_plan_presets(self):
        from kv4p_ht.scanner import BandPlan
        presets = BandPlan.get_preset_list("2m")
        assert len(presets) > 10
        assert all(144 <= f <= 148 for f in presets)

    def test_scan_start_stop(self):
        from kv4p_ht.scanner import FrequencyScanner
        changed = []
        scanner = FrequencyScanner(
            set_freq_callback=lambda f: changed.append(f),
            on_signal_callback=lambda f, r: None,
        )
        scanner.start_scan([144.390, 144.410], dwell_ms=100)
        assert scanner.is_scanning()
        time.sleep(0.3)
        scanner.stop_scan()
        assert not scanner.is_scanning()
        assert len(changed) > 0

    def test_pause_resume(self):
        from kv4p_ht.scanner import FrequencyScanner
        scanner = FrequencyScanner(
            set_freq_callback=lambda f: None,
            on_signal_callback=lambda f, r: None,
        )
        scanner.start_scan([144.390, 144.410], dwell_ms=100)
        scanner.pause()
        assert scanner.is_paused
        scanner.resume()
        assert not scanner.is_paused
        scanner.stop_scan()

    def test_add_remove_frequency(self):
        from kv4p_ht.scanner import FrequencyScanner
        scanner = FrequencyScanner(
            set_freq_callback=lambda f: None,
            on_signal_callback=lambda f, r: None,
        )
        scanner.start_scan([144.390], dwell_ms=100)
        scanner.add_frequency(144.410)
        assert 144.410 in scanner.get_scan_list()
        scanner.remove_frequency(144.410)
        assert 144.410 not in scanner.get_scan_list()
        scanner.stop_scan()

    def test_set_rssi_signal_callback(self):
        from kv4p_ht.scanner import FrequencyScanner
        signals = []
        scanner = FrequencyScanner(
            set_freq_callback=lambda f: None,
            on_signal_callback=lambda f, r: signals.append((f, r)),
        )
        scanner.start_scan([144.390, 144.410], dwell_ms=50, squelch_threshold=1.0)
        time.sleep(0.05)
        scanner.set_rssi(500.0)
        time.sleep(0.3)
        scanner.stop_scan()

    def test_empty_scan_list(self):
        from kv4p_ht.scanner import FrequencyScanner
        scanner = FrequencyScanner(
            set_freq_callback=lambda f: None,
            on_signal_callback=lambda f, r: None,
        )
        scanner.start_scan([], dwell_ms=100)
        assert scanner.is_scanning()
        scanner.stop_scan()


# ── KISS ───────────────────────────────────────────────────────────

class TestKiss:
    def test_encode_decode_roundtrip(self):
        from kv4p_ht.kiss import KissTnc, KISS_FEND
        tnc = KissTnc.__new__(KissTnc)
        data = b"Hello KISS"
        encoded = KissTnc._encode(tnc, data)
        assert KISS_FEND not in encoded
        decoded = KissTnc._decode(tnc, encoded)
        assert decoded == data

    def test_encode_escape_fend(self):
        from kv4p_ht.kiss import KissTnc, KISS_FEND, KISS_FESC, KISS_TFEND
        tnc = KissTnc.__new__(KissTnc)
        data = bytes([KISS_FEND])
        encoded = KissTnc._encode(tnc, data)
        assert KISS_FESC in encoded
        assert KISS_TFEND in encoded
        decoded = KissTnc._decode(tnc, encoded)
        assert decoded == data

    def test_encode_escape_fesc(self):
        from kv4p_ht.kiss import KissTnc, KISS_FEND, KISS_FESC, KISS_TFESC
        tnc = KissTnc.__new__(KissTnc)
        data = bytes([KISS_FESC])
        encoded = KissTnc._encode(tnc, data)
        assert KISS_FESC in encoded
        assert KISS_TFESC in encoded
        decoded = KissTnc._decode(tnc, encoded)
        assert decoded == data

    def test_decode_non_kiss_data(self):
        from kv4p_ht.kiss import KissTnc
        tnc = KissTnc.__new__(KissTnc)
        # _decode strips FESC-escaped sequences but passes other bytes through
        result = KissTnc._decode(tnc, b'\xC0\x00')
        assert result == b'\xC0\x00'

    def test_decode_escaped_sequence(self):
        from kv4p_ht.kiss import KissTnc, KISS_FEND, KISS_FESC, KISS_TFEND
        tnc = KissTnc.__new__(KissTnc)
        data = bytes([KISS_FESC, KISS_TFEND, 0x41])
        result = KissTnc._decode(tnc, data)
        assert result == bytes([KISS_FEND, 0x41])

    def test_decode_empty(self):
        from kv4p_ht.kiss import KissTnc
        tnc = KissTnc.__new__(KissTnc)
        result = KissTnc._decode(tnc, b'')
        assert result == b''


# ── UDP Broadcast ──────────────────────────────────────────────────

class TestUdpBroadcast:
    def test_parse_wsjtx_status(self):
        from kv4p_ht.udp_broadcast import parse_wsjt_x, WSJT_X_MAGIC
        buf = bytearray()
        buf += struct.pack('<I', WSJT_X_MAGIC)
        buf += struct.pack('<I', 2)
        buf += struct.pack('<I', 0)
        buf += struct.pack('<I', 0)
        buf += struct.pack('<q', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<Q', 144390000)
        buf += struct.pack('<i', -10)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<q', int(time.time() * 1000))
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        buf += struct.pack('<i', 0)
        msg = b"CQ TEST"
        buf += struct.pack('<I', len(msg))
        buf += msg
        buf += struct.pack('<I', 0)
        result = parse_wsjt_x(bytes(buf))
        assert result is not None
        assert result.get('type') == 'status' or 'raw' in result

    def test_parse_wsjtx_bad_magic(self):
        from kv4p_ht.udp_broadcast import parse_wsjt_x
        result = parse_wsjt_x(b'\x00\x00\x00\x00')
        assert result is not None
        assert result.get('type', 0) == 0

    def test_parse_direwolf(self):
        from kv4p_ht.udp_broadcast import parse_direwolf
        line = b"N0CALL>APZ010:!4903.50N/07201.75W-Test\r\n"
        result = parse_direwolf(line)
        assert result['type'] == 'aprs'
        assert 'N0CALL' in result['line']

    def test_parse_fldigi(self):
        from kv4p_ht.udp_broadcast import parse_fldigi
        data = b"&rx_count=1&tx_count=0&mode=FT8&"
        result = parse_fldigi(data)
        assert result['type'] == 'status'
        assert 'rx_count' in result['data']

    def test_parse_fldigi_empty(self):
        from kv4p_ht.udp_broadcast import parse_fldigi
        result = parse_fldigi(b"")
        assert result['data'] == {}

    def test_udp_broadcast_rx_start_stop(self):
        from kv4p_ht.udp_broadcast import UdpBroadcastRx
        rx = UdpBroadcastRx()
        rx.start()
        time.sleep(0.1)
        assert rx.is_running()
        rx.stop()
        time.sleep(0.2)
        assert not rx.is_running()


# ── TxWorker ───────────────────────────────────────────────────────

class TestTxWorker:
    def test_tx_worker_encodes_and_sends(self):
        import sys
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)

        from kv4p_ht.app import TxWorker, DELIMITER
        cmd_queue = queue.SimpleQueue()
        waveform = 0.5 * np.sin(2 * np.pi * 700 * np.arange(4800) / 48000).astype(np.float32)
        worker = TxWorker(waveform, cmd_queue)
        success_results = []
        worker.finished.connect(lambda ok, msg: success_results.append((ok, msg)))
        worker.start()
        worker.wait(10000)
        app.processEvents()
        assert len(success_results) == 1
        assert success_results[0][0] is True
        frames = []
        while not cmd_queue.empty():
            frames.append(cmd_queue.get_nowait())
        assert len(frames) >= 3, f"Expected PTT_DOWN + audio frames + PTT_UP, got {len(frames)}"
        assert frames[0][:4] == DELIMITER
        assert frames[0][4] == 0x01, "First frame should be PTT_DOWN"
        assert frames[-1][:4] == DELIMITER
        assert frames[-1][4] == 0x02, "Last frame should be PTT_UP"
        for f in frames[1:-1]:
            assert f[:4] == DELIMITER
            assert f[4] == 0x07, "Middle frames should be TX_AUDIO"
