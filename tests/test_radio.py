from __future__ import annotations

import queue

from kv4p_ht.protocol import (
    DELIMITER,
    EspCmd,
    HostCmd,
    pack_config,
    pack_group,
    pack_hl,
)
from kv4p_ht.radio import SerialWorker, find_kv4p_port


class FakePort:
    def __init__(self, vid, pid, device):
        self.vid = vid
        self.pid = pid
        self.device = device


class TestFindPort:
    def test_finds_known_device(self, monkeypatch):
        monkeypatch.setattr(
            "kv4p_ht.radio.list_ports.comports",
            lambda: [FakePort(0x1A86, 0x7523, "/dev/ttyUSB0")],
        )
        monkeypatch.setattr("kv4p_ht.radio._CACHED_PORT", None)
        monkeypatch.setattr("kv4p_ht.radio._CACHED_PORT_TTL", 0)
        assert find_kv4p_port() == "/dev/ttyUSB0"

    def test_no_device_returns_none(self, monkeypatch):
        monkeypatch.setattr("kv4p_ht.radio.list_ports.comports", list)
        monkeypatch.setattr("kv4p_ht.radio._CACHED_PORT", None)
        monkeypatch.setattr("kv4p_ht.radio._CACHED_PORT_TTL", 0)
        assert find_kv4p_port() is None


class TestSerialWorkerDispatch:
    def make_worker(self):
        w = SerialWorker()
        w.hello_received = _SigSpy()
        w.version_received = _SigSpy()
        w.smeter = _SigSpy()
        w.ax25_packet = _SigSpy()
        w.window_update = _SigSpy()
        w.debug_msg = _SigSpy()
        return w

    def test_hello(self):
        w = self.make_worker()
        w._on_esp_command(EspCmd.HELLO, b"")
        assert w.hello_received.emitted == 1

    def test_version(self):
        import struct
        w = self.make_worker()
        payload = struct.pack('<H B I I B', 12, ord('f'), 2048, 0, 0)
        w._on_esp_command(EspCmd.VERSION, payload)
        assert len(w.version_received.emits) == 1
        info = w.version_received.emits[0]
        assert info['ver'] == 12
        assert info['radio_status'] == 'f'

    def test_smeter(self):
        w = self.make_worker()
        w._on_esp_command(EspCmd.SMETER_REPORT, b'\x64')
        assert w.smeter.emits == [100]

    def test_ax25_packet(self):
        w = self.make_worker()
        w._on_esp_command(EspCmd.RX_AX25_PACKET, b'\x00\xAA\xBB')
        assert w.ax25_packet.emits == [(0, b'\xAA\xBB')]

    def test_window_update(self):
        w = self.make_worker()
        import struct
        w._on_esp_command(EspCmd.WINDOW_UPDATE, struct.pack('<I', 100))
        assert w.window_update.emits == [2148]

    def test_debug_message(self):
        w = self.make_worker()
        w._on_esp_command(EspCmd.DEBUG_WARN, b"temp high")
        assert w.debug_msg.emits == [(3, "temp high")]

    def test_rx_audio_queued_when_attached(self):
        w = self.make_worker()
        q = queue.SimpleQueue()
        w.rx_opus_queue = q
        w._on_esp_command(EspCmd.RX_AUDIO, b"\x01\x02")
        assert q.get_nowait() == b"\x01\x02"

    def test_rx_audio_ignored_without_queue(self):
        w = self.make_worker()
        w.rx_opus_queue = None
        w._on_esp_command(EspCmd.RX_AUDIO, b"\x01\x02")


class _SigSpy:
    def __init__(self):
        self.emitted = 0
        self.emits = []

    def emit(self, *args):
        self.emitted += 1
        self.emits.append(args[0] if len(args) == 1 else args)


class TestFrameBuilding:
    def test_group_frame_bytes(self):
        payload = pack_group(0, 144.39, 144.39, 0, 3, 0)
        frame = bytearray(DELIMITER + bytes([HostCmd.GROUP]) +
                          len(payload).to_bytes(2, 'little') + payload)
        assert frame[:4] == DELIMITER
        assert frame[4] == HostCmd.GROUP
        assert int.from_bytes(frame[5:7], 'little') == len(payload)

    def test_config_and_hl_pack(self):
        assert pack_config(True) == b'\x01'
        assert pack_hl(False) == b'\x00'
