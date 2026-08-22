from __future__ import annotations

import socket
import struct
import time

from kv4p_ht.udp_broadcast import (
    WSJT_X_MAGIC,
    UdpBroadcastRx,
    parse_direwolf,
    parse_fldigi,
    parse_wsjt_x,
)


def _wj_status(id_str, mode, freq_hz, snr):
    parts = [struct.pack('<II', WSJT_X_MAGIC, 1),
             id_str.encode() + b'\x00',
             mode.encode() + b'\x00',
             struct.pack('<Q', freq_hz),
             struct.pack('<i', snr),
             struct.pack('<i', 1),
             struct.pack('<i', 0),
             struct.pack('<i', 1),
             struct.pack('<Q', freq_hz),
             struct.pack('<i', 0)]
    return b''.join(parts)


def _wj_decode(id_str, message):
    parts = [struct.pack('<II', WSJT_X_MAGIC, 2),
             id_str.encode() + b'\x00',
             struct.pack('<I', 123456),
             struct.pack('<i', -12),
             struct.pack('<d', 0.4),
             struct.pack('<i', 500),
             message.encode() + b'\x00',
             b'\x00']
    return b''.join(parts)


class TestParseWsjtx:
    def test_status_packet(self):
        data = _wj_status("WJ-123", "FT8", 144_390_000, 7)
        r = parse_wsjt_x(data)
        assert r['type'] == 1
        assert r['id'] == "WJ-123"
        assert r['mode'] == "FT8"
        assert r['frequency'] == 144_390_000
        assert r['snr'] == 7

    def test_decode_packet(self):
        data = _wj_decode("WJ-123", "K1ABC W2DEF FN42")
        r = parse_wsjt_x(data)
        assert r['type'] == 2
        assert r['snr'] == -12
        assert r['message'] == "K1ABC W2DEF FN42"

    def test_bad_magic(self):
        data = struct.pack('<II', 0xDEADBEEF, 1)
        r = parse_wsjt_x(data)
        assert r['type'] == 0

    def test_short_data(self):
        r = parse_wsjt_x(b'\x01\x02')
        assert r['type'] == 0


class TestParseDirewolf:
    def test_line(self):
        r = parse_direwolf(b"K1ABC>APZ010:>hello\r\n")
        assert r['type'] == 'aprs'
        assert 'hello' in r['line']


class TestParseFldigi:
    def test_fields(self):
        data = b"rig=KX3&freq=144.390&mode=FM"
        r = parse_fldigi(data)
        assert r['type'] == 'status'
        assert r['data']['rig'] == 'KX3'
        assert r['data']['freq'] == '144.390'
        assert r['data']['mode'] == 'FM'

    def test_empty(self):
        r = parse_fldigi(b"")
        assert r['data'] == {}


class TestUdpBroadcastRx:
    def test_rx_roundtrip(self):
        rx = UdpBroadcastRx()
        got = []
        rx.set_callback('wsjt-x', lambda name, pkt: got.append(pkt))
        rx.start()
        time.sleep(0.2)

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = _wj_status("WJ-1", "FT8", 145_000_000, 3)
        s.sendto(payload, ("127.0.0.1", 2237))
        s.close()

        deadline = time.time() + 2
        while not got and time.time() < deadline:
            time.sleep(0.01)
        rx.stop()
        assert got, "expected to receive a WSJT-X packet"
        assert got[0]['frequency'] == 145_000_000

    def test_restart_after_stop(self):
        rx = UdpBroadcastRx()
        rx.start()
        time.sleep(0.2)
        rx.stop()
        rx.start()
        time.sleep(0.2)
        rx.stop()
        assert rx.is_running() is False

    def test_is_running(self):
        rx = UdpBroadcastRx()
        assert rx.is_running() is False
        rx.start()
        time.sleep(0.2)
        assert rx.is_running() is True
        rx.stop()
