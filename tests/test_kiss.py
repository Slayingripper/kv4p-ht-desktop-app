from __future__ import annotations

import socket
import threading
import time

from kv4p_ht.kiss import (
    COMMAND_DATA,
    KISS_FEND,
    KISS_FESC,
    KISS_TFEND,
    KISS_TFESC,
    KissTnc,
)


class TestEncodeDecode:
    def test_roundtrip_empty(self):
        k = KissTnc()
        assert k._decode(k._encode(b"")) == b""

    def test_roundtrip_plain(self):
        k = KissTnc()
        data = b"hello world"
        assert k._decode(k._encode(data)) == data

    def test_escape_fend(self):
        k = KissTnc()
        enc = k._encode(bytes([KISS_FEND]))
        assert enc == bytes([KISS_FESC, KISS_TFEND])

    def test_escape_fesc(self):
        k = KissTnc()
        enc = k._encode(bytes([KISS_FESC]))
        assert enc == bytes([KISS_FESC, KISS_TFESC])

    def test_roundtrip_with_control_bytes(self):
        k = KissTnc()
        data = bytes([KISS_FEND, 0x01, KISS_FESC, 0x02, KISS_FEND])
        assert k._decode(k._encode(data)) == data


class TestParseFrame:
    def test_data_frame(self):
        k = KissTnc()
        body = b"\xAA\xBB\xCC"
        frame = bytes([COMMAND_DATA]) + body
        assert k._parse_frame(frame) == body

    def test_escaped_data_frame(self):
        k = KissTnc()
        body = bytes([KISS_FEND, KISS_FESC])
        escaped = k._encode(body)
        frame = bytes([COMMAND_DATA]) + escaped
        assert k._parse_frame(frame) == body

    def test_unknown_command_ignored(self):
        k = KissTnc()
        assert k._parse_frame(bytes([0x02, 0x01, 0x02])) is None

    def test_empty_frame_ignored(self):
        k = KissTnc()
        assert k._parse_frame(b"") is None


class TestSendAx25:
    def test_tcp_send(self):
        srv, cli = socket.socketpair()
        k = KissTnc(host="localhost", tcp_port=8001)
        with k._lock:
            k._sock = cli
        body = b"\x03\xF0\x00"
        k.send_ax25(body)
        srv.settimeout(2)
        raw = bytearray()
        while len(raw) < 3:
            raw += srv.recv(64)
        assert raw[0] == KISS_FEND
        assert raw[-1] == KISS_FEND
        assert k._parse_frame(raw[1:-1]) == body
        srv.close()
        cli.close()

    def test_send_not_connected(self):
        k = KissTnc(host="localhost", tcp_port=8001)
        k.send_ax25(b"\x01")  # should not raise


class TestReader:
    def test_receives_data_frames(self):
        srv, cli = socket.socketpair()
        received = []
        k = KissTnc(host="localhost", tcp_port=8001,
                    callback=received.append)
        with k._lock:
            k._sock = cli
        body = b"\xAA\xBB"
        frame = bytes([KISS_FEND, COMMAND_DATA]) + body + bytes([KISS_FEND])
        srv.sendall(frame)
        k._running = True
        t = threading.Thread(target=k._reader, daemon=True)
        t.start()
        deadline = time.time() + 2
        while not received and time.time() < deadline:
            time.sleep(0.01)
        k._running = False
        t.join(timeout=1)
        assert received == [body]
        srv.close()
        cli.close()

    def test_multiple_frames_in_stream(self):
        srv, cli = socket.socketpair()
        received = []
        k = KissTnc(host="localhost", tcp_port=8001,
                    callback=received.append)
        with k._lock:
            k._sock = cli
        srv.sendall(b"\xC0\x00" + b"AAA" + b"\xC0")
        srv.sendall(b"\xC0\x00" + b"BBB" + b"\xC0")
        k._running = True
        t = threading.Thread(target=k._reader, daemon=True)
        t.start()
        deadline = time.time() + 2
        while len(received) < 2 and time.time() < deadline:
            time.sleep(0.01)
        k._running = False
        t.join(timeout=1)
        assert received == [b"AAA", b"BBB"]
        srv.close()
        cli.close()


class TestTcpConnect:
    def test_connect_roundtrip(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        conn_holder = []

        def server():
            conn, _ = listener.accept()
            conn_holder.append(conn)
            data = conn.recv(64)
            conn.sendall(data)

        t = threading.Thread(target=server, daemon=True)
        t.start()

        received = []
        k = KissTnc(host="127.0.0.1", tcp_port=port, callback=received.append)
        assert k.connect() is True
        k.start()
        time.sleep(0.3)
        assert k.is_connected() is True
        body = b"\xDE\xAD\xBE\xEF"
        k.send_ax25(body)
        time.sleep(0.3)
        assert received == [body]
        k.stop()
        k.disconnect()
        listener.close()

    def test_connect_failure(self):
        k = KissTnc(host="127.0.0.1", tcp_port=1)
        assert k.connect() is False
        k.disconnect()


class TestLogFn:
    def test_log_fn_called(self):
        logs = []
        k = KissTnc(log_fn=logs.append)
        k._log("hi")
        assert logs == ["hi"]
