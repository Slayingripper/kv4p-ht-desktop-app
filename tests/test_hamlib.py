from __future__ import annotations

import socket
import threading
import time

import pytest

from kv4p_ht.hamlib import RigCtlD


class TestHandleResponse:
    def test_frequency(self):
        r = RigCtlD()
        r._handle_response("144390000")
        assert r.get_frequency() == pytest.approx(144.39)

    def test_frequency_callback(self):
        seen = []
        r = RigCtlD(freq_callback=lambda f: seen.append(f))
        r._handle_response("145000000")
        r._handle_response("145000000")
        assert seen == [145.0]
        r._handle_response("145001000")
        assert seen == [145.0, 145.001]

    def test_ptt(self):
        seen = []
        r = RigCtlD(ptt_callback=lambda on: seen.append(on))
        r._handle_response("0")
        r._handle_response("1")
        r._handle_response("1")
        assert seen == [False, True]
        assert r.get_ptt() is True

    def test_mode(self):
        seen = []
        r = RigCtlD(mode_callback=lambda m: seen.append(m))
        r._handle_response("FM")
        r._handle_response("FM")
        r._handle_response("USB")
        assert seen == ["FM", "USB"]
        assert r.get_mode() == "USB"

    def test_squelch_and_strength(self):
        r = RigCtlD()
        r._handle_response("L RIG_SQL 3")
        r._handle_response("0 L RIG_STRENGTH")
        assert r.get_squelch() == 3
        assert r.get_strength() == 0

    def test_empty_line_ignored(self):
        r = RigCtlD()
        r._handle_response("")
        assert r.get_frequency() is None
        assert r.get_ptt() is None


class TestCommands:
    def test_set_frequency_requires_connection(self):
        r = RigCtlD()
        assert r.set_frequency(144.39) is False

    def test_send_commands_over_socket(self):
        srv, cli = socket.socketpair()
        r = RigCtlD()
        with r._lock:
            r._sock = cli
        assert r.set_frequency(144.39) is True
        assert r.set_ptt(True) is True
        assert r.set_mode("FM") is True
        assert r.set_squelch(3) is True
        srv.settimeout(2)
        buf = bytearray()
        while len(buf.split(b"\n")) < 5:
            buf += srv.recv(64)
        lines = [l.decode().strip() for l in buf.split(b"\n") if l]
        assert lines == ["F 144390000", "T 1", "M FM 0", "L RIG_SQL 3"]
        srv.close()
        cli.close()

    def test_poll_without_connection(self):
        r = RigCtlD()
        assert r.poll() is None


class TestRecvLine:
    def test_lf_terminated(self):
        srv, cli = socket.socketpair()
        srv.sendall(b"FM\n")
        r = RigCtlD()
        r._running = True
        assert r._recv_line(cli) == "FM"
        srv.close()
        cli.close()

    def test_crlf_terminated(self):
        srv, cli = socket.socketpair()
        srv.sendall(b"1\r\n")
        r = RigCtlD()
        r._running = True
        assert r._recv_line(cli) == "1"
        srv.close()
        cli.close()

    def test_closed_socket_returns_none(self):
        srv, cli = socket.socketpair()
        srv.close()
        r = RigCtlD()
        r._running = True
        assert r._recv_line(cli) is None
        cli.close()


class TestLogFn:
    def test_constructor_accepts_log_fn(self):
        logs = []
        r = RigCtlD(log_fn=logs.append)
        r._log("hello")
        assert logs == ["hello"]


class TestTcpConnect:
    def test_connect_and_poll(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def rig_server():
            conn, _ = listener.accept()
            conn.settimeout(5)
            try:
                conn.recv(64)
                conn.sendall(b"145000000\n")
                conn.recv(64)
                conn.sendall(b"FM\n")
                conn.recv(64)
                conn.sendall(b"0\n")
                time.sleep(0.2)
            finally:
                conn.close()

        t = threading.Thread(target=rig_server, daemon=True)
        t.start()

        r = RigCtlD(freq_callback=lambda f: None)
        assert r.connect(host="127.0.0.1", port=port) is True
        time.sleep(0.5)
        assert r.get_frequency() is not None
        assert r.get_mode() == "FM"
        assert r.get_ptt() is False
        r.disconnect()
        listener.close()

    def test_connect_failure_returns_false(self):
        r = RigCtlD()
        assert r.connect(host="127.0.0.1", port=1) is False
        r.disconnect()
