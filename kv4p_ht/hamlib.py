from __future__ import annotations

import re
import socket
import threading
import time
from collections.abc import Callable


class RigCtlD:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 4532,
        poll_interval: float = 0.5,
        freq_callback: Callable[[float], None] | None = None,
        ptt_callback: Callable[[bool], None] | None = None,
        mode_callback: Callable[[str], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self._host = host
        self._port = port
        self._log_fn = log_fn
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._poll_interval = poll_interval

        self._freq: float | None = None
        self._ptt: bool | None = None
        self._mode: str | None = None
        self._squelch: int | None = None
        self._strength: int | None = None

        self._freq_callback = freq_callback
        self._ptt_callback = ptt_callback
        self._mode_callback = mode_callback

        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)

    def connect(self, host: str = "localhost", port: int = 4532) -> bool:
        self.disconnect()
        self._host = host
        self._port = port
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._wait_connected(timeout=5.0)

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close_socket()

    def get_frequency(self) -> float | None:
        with self._lock:
            return self._freq

    def set_frequency(self, mhz: float) -> bool:
        hz = round(mhz * 1_000_000)
        return self._cmd(f"F {hz}\n")

    def get_ptt(self) -> bool | None:
        with self._lock:
            return self._ptt

    def set_ptt(self, on: bool) -> bool:
        return self._cmd(f"T {1 if on else 0}\n")

    def get_mode(self) -> str | None:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str, passband: int = 0) -> bool:
        """Set operating mode (FM, USB, LSB, AM, CW, etc.).
        passband=0 uses rigctld default for the mode."""
        return self._cmd(f"M {mode} {passband}\n")

    def set_mode_fm(self) -> bool:
        return self.set_mode("FM")

    def set_mode_usb(self) -> bool:
        return self.set_mode("USB")

    def set_mode_lsb(self) -> bool:
        return self.set_mode("LSB")

    def set_mode_am(self) -> bool:
        return self.set_mode("AM")

    def set_mode_cw(self) -> bool:
        return self.set_mode("CW")

    def get_mode_with_passband(self) -> tuple[str | None, int | None]:
        """Get current mode and passband width."""
        with self._lock:
            return self._mode, getattr(self, '_passband', None)

    def get_squelch(self) -> int | None:
        with self._lock:
            return self._squelch

    def set_squelch(self, val: int) -> bool:
        return self._cmd(f"L RIG_SQL {val}\n")

    def get_strength(self) -> int | None:
        with self._lock:
            return self._strength

    def poll(self) -> dict | None:
        with self._lock:
            if self._sock is None:
                return None
            return {
                "freq": self._freq,
                "mode": self._mode,
                "ptt": self._ptt,
                "squelch": self._squelch,
                "strength": self._strength,
            }

    def _wait_connected(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._sock is not None:
                    return True
            time.sleep(0.05)
        return False

    def _close_socket(self):
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock:
            try:
                sock.close()
            except OSError:
                pass

    def _cmd(self, cmd: str) -> bool:
        with self._lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall(cmd.encode())
            return True
        except OSError:
            self._close_socket()
            return False

    def _recv_line(self, sock: socket.socket) -> str | None:
        buf = bytearray()
        while self._running:
            try:
                b = sock.recv(1)
            except OSError:
                return None
            if not b:
                return None
            if b in (b"\n", b"\r"):
                if b == b"\r":
                    try:
                        n = sock.recv(1)
                        if n != b"\n":
                            buf.extend(n)
                    except OSError:
                        pass
                return buf.decode("utf-8", errors="replace")
            buf.extend(b)
            if len(buf) > 4096:
                return buf.decode("utf-8", errors="replace")
        return None

    def _run(self):
        while self._running:
            sock = self._try_connect()
            with self._lock:
                self._sock = sock
            if sock is None:
                if not self._running:
                    break
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                continue
            self._reconnect_delay = 1.0
            self._poll_loop(sock)
        self._close_socket()

    def _try_connect(self) -> socket.socket | None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self._host, self._port))
            sock.settimeout(1.0)
            return sock
        except (TimeoutError, OSError):
            return None

    def _poll_loop(self, sock: socket.socket):
        while self._running:
            if not self._send_polls(sock):
                break
            time.sleep(self._poll_interval)
        self._close_socket()

    def _send_polls(self, sock: socket.socket) -> bool:
        cmds = [b"f\n", b"m\n", b"t\n"]
        for c in cmds:
            try:
                sock.sendall(c)
                line = self._recv_line(sock)
            except OSError:
                return False
            if line is None:
                return False
            self._handle_response(line)

        sql_cmd = b"l RIG_SQL\n"
        str_cmd = b"l RIG_STRENGTH\n"
        for cmd in (sql_cmd, str_cmd):
            try:
                sock.sendall(cmd)
                line = self._recv_line(sock)
            except OSError:
                return False
            if line is not None:
                self._handle_response(line)

        return True

    _MODE_RE = re.compile(r"^[A-Z][A-Z0-9]*$")

    def _set_freq(self, mhz: float):
        old = self._freq
        with self._lock:
            self._freq = mhz
        if self._freq_callback and (old is None or abs(old - mhz) > 0.000001):
            self._freq_callback(mhz)

    def _set_ptt(self, on: bool):
        old = self._ptt
        with self._lock:
            self._ptt = on
        if self._ptt_callback and old != on:
            self._ptt_callback(on)

    def _set_mode(self, mode: str):
        old = self._mode
        with self._lock:
            self._mode = mode
        if self._mode_callback and old != mode:
            self._mode_callback(mode)

    def _handle_response(self, line: str):
        line = line.strip()
        if not line:
            return

        # PTT state (t command): single "0" or "1"
        if line in ("0", "1"):
            self._set_ptt(line == "1")
            return

        # Level read responses (l RIG_SQL / l RIG_STRENGTH)
        upper = line.upper()
        if "RIG_SQL" in upper or "RIG_STRENGTH" in upper:
            tokens = re.findall(r"-?\d+", line)
            if tokens:
                val = int(tokens[0])
                with self._lock:
                    if "RIG_SQL" in upper:
                        self._squelch = val
                    else:
                        self._strength = val
            return

        # Mode (m command): single alphabetic token like FM, USB, LSB
        if re.fullmatch(self._MODE_RE.pattern, line):
            self._set_mode(line)
            return

        # Frequency (f command): plain integer Hz
        if line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            self._set_freq(int(line) / 1_000_000.0)
            return

        parts = line.split()
        if len(parts) >= 1 and parts[0].isdigit():
            val = int(parts[0])
            if len(parts) >= 2 and parts[1] == "RIG_SQL":
                with self._lock:
                    self._squelch = val
            elif len(parts) >= 2 and parts[1] == "RIG_STRENGTH":
                with self._lock:
                    self._strength = val
            elif len(parts) == 1:
                self._set_freq(val / 1_000_000.0)
