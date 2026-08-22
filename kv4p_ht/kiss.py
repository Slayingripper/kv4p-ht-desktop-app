from __future__ import annotations

import queue
import socket
import threading
import time
from collections.abc import Callable

KISS_FEND = 0xC0
KISS_FESC = 0xDB
KISS_TFEND = 0xDC
KISS_TFESC = 0xDD
COMMAND_DATA = 0x00


class KissTnc:
    def __init__(
        self,
        port: str = "/dev/pts/0",
        host: str | None = None,
        tcp_port: int = 8001,
        callback: Callable[[bytes], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ):
        self._port = port
        self._host = host
        self._tcp_port = tcp_port
        self._callback = callback
        self._log_fn = log_fn

        self._sock: socket.socket | None = None
        self._serial: object | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._rx_queue: queue.Queue = queue.Queue()
        self._reader_thread: threading.Thread | None = None

        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    def connect(self) -> bool:
        self.disconnect()
        self._running = True
        if self._host is not None:
            self._thread = threading.Thread(target=self._tcp_connect_loop, daemon=True)
            self._thread.start()
            return self._wait_connected(timeout=5.0)
        else:
            return self._serial_connect()

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close_connection()

    def is_connected(self) -> bool:
        with self._lock:
            if self._host is not None:
                return self._sock is not None
            else:
                return self._serial is not None

    def send_ax25(self, frame_body: bytes):
        data = bytes([COMMAND_DATA]) + self._encode(frame_body)
        frame = bytes([KISS_FEND]) + data + bytes([KISS_FEND])
        with self._lock:
            if self._host is not None and self._sock is not None:
                try:
                    self._sock.sendall(frame)
                except OSError:
                    self._close_connection()
            elif self._serial is not None:
                try:
                    self._serial.write(frame)
                except OSError:
                    self._close_connection()

    def _encode(self, data: bytes) -> bytes:
        result = bytearray()
        for b in data:
            if b == KISS_FEND:
                result.extend([KISS_FESC, KISS_TFEND])
            elif b == KISS_FESC:
                result.extend([KISS_FESC, KISS_TFESC])
            else:
                result.append(b)
        return bytes(result)

    def _decode(self, data: bytes) -> bytes:
        result = bytearray()
        escaped = False
        for b in data:
            if escaped:
                if b == KISS_TFEND:
                    result.append(KISS_FEND)
                elif b == KISS_TFESC:
                    result.append(KISS_FESC)
                escaped = False
            elif b == KISS_FESC:
                escaped = True
            else:
                result.append(b)
        return bytes(result)

    def start(self):
        self._running = True
        if self._reader_thread is None:
            self._reader_thread = threading.Thread(target=self._reader, daemon=True)
            self._reader_thread.start()

    def stop(self):
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None

    def _wait_connected(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._sock is not None:
                    return True
            time.sleep(0.05)
        return False

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)

    def _close_connection(self):
        with self._lock:
            sock = self._sock
            self._sock = None
            ser = self._serial
            self._serial = None
        if sock:
            try:
                sock.close()
            except OSError:
                pass
        if ser:
            try:
                ser.close()
            except OSError:
                pass

    def _tcp_connect_loop(self):
        while self._running:
            sock = self._try_tcp_connect()
            if sock is not None:
                with self._lock:
                    self._sock = sock
                break
            if not self._running:
                break
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def _try_tcp_connect(self) -> socket.socket | None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self._host, self._tcp_port))
            sock.settimeout(None)
            return sock
        except (TimeoutError, OSError):
            return None

    def _serial_connect(self) -> bool:
        try:
            import serial
        except ImportError:
            return False
        try:
            ser = serial.Serial(
                port=self._port,
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            with self._lock:
                self._serial = ser
            return True
        except OSError:
            return False

    def _reader(self):
        buf = bytearray()
        while self._running:
            data = self._read_chunk()
            if data is None:
                time.sleep(0.05)
                continue
            for b in data:
                if b == KISS_FEND:
                    if len(buf) > 0:
                        frame = self._parse_frame(bytes(buf))
                        if frame is not None and self._callback:
                            self._callback(frame)
                    buf = bytearray()
                else:
                    buf.append(b)

    def _parse_frame(self, data: bytes) -> bytes | None:
        if len(data) < 1:
            return None
        command = data[0]
        payload = data[1:]
        if command == COMMAND_DATA:
            return self._decode(payload)
        return None

    def _read_chunk(self) -> bytes | None:
        with self._lock:
            if self._host is not None:
                sock = self._sock
                if sock is None:
                    return None
                try:
                    sock.settimeout(0.5)
                    chunk = sock.recv(4096)
                    if not chunk:
                        self._close_connection()
                        return None
                    return chunk
                except (TimeoutError, OSError):
                    return None
            else:
                ser = self._serial
                if ser is None:
                    return None
                try:
                    chunk = ser.read(4096)
                    if chunk:
                        return chunk
                    return None
                except OSError:
                    self._close_connection()
                    return None
