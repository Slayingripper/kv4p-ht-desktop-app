"""
APRS: AX.25 framing, APRS info-field parsing, iGate (APRS-IS), digipeater.
"""
from __future__ import annotations

import queue
import socket
import threading
import time

# ── AX.25 constants ──────────────────────────────────────────────
AX25_UI   = 0x03
PID_NO_L3 = 0xF0

def _encode_addr(callsign: str, is_last: bool) -> bytes:
    parts = callsign.split('-')
    base = parts[0].ljust(6).encode('ascii')
    ssid = int(parts[1]) if len(parts) > 1 else 0
    addr = bytearray(7)
    for i in range(6):
        addr[i] = base[i] << 1
    addr[6] = 0x60 | (ssid << 1) | (0x01 if is_last else 0x00)
    return bytes(addr)

def encode_ax25_ui(source: str, destination: str,
                   digipeaters: list[str], info: bytes) -> bytes:
    """AX.25 UI frame body (addresses + control + PID + info).
    No flags, no CRC — the ESP32 firmware's AFSK modulator adds those."""
    frame = bytearray()
    frame.extend(_encode_addr(destination, False))
    frame.extend(_encode_addr(source, len(digipeaters) == 0))
    for i, dp in enumerate(digipeaters):
        frame.extend(_encode_addr(dp, i == len(digipeaters) - 1))
    frame.append(AX25_UI)
    frame.append(PID_NO_L3)
    frame.extend(info)
    return bytes(frame)

def decode_address(addr: bytes) -> str:
    call = ''.join(chr(b >> 1) for b in addr[:6]).rstrip()
    ssid = (addr[6] >> 1) & 0x0F
    return f"{call}-{ssid}" if ssid else call


def decode_ax25_frame(frame: bytes) -> dict:
    """Decode a complete AX.25 UI frame body (starting with addresses, no flags)."""
    result = {}
    offset = 0
    if len(frame) < 14:
        return {'raw': frame}
    # Destination (7 bytes)
    result['destination'] = decode_address(frame[offset:offset + 7])
    offset += 7
    # Source (7 bytes)
    src_addr = frame[offset:offset + 7]
    result['source'] = decode_address(src_addr)
    src_last = bool(src_addr[6] & 0x01)
    offset += 7
    # Digipeaters (only if source was NOT the last address)
    digis = []
    if not src_last:
        while offset + 6 < len(frame):
            is_last = bool(frame[offset + 6] & 0x01)
            digis.append(decode_address(frame[offset:offset + 7]))
            offset += 7
            if is_last:
                break
    result['digipeaters'] = digis
    # Control + PID
    if offset < len(frame):
        result['control'] = frame[offset]
        offset += 1
    if offset < len(frame):
        result['pid'] = frame[offset]
        offset += 1
    # Info
    result['info'] = frame[offset:]
    return result

# ── APRS info-field parser ───────────────────────────────────────

def parse_aprs(info_field: str, source: str = '') -> dict:
    """Lightweight APRS info parser — returns a structured dict."""
    result = {'source': source, 'raw': info_field}

    if not info_field:
        return result

    if info_field[0] == ':':
        # Message: :ADDRESSEE:TEXT{msgid
        rest = info_field[1:]
        colon = rest.find(':')
        if colon >= 0:
            result['type'] = 'message'
            result['addressee'] = rest[:colon].strip()
            text_rest = rest[colon + 1:]
            if '{' in text_rest:
                result['text'], result['msg_id'] = text_rest.split('{', 1)
            else:
                result['text'] = text_rest
                result['msg_id'] = ''
    elif info_field[0] in ('=', '!', '@', '/'):
        # Position report
        result['type'] = 'position'
        if info_field[0] in ('=', '@'):
            data = info_field[1:]
        else:
            data = info_field
        result['raw_position'] = data[:19]
        if len(data) > 19:
            # The APRS position field is a fixed-length prefix plus an optional
            # comment. Keep the first character of the comment, which is part of
            # the data immediately after the position text.
            result['comment'] = data[19:]
        else:
            result['comment'] = ''
    elif info_field[0] == '>':
        result['type'] = 'status'
        result['text'] = info_field[1:]
    elif info_field[0] == '_':
        # Weather
        result['type'] = 'weather'
        result['raw'] = info_field
    elif info_field[0] == ';':
        result['type'] = 'object'
        result['raw'] = info_field
    elif info_field[0] == 'T':
        result['type'] = 'telemetry'
    else:
        result['type'] = 'beacon'
        result['text'] = info_field

    return result

def format_beacon(callsign: str, lat: float, lon: float,
                  comment: str = '', symbol: str = '/-') -> str:
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_dir = 'E' if lon >= 0 else 'W'
    return (f"={lat_deg:02d}{lat_min:05.2f}{lat_dir}"
            f"{symbol[0]}{lon_deg:03d}{lon_min:05.2f}{lon_dir}"
            f"{symbol[1]}{comment}")

def format_message(dest: str, text: str, msg_id: str = '') -> str:
    msg = f":{dest:<9}:{text}"
    if msg_id:
        msg += f"{{{msg_id}"
    return msg[:256]

def format_ack(addressee: str, msg_id: str) -> str:
    return f":{addressee:<9}:ack{msg_id}"

# ── iGate (APRS-IS gateway) ──────────────────────────────────────

class IGate(threading.Thread):
    """Connects to APRS-IS, forwards RF messages to Internet and vice versa."""

    def __init__(self, callsign: str,
                 rf_tx_callback,  # callable(bytes) — send AX.25 frame to radio
                 aprs_is_callback,  # callable(str) — APRS line received from IS
                 lat: float = 0.0, lon: float = 0.0,
                 filter_str: str = '',
                 tx_enabled: bool = True,
                 status_text: str = '',
                 beacon_interval: int = 600,
                 passcode: str = '-1',
                 ):
        super().__init__(daemon=True)
        self.callsign = callsign
        self._rf_tx = rf_tx_callback
        self._aprs_is_rx = aprs_is_callback
        self.lat = lat
        self.lon = lon
        self.filter_str = filter_str
        self.tx_enabled = tx_enabled
        self.status_text = status_text
        self._beacon_interval = beacon_interval
        self.passcode = passcode
        self._to_is_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._running = True
        self._last_beacon = 0.0

    def stop(self):
        self._running = False
        self._to_is_queue.put(None)

    def send_to_is(self, source: str, info_field: str):
        self._to_is_queue.put((source, info_field))

    def run(self):
        while self._running:
            try:
                sock = socket.create_connection(
                    ("rotate.aprs.net", 14580), timeout=30
                )
                sock.settimeout(60)
                login = (
                    f"user {self.callsign} pass {self.passcode} vers kv4p-desktop 0.1\r\n"
                )
                sock.sendall(login.encode())
                time.sleep(0.5)

                if self.filter_str:
                    sock.sendall(f"{self.filter_str}\r\n".encode())

                f = sock.makefile('r', encoding='ascii', errors='replace')
                self._aprs_is_rx(f"Connected to APRS-IS as {self.callsign}")

                while self._running:
                    now = time.time()

                    # Forward from RF -> IS
                    try:
                        item = self._to_is_queue.get(timeout=0.1)
                        if item is None:
                            break
                        src, info = item
                        line = f"{src}>APZ010,TCPIP*:{info}\r\n"
                        sock.sendall(line.encode())
                    except queue.Empty:
                        pass

                    # Send IS beacon periodically
                    if self._beacon_interval > 0 and now - self._last_beacon >= self._beacon_interval:
                        self._last_beacon = now
                        if self.lat != 0.0 or self.lon != 0.0:
                            bcn = format_beacon(
                                self.callsign, self.lat, self.lon,
                                comment=self.status_text or 'KV4P-Desktop iGate',
                            )
                            sock.sendall(f"{self.callsign}>APZ010,TCPIP*:{bcn}\r\n".encode())

                    # Forward from IS -> RF
                    try:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line and ':' in line:
                            self._aprs_is_rx(line)
                            if line[0] != '#' and self.tx_enabled:
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    header = parts[0]
                                    info = parts[1]
                                    src = header.split('>')[0] if '>' in header else '?'
                                    if info.startswith(':'):
                                        frame = encode_ax25_ui(src, self.callsign, [], info.encode())
                                        try:
                                            self._rf_tx(frame, from_igate=True)
                                        except Exception:
                                            pass
                    except TimeoutError:
                        continue

                sock.close()
            except (TimeoutError, ConnectionRefusedError, OSError) as e:
                self._aprs_is_rx(f"APRS-IS error: {e}")
                time.sleep(30)

# ── Digipeater ───────────────────────────────────────────────────

class Digipeater:
    """Digipeater — forwards packets heard on RF, adding our callsign to path."""

    def __init__(self, mycall: str, tx_callback):
        self.mycall = mycall.upper()
        self._tx = tx_callback
        self._heard: dict[tuple[str, str], float] = {}

    def process(self, ax25_frame: bytes) -> bool:
        try:
            dec = decode_ax25_frame(ax25_frame)
            src = dec.get('source', '')
            dst = dec.get('destination', '')
            digis = dec.get('digipeaters', [])
        except Exception:
            return False
        if not src or not dst:
            return False

        now = time.time()
        self._heard = {k: v for k, v in self._heard.items() if now - v < 30}
        if (src, dst) in self._heard:
            return False
        self._heard[(src, dst)] = now

        if self.mycall in [d.upper().rstrip('*') for d in digis]:
            return False

        new_digis = (digis if digis else []) + [f"{self.mycall}*"]
        info = dec.get('info', b'')
        new_frame = encode_ax25_ui(src, dst, new_digis, info)
        try:
            self._tx(new_frame)
        except Exception:
            return False
        return True


