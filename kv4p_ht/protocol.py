"""
Binary frame protocol matching the ESP32 firmware and Android app.

Frame format:
  [0xDE 0xAD 0xBE 0xEF] [1B cmd] [2B LE length] [N bytes payload]

Flow control: window-based. ESP32 declares initial window in VERSION,
host decrements on send, ESP32 replenishes via WINDOW_UPDATE.
"""
from __future__ import annotations

import struct
from enum import IntEnum

DELIMITER = b'\xDE\xAD\xBE\xEF'
PROTO_MTU = 2048

# ── Host -> ESP32 commands ───────────────────────────────────────
class HostCmd(IntEnum):
    PTT_DOWN  = 0x01
    PTT_UP    = 0x02
    GROUP     = 0x03
    FILTERS   = 0x04
    STOP      = 0x05
    CONFIG    = 0x06
    TX_AUDIO  = 0x07
    HL        = 0x08
    RSSI      = 0x09
    TX_AX25   = 0x0A

# ── ESP32 -> Host commands ───────────────────────────────────────
class EspCmd(IntEnum):
    SMETER_REPORT  = 0x53
    PHYS_PTT_DOWN  = 0x44
    PHYS_PTT_UP    = 0x55
    DEBUG_INFO     = 0x01
    DEBUG_ERROR    = 0x02
    DEBUG_WARN     = 0x03
    DEBUG_DEBUG    = 0x04
    DEBUG_TRACE    = 0x05
    HELLO          = 0x06
    RX_AUDIO       = 0x07
    VERSION        = 0x08
    WINDOW_UPDATE  = 0x09
    RX_AX25_PACKET = 0x0A

# ── Feature flags (from Version.features) ────────────────────────
FEAT_HAS_HL        = 1 << 0
FEAT_HAS_PHYS_PTT  = 1 << 1
FEAT_HAS_ESP32_AFSK = 1 << 2

# ── Struct helpers ───────────────────────────────────────────────

# Standard 38-tone CTCSS table, matching the kv4p-ht Android app's ToneHelper.
# The firmware GROUP command carries a tone *index* into this table (0 = none),
# not the tone value itself.
CTCSS_TONES = (
    67.0, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5,
    91.5, 94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8,
    118.8, 123.0, 127.3, 131.8, 136.5, 141.3, 146.2, 151.4,
    156.7, 162.2, 167.9, 173.8, 179.9, 186.2, 192.8, 203.5,
    210.7, 218.1, 225.7, 233.6, 241.8, 250.3,
)


def ctcss_to_index(tenths: int) -> int:
    """Convert a CTCSS tone in tenths of Hz (e.g. 885 == 88.5 Hz; 0 == none)
    to its wire-format index in CTCSS_TONES (1-based; 0 == none)."""
    if not tenths or tenths <= 0:
        return 0
    hz = tenths / 10.0
    idx = min(range(len(CTCSS_TONES)), key=lambda i: abs(CTCSS_TONES[i] - hz))
    return idx + 1 if abs(CTCSS_TONES[idx] - hz) <= 1.0 else 0


def pack_group(bw: int, freq_tx: float, freq_rx: float,
               ctcss_tx: int, squelch: int, ctcss_rx: int) -> bytes:
    return struct.pack('<BffBBB', bw, freq_tx, freq_rx,
                       ctcss_tx, squelch, ctcss_rx)

def pack_filters(pre: bool, high: bool, low: bool) -> bytes:
    flags = 0
    if pre:  flags |= 1
    if high: flags |= 2
    if low:  flags |= 4
    return struct.pack('<B', flags)

def pack_config(is_high: bool) -> bytes:
    return struct.pack('<?', is_high)

def pack_hl(is_high: bool) -> bytes:
    return struct.pack('<?', is_high)

def pack_rssi(on: bool) -> bytes:
    return struct.pack('<?', on)

def unpack_version(data: bytes) -> dict:
    # C packed struct: uint16 + char + size_t(4) + int32(4) + uint8 = 12 bytes
    ver, status, win_size, mod_type, features = struct.unpack_from(
        '<H B I I B', data
    )
    return {
        'ver': ver,
        'radio_status': chr(status) if 0 <= status < 256 else '?',
        'window_size': win_size,
        'module_type': mod_type,  # 0=SA818_VHF, 1=SA818_UHF
        'has_hl': bool(features & FEAT_HAS_HL),
        'has_phys_ptt': bool(features & FEAT_HAS_PHYS_PTT),
        'has_esp32_afsk': bool(features & FEAT_HAS_ESP32_AFSK),
    }

def unpack_window_update(data: bytes) -> int:
    return struct.unpack_from('<I', data)[0]

def unpack_rssi(data: bytes) -> int:
    return data[0] if data else 0

def rssi_to_s_meter(rssi: int) -> int:
    if rssi == 0:
        return 1
    import math
    val = 9.73 * math.log(0.0297 * rssi) - 1.88
    return max(1, min(9, round(val)))

# ── Frame sender ─────────────────────────────────────────────────

class FrameSender:
    def __init__(self, write_fn):
        self._write = write_fn

    def _send(self, cmd: HostCmd, payload: bytes = b''):
        frame = DELIMITER + bytes([cmd]) + struct.pack('<H', len(payload)) + payload
        self._write(frame)

    def ptt_down(self):      self._send(HostCmd.PTT_DOWN)
    def ptt_up(self):        self._send(HostCmd.PTT_UP)
    def stop(self):          self._send(HostCmd.STOP)

    def set_group(self, bw=0, freq_tx=144.390, freq_rx=144.390,
                  ctcss_tx=0, squelch=3, ctcss_rx=0):
        self._send(HostCmd.GROUP, pack_group(bw, freq_tx, freq_rx,
                                              ctcss_tx, squelch, ctcss_rx))

    def set_filters(self, pre=False, high=False, low=False):
        self._send(HostCmd.FILTERS, pack_filters(pre, high, low))

    def send_config(self, is_high=True):
        self._send(HostCmd.CONFIG, pack_config(is_high))

    def send_tx_audio(self, opus_data: bytes):
        self._send(HostCmd.TX_AUDIO, opus_data)

    def set_hl(self, is_high: bool):
        self._send(HostCmd.HL, pack_hl(is_high))

    def set_rssi(self, on: bool = True):
        self._send(HostCmd.RSSI, pack_rssi(on))

    def send_tx_ax25(self, ax25_frame: bytes):
        self._send(HostCmd.TX_AX25, ax25_frame)


# ── Frame parser ─────────────────────────────────────────────────

class FrameParser:
    def __init__(self, callback):
        self._cb = callback
        self._state = 'sync'
        self._match = 0
        self._cmd = 0
        self._plen = 0
        self._pbuf = bytearray()

    def feed(self, data: bytes):
        for b in data:
            self._feed_byte(b)

    def _feed_byte(self, b: int):
        if self._state == 'sync':
            self._match = (self._match + 1) if b == DELIMITER[self._match] else (1 if b == DELIMITER[0] else 0)
            if self._match == len(DELIMITER):
                self._state = 'cmd'
        elif self._state == 'cmd':
            self._cmd = b
            self._state = 'plen_lo'
        elif self._state == 'plen_lo':
            self._plen = b
            self._state = 'plen_hi'
        elif self._state == 'plen_hi':
            self._plen |= b << 8
            self._pbuf = bytearray()
            if self._plen == 0:
                self._cb(self._cmd, b'')
                self._reset()
            elif self._plen > PROTO_MTU:
                self._reset()
            else:
                self._state = 'payload'
        elif self._state == 'payload':
            self._pbuf.append(b)
            if len(self._pbuf) >= self._plen:
                self._cb(self._cmd, bytes(self._pbuf))
                self._reset()

    def _reset(self):
        self._state = 'sync'
        self._match = 0
        self._cmd = 0
        self._plen = 0
        self._pbuf = bytearray()
