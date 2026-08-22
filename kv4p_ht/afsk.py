"""
AFSK1200 modulation (Bell 202) for APRS transmission via Opus audio.
"""
from __future__ import annotations

import struct

import numpy as np

from .aprs import encode_ax25_ui

MARK   = 1200.0  # Hz (binary 1)
SPACE  = 2200.0  # Hz (binary 0)
BAUD   = 1200
SR     = 48000
FLAG   = 0x7E
PREAMBLE_FLAGS = 50
POSTAMBLE_FLAGS = 3
SQUELCH_OPEN_MS = 300
LEAD_MS  = 0
TAIL_MS  = 0

SPB = SR // BAUD  # 40 samples per bit


def crc_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc ^ 0xFFFF


def _flag_bits() -> list[int]:
    bits = []
    for i in range(8):
        bits.append((FLAG >> i) & 1)
    return bits


def _bytes_to_bits_lsb(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


def _bit_stuff(bits: list[int]) -> list[int]:
    out: list[int] = []
    ones = 0
    for b in bits:
        out.append(b)
        if b == 1:
            ones += 1
        else:
            ones = 0
        if ones == 5:
            out.append(0)
            ones = 0
    return out


def _nrzi_encode(bits: list[int], initial: int = 1) -> list[int]:
    out: list[int] = []
    prev = initial
    for b in bits:
        if b == 0:
            prev = 1 - prev
        out.append(prev)
    return out


def _frame_to_nrzi(frame: bytes) -> list[int]:
    flag_bits = _flag_bits()
    stuffed = _bit_stuff(_bytes_to_bits_lsb(frame))
    return _nrzi_encode(
        flag_bits * PREAMBLE_FLAGS + stuffed + flag_bits * POSTAMBLE_FLAGS
    )


def _modulate_nrz(bits: list[int], squelch_open: bool = False) -> np.ndarray:
    freq_table = np.where(np.repeat(np.array(bits), SPB) == 1, MARK, SPACE)
    if squelch_open:
        carrier = np.full(int(SR * SQUELCH_OPEN_MS / 1000), MARK)
        freq_table = np.concatenate([carrier, freq_table])
    phase = 2.0 * np.pi * np.cumsum(freq_table / SR)
    return (np.sin(phase) * 0.3).astype(np.float32)


def build_ax25_bits(source: str, dest: str,
                    digipeaters: list[str], info: bytes) -> list[int]:
    """Build complete AX.25 bitstream: flag + body + CRC + flag, bit-stuffed, NRZI."""
    body = encode_ax25_ui(source, dest, digipeaters, info)
    crc = crc_ccitt(body)
    frame = body + struct.pack('<H', crc)

    return _frame_to_nrzi(frame)


def modulate_ax25(source: str, dest: str,
                  digipeaters: list[str], info: bytes) -> np.ndarray:
    """Generate AFSK1200 audio waveform as float32 numpy array."""
    bits = build_ax25_bits(source, dest, digipeaters, info)

    return _modulate_nrz(bits)


def build_tx_waveform(source: str, dest: str,
                      digipeaters: list[str], info: bytes) -> np.ndarray:
    """AX.25 waveform with a carrier warm-up and flag preamble."""
    return _modulate_nrz(build_ax25_bits(source, dest, digipeaters, info), True)


def build_tx_waveform_from_body(body: bytes) -> np.ndarray:
    """Generate full TX waveform from a raw AX.25 UI frame body (no CRC).
    Body includes addresses + control + PID + info."""
    crc = crc_ccitt(body)
    frame = body + struct.pack('<H', crc)

    nrz = _frame_to_nrzi(frame)

    return _modulate_nrz(nrz, True)
