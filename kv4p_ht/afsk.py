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
LEAD_MS  = 1100
TAIL_MS  = 700

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


def build_ax25_bits(source: str, dest: str,
                    digipeaters: list[str], info: bytes) -> list[int]:
    """Build complete AX.25 bitstream: flag + body + CRC + flag, bit-stuffed, NRZI."""
    body = encode_ax25_ui(source, dest, digipeaters, info)
    crc = crc_ccitt(body)
    frame = body + struct.pack('<H', crc)

    flag_bits = _flag_bits()
    data_bits = _bytes_to_bits_lsb(frame)
    stuffed = _bit_stuff(data_bits)
    nrz = _nrzi_encode(flag_bits + stuffed + flag_bits)
    return nrz


def modulate_ax25(source: str, dest: str,
                  digipeaters: list[str], info: bytes) -> np.ndarray:
    """Generate AFSK1200 audio waveform as float32 numpy array."""
    bits = build_ax25_bits(source, dest, digipeaters, info)

    total = len(bits) * SPB
    waveform = np.empty(total, dtype=np.float32)

    # Precompute per-sample frequencies using repeat
    freq_table = np.where(np.array(bits) == 1, MARK, SPACE)
    freqs = np.repeat(freq_table, SPB)

    # Continuous-phase FSK via cumulative sum
    dt = 1.0 / SR
    phase = 2.0 * np.pi * np.cumsum(freqs * dt)
    np.sin(phase, out=waveform)

    return waveform * 0.3  # keep some headroom


def build_tx_waveform(source: str, dest: str,
                      digipeaters: list[str], info: bytes) -> np.ndarray:
    """Full TX waveform with lead/tail silence, ready to send as Opus."""
    audio = modulate_ax25(source, dest, digipeaters, info)
    lead = np.zeros(int(SR * LEAD_MS / 1000), dtype=np.float32)
    tail = np.zeros(int(SR * TAIL_MS / 1000), dtype=np.float32)
    return np.concatenate([lead, audio, tail])


def build_tx_waveform_from_body(body: bytes) -> np.ndarray:
    """Generate full TX waveform from a raw AX.25 UI frame body (no CRC).
    Body includes addresses + control + PID + info."""
    crc = crc_ccitt(body)
    frame = body + struct.pack('<H', crc)

    flag_bits = _flag_bits()
    data_bits = _bytes_to_bits_lsb(frame)
    stuffed = _bit_stuff(data_bits)
    nrz = _nrzi_encode(flag_bits + stuffed + flag_bits)

    total = len(nrz) * SPB
    waveform = np.empty(total, dtype=np.float32)
    freq_table = np.where(np.array(nrz) == 1, MARK, SPACE)
    freqs = np.repeat(freq_table, SPB)
    dt = 1.0 / SR
    phase = 2.0 * np.pi * np.cumsum(freqs * dt)
    np.sin(phase, out=waveform)

    audio = waveform * 0.3
    lead = np.zeros(int(SR * LEAD_MS / 1000), dtype=np.float32)
    tail = np.zeros(int(SR * TAIL_MS / 1000), dtype=np.float32)
    return np.concatenate([lead, audio, tail])
