"""
AFSK1200 demodulator (Bell 202) for APRS reception.

Sliding-window Goertzel-based detector matching javAX25 algorithm.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from .aprs import decode_ax25_frame, parse_aprs

MARK_HZ  = 1200.0
SPACE_HZ = 2200.0
BAUD     = 1200
SR       = 48000
SPB      = SR // BAUD  # 40 samples per bit
AX25_CRC_OK = 0xF0B8


class AfskDemodulator:
    def __init__(self, callback: Callable[[dict], None] | None = None):
        self.callback = callback

        # quadrature oscillators
        self._ph_mark  = 0.0
        self._ph_space = 0.0
        self._inc_mark  = 2.0 * math.pi * MARK_HZ / SR
        self._inc_space = 2.0 * math.pi * SPACE_HZ / SR

        # sliding correlation arrays (one per sample offset in bit period)
        self._c0r = [0.0] * SPB
        self._c0i = [0.0] * SPB
        self._c1r = [0.0] * SPB
        self._c1i = [0.0] * SPB
        self._jcorr = 0

        # smoothing history for fdiff
        self._fdiff_hist_len = 3
        self._fdiff_hist: list[float] = []
        self._last_fdiff = 0.0

        # timing recovery
        self._last_xing = -1
        self._nsamp = 0

        # bit decoder state (matches Java bit accumulation)
        self._data = 0
        self._bitcount = 0
        self._state = 'WAITING'
        self._flag_count = 0
        self._flag_sep = False

        # frame buffer
        self._frame = bytearray()
        self._crc = 0xFFFF

        # carrier detect
        self._carrier_energy = 0.0
        self._carrier_threshold = 0.001
        self._carrier_on = False
        self.carrier_starts = 0
        self.valid_frames = 0
        self.rejected_frames = 0

    def diagnostics(self) -> dict[str, int | bool]:
        return {
            'carrier': self._carrier_on,
            'carrier_starts': self.carrier_starts,
            'valid_frames': self.valid_frames,
            'rejected_frames': self.rejected_frames,
        }

    def _reset(self):
        self._state = 'WAITING'
        self._flag_count = 0
        self._flag_sep = False
        self._data = 0
        self._bitcount = 0
        self._frame = bytearray()
        self._crc = 0xFFFF
        self._carrier_energy = 0.0

    def _crc_upd(self, b: int):
        self._crc = ((self._crc >> 8) ^ _CRC_TABLE[(self._crc ^ b) & 0xFF])

    def _emit_byte(self, b: int):
        self._frame.append(b)
        self._crc_upd(b)

    def _finalize(self):
        if len(self._frame) < 18:
            self.rejected_frames += 1
            self._reset()
            return
        if self._crc != AX25_CRC_OK:
            self.rejected_frames += 1
            self._reset()
            return
        body = bytes(self._frame[:-2])
        try:
            dec = decode_ax25_frame(body)
            info_str = dec.get('info', b'').decode('ascii', errors='replace')
            parsed = parse_aprs(info_str, dec.get('source', ''))
            parsed['raw_frame'] = body
            if self.callback:
                self.callback(parsed)
            self.valid_frames += 1
        except Exception:
            self.rejected_frames += 1
            pass
        self._reset()

    def process(self, samples: list[float]):
        for s in samples:
            self._process_sample(s)

    def _process_sample(self, s: float):
        # mix with quadrature carriers (sliding Goertzel)
        cm = math.cos(self._ph_mark)
        sm = math.sin(self._ph_mark)
        cs = math.cos(self._ph_space)
        ss = math.sin(self._ph_space)

        self._c0r[self._jcorr] = s * cm
        self._c0i[self._jcorr] = s * sm
        self._c1r[self._jcorr] = s * cs
        self._c1i[self._jcorr] = s * ss

        self._ph_mark  += self._inc_mark
        self._ph_space += self._inc_space
        if self._ph_mark  > 2.0 * math.pi: self._ph_mark  -= 2.0 * math.pi
        if self._ph_space > 2.0 * math.pi: self._ph_space -= 2.0 * math.pi

        # sum across all positions (one full bit period sliding window)
        sr0 = sum(self._c0r)
        si0 = sum(self._c0i)
        sr1 = sum(self._c1r)
        si1 = sum(self._c1i)

        mm = math.hypot(sr0, si0)
        ms = math.hypot(sr1, si1)
        fdiff = mm - ms

        # carrier energy estimate (for squelch)
        self._carrier_energy = self._carrier_energy * 0.99 + (mm + ms) * 0.01
        carrier_was_on = self._carrier_on
        self._carrier_on = self._carrier_energy > self._carrier_threshold
        if self._carrier_on and not carrier_was_on:
            self.carrier_starts += 1

        self._fdiff_hist.append(fdiff)
        if len(self._fdiff_hist) > self._fdiff_hist_len:
            self._fdiff_hist.pop(0)
        fdiff_s = sum(self._fdiff_hist) / len(self._fdiff_hist)

        if not self._carrier_on:
            self._last_fdiff = fdiff_s
            self._jcorr += 1
            if self._jcorr >= SPB:
                self._jcorr = 0
            self._nsamp += 1
            return

        # zero-crossing detection with hysteresis
        xing = False
        if self._last_xing < 0:
            self._last_xing = self._nsamp
            self._last_fdiff = fdiff_s
        else:
            hyst = 0.0005
            if (self._last_fdiff >= hyst and fdiff_s < -hyst) or \
               (self._last_fdiff < -hyst and fdiff_s > hyst):
                xing = True

        if xing:
            p = self._nsamp - self._last_xing
            self._last_xing = self._nsamp
            bits = round(p / SPB)
            self._on_transition(bits)

        self._last_fdiff = fdiff_s

        self._jcorr += 1
        if self._jcorr >= SPB:
            self._jcorr = 0
        self._nsamp += 1

    def _on_transition(self, bits: int):
        if bits == 0 or bits > 7:
            if bits > 7:
                pass
            return

        if bits == 7:
            self._flag_count += 1
            self._flag_sep = False
            self._data = 0
            self._bitcount = 0
            if self._state == 'WAITING':
                self._state = 'PRE_FLAG'
            elif self._state == 'DECODING':
                if self._crc == AX25_CRC_OK and len(self._frame) >= 18:
                    self._finalize()
                else:
                    self._reset()
                self._state = 'PRE_FLAG'
            return

        if 1 <= bits <= 6:
            if self._state == 'PRE_FLAG':
                self._state = 'DECODING'
                self._frame = bytearray()
                self._crc = 0xFFFF
                self._flag_count = 0

            if self._state == 'DECODING':
                if bits != 1:
                    self._flag_count = 0
                else:
                    if self._flag_count > 0 and not self._flag_sep:
                        self._flag_sep = True
                    else:
                        self._flag_count = 0

                for _ in range(bits - 1):
                    self._add_bit(1)
                if bits - 1 != 5:
                    self._add_bit(0)

    def _add_bit(self, bit: int):
        self._data >>= 1
        if bit:
            self._data |= 0x80
        self._bitcount += 1
        if self._bitcount == 8:
            self._emit_byte(self._data)
            self._data = 0
            self._bitcount = 0


_CRC_TABLE: list[int] = []
if not _CRC_TABLE:
    for i in range(256):
        crc = 0
        b = i
        for _ in range(8):
            if (crc ^ b) & 1:
                crc = ((crc >> 1) ^ 0x8408) & 0xFFFF
            else:
                crc = (crc >> 1) & 0xFFFF
            b >>= 1
        _CRC_TABLE.append(crc)
