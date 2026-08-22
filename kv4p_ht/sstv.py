"""
SSTV (Slow Scan Television) encoder and decoder.

Encoder wraps PySSTV for correct VIS headers and line timing across all modes.
Decoder is a port of Open-SSTV's DSP pipeline (Hilbert demod, adaptive sync
detection, per-mode pixel slicing, slant correction).

Decoder ported from Open-SSTV (GPL-3.0-or-later) by Kevin (W0AEZ).
https://github.com/bucknova/Open-SSTV

Debug: Set SSTV_DEBUG=1 env var for verbose encode/decode logging.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy import ndimage, signal
from scipy.signal import sosfiltfilt

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
_debug = logging.getLogger(__name__ + ".debug")

COLOR_BLACK_HZ = 1500.0
COLOR_WHITE_HZ = 2300.0
SYNC_HZ = 1200.0
COLOR_RANGE = COLOR_WHITE_HZ - COLOR_BLACK_HZ


class SstvMode:
    def __init__(self, name: str, lines: int, vis_code: int,
                 scan_order: str = "rgb", width: int = 320):
        self.name = name
        self.lines = lines
        self.vis_code = vis_code
        self.scan_order = scan_order
        self.resolution = (width, lines)


MODES: dict[str, SstvMode] = {
    "M1": SstvMode("Martin M1", lines=256, vis_code=0x2C, scan_order="gbr"),
    "M2": SstvMode("Martin M2", lines=256, vis_code=0x28, scan_order="gbr", width=160),
    "M3": SstvMode("Martin M3", lines=128, vis_code=0x24, scan_order="gbr"),
    "M4": SstvMode("Martin M4", lines=128, vis_code=0x20, scan_order="gbr", width=160),
    "S1": SstvMode("Scottie S1", lines=256, vis_code=0x3C, scan_order="gbr"),
    "S2": SstvMode("Scottie S2", lines=256, vis_code=0x38, scan_order="gbr", width=160),
    "S3": SstvMode("Scottie S3", lines=128, vis_code=0x34, scan_order="gbr"),
    "S4": SstvMode("Scottie S4", lines=128, vis_code=0x30, scan_order="gbr", width=160),
    "SDX": SstvMode("Scottie DX", lines=256, vis_code=0x4C, scan_order="gbr"),
    "PD50": SstvMode("PD-50", lines=256, vis_code=0x5D, scan_order="rgb"),
    "PD90": SstvMode("PD-90", lines=256, vis_code=0x63, scan_order="rgb"),
    "PD120": SstvMode("PD-120", lines=496, vis_code=0x5F, scan_order="rgb", width=640),
    "PD160": SstvMode("PD-160", lines=400, vis_code=0x62, scan_order="rgb", width=512),
    "PD180": SstvMode("PD-180", lines=496, vis_code=0x60, scan_order="rgb", width=640),
    "PD240": SstvMode("PD-240", lines=496, vis_code=0x61, scan_order="rgb", width=640),
    "PD290": SstvMode("PD-290", lines=616, vis_code=0x5E, scan_order="rgb", width=800),
    "R36": SstvMode("Robot 36", lines=240, vis_code=0x08, scan_order="ycbcr", width=320),
    "SC2_120": SstvMode("SC2-120", lines=256, vis_code=0x3F, scan_order="rgb"),
    "SC2_180": SstvMode("SC2-180", lines=256, vis_code=0x37, scan_order="rgb"),
    "P3": SstvMode("Pasokon P3", lines=496, vis_code=0x71, scan_order="rgb", width=640),
    "P5": SstvMode("Pasokon P5", lines=496, vis_code=0x72, scan_order="rgb", width=640),
    "P7": SstvMode("Pasokon P7", lines=496, vis_code=0x73, scan_order="rgb", width=640),
}


def freq_to_luminance(freq: float) -> int:
    normalized = (freq - COLOR_BLACK_HZ) / COLOR_RANGE
    return max(0, min(255, int(normalized * 255 + 0.5)))


def luminance_to_freq(lum: int) -> float:
    return COLOR_BLACK_HZ + (lum / 255.0) * COLOR_RANGE


# ENCODER

class SstvEncoder:
    def __init__(self, mode: str = "M1", sample_rate: int = 48000):
        self.mode = MODES.get(mode, MODES["M1"])
        self.sample_rate = sample_rate
        self._mode_key = mode

    def encode_image(self, image_data: np.ndarray) -> np.ndarray:
        from pysstv.color import (
            PD90, PD120, PD160, PD180, PD240, PD290,
            MartinM1, MartinM2,
            PasokonP3, PasokonP5, PasokonP7,
            Robot36, ScottieDX, ScottieS1, ScottieS2,
            WraaseSC2120, WraaseSC2180,
        )
        mode = self.mode
        if image_data.ndim == 2:
            img = np.stack([image_data] * 3, axis=-1)
        else:
            img = image_data.copy()
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if self._mode_key == "R36":
            pil_img = Image.fromarray(img).convert("RGB")
            w, h = mode.resolution
            pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
            sstv = Robot36(pil_img, self.sample_rate, 16)
        else:
            _PYSSTV_DIMS = {
                "M1": (320, 256), "M2": (160, 256), "M3": (320, 256), "M4": (160, 256),
                "S1": (320, 256), "S2": (160, 256), "S3": (320, 256), "S4": (160, 256),
                "SDX": (320, 256),
                "PD50": (320, 256), "PD90": (320, 256), "PD120": (640, 496),
                "PD160": (512, 400), "PD180": (640, 496), "PD240": (640, 496), "PD290": (800, 616),
                "SC2_120": (320, 256), "SC2_180": (320, 256),
                "P3": (640, 496), "P5": (640, 496), "P7": (640, 496),
            }
            class_map = {
                "M1": MartinM1, "M2": MartinM2,
                "M3": MartinM1, "M4": MartinM2,
                "S1": ScottieS1, "S2": ScottieS2,
                "S3": ScottieS1, "S4": ScottieS2,
                "SDX": ScottieDX,
                "PD50": PD90, "PD90": PD90, "PD120": PD120,
                "PD160": PD160, "PD180": PD180, "PD240": PD240, "PD290": PD290,
                "SC2_120": WraaseSC2120, "SC2_180": WraaseSC2180,
                "P3": PasokonP3, "P5": PasokonP5, "P7": PasokonP7,
            }
            pysstv_cls = class_map.get(self._mode_key)
            if pysstv_cls is None:
                raise ValueError(f"Unsupported SSTV mode: {self._mode_key}")
            w, h = _PYSSTV_DIMS.get(self._mode_key, mode.resolution)
            pil_img = Image.fromarray(img).convert("RGB")
            pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
            sstv = pysstv_cls(pil_img, self.sample_rate, 16)
        waveform = np.fromiter(sstv.gen_samples(), dtype=np.int16)
        return waveform.astype(np.float32) / 32768.0


class _Robot36LinePair:
    def __init__(self, image, sample_rate, bits):
        from pysstv.color import Robot36
        self._inner = Robot36(image, sample_rate, bits)
        self.image = image
        self.sample_rate = sample_rate

    def gen_samples(self):
        from pysstv.sstv import FREQ_SYNC, FREQ_BLACK, FREQ_VIS_START, byte_to_freq
        yuv = self.image.convert("YCbCr").load()
        w, h = self.image.size
        Y_SCAN, C_SCAN = 88.0, 44.0
        SYNC, SYNC_PORCH = 9.0, 3.0
        INTER_CH_GAP, PORCH = 4.5, 1.5
        y_pixel_ms = Y_SCAN / w
        c_pixel_ms = C_SCAN / w
        sr = self.sample_rate
        for row in range(0, h, 2):
            even_pixels = [yuv[col, row] for col in range(w)]
            odd_pixels = [yuv[col, row + 1] for col in range(w)]
            for half_idx, (pix_a, pix_b) in enumerate(
                [(even_pixels, odd_pixels), (odd_pixels, even_pixels)]
            ):
                yield from _tone_gen(FREQ_SYNC, SYNC, sr)
                yield from _tone_gen(FREQ_BLACK, SYNC_PORCH, sr)
                for p in pix_a:
                    yield from _tone_gen(byte_to_freq(p[0]), y_pixel_ms, sr)
                sep = FREQ_BLACK if half_idx == 0 else 2300.0
                yield from _tone_gen(sep, INTER_CH_GAP, sr)
                yield from _tone_gen(FREQ_VIS_START, PORCH, sr)
                if half_idx == 0:
                    for ep, op in zip(pix_a, pix_b, strict=True):
                        cr = (ep[2] + op[2]) / 2
                        yield from _tone_gen(byte_to_freq(cr), c_pixel_ms, sr)
                else:
                    for ep, op in zip(pix_a, pix_b, strict=True):
                        cb = (ep[1] + op[1]) / 2
                        yield from _tone_gen(byte_to_freq(cb), c_pixel_ms, sr)


def _tone_gen(freq: float, duration_ms: float, sr: int):
    n = int(sr * duration_ms / 1000.0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float64) / sr
    for s in (np.sin(2.0 * np.pi * freq * t) * 32767 * 0.8).astype(np.int16):
        yield int(s)


# ══════════════════════════════════════════════════════════════════
# DECODER DSP — ported from Open-SSTV (GPL-3.0-or-later)
# Copyright 2025 Kevin (W0AEZ), https://github.com/bucknova/Open-SSTV
# ══════════════════════════════════════════════════════════════════

_BP_LOW_HZ = 1000.0
_BP_HIGH_HZ = 2500.0
_BP_ORDER = 4
_BP_MIN_SAMPLES = 256
_SYNC_REJECT_HZ = 1400.0


@unique
class _SyncPosition(StrEnum):
    LINE_START = "line_start"
    BEFORE_RED = "before_red"


@dataclass(frozen=True, slots=True)
class _ModeSpec:
    name: str
    vis_code: int
    width: int
    height: int
    sync_pulse_ms: float
    sync_porch_ms: float
    line_time_ms: float
    sync_position: _SyncPosition
    display_height: int = 0
    scan_order: str = "rgb"

    def __post_init__(self):
        if self.display_height == 0:
            object.__setattr__(self, "display_height", self.height)


# Mode timing constants from Open-SSTV modes.py
_MS1 = 146.432
_MS2 = 73.216
_MP = 0.572
_Msync = 4.862
_SS1 = 136.74
_SS2 = 86.564
_SDX = 344.1
_SP = 1.5
_Ssync = 9.0
_R36_Y = 88.0
_R36_C = 44.0
_R36_G = 4.5
_R36_P = 1.5
_R36_SYNC = 9.0
_R36_SP = 3.0
_PD_SYNC = 20.0
_PD_P = 2.08
_PD50 = 320 * 0.286
_PD90 = 320 * 0.532
_PD120 = 640 * 0.190
_PD160 = 512 * 0.382
_PD180 = 640 * 0.286
_PD240 = 640 * 0.382
_PD290 = 800 * 0.286
_WS = 5.5225
_WP = 0.5
_W120 = 156.0
_W180 = 235.0
_PP3S = 1000 / 4800 * 25
_PP3 = 1000 / 4800 * 640
_PP3G = 1000 / 4800 * 5
_PP5S = 1000 / 3200 * 25
_PP5 = 1000 / 3200 * 640
_PP5G = 1000 / 3200 * 5
_PP7S = 1000 / 2400 * 25
_PP7 = 1000 / 2400 * 640
_PP7G = 1000 / 2400 * 5


def _build_mode_table() -> dict[int, _ModeSpec]:
    t: dict[int, _ModeSpec] = {}

    def a(n, v, w, h, sm, pm, lm, sp, dh=0, so="rgb"):
        t[v] = _ModeSpec(n, v, w, h, sm, pm, lm, sp, dh or h, so)

    ls = _SyncPosition.LINE_START
    br = _SyncPosition.BEFORE_RED
    a("Martin M1", 0x2C, 320, 256, _Msync, _MP, _Msync + 4 * _MP + 3 * _MS1, ls, so="gbr")
    a("Martin M2", 0x28, 160, 256, _Msync, _MP, _Msync + 4 * _MP + 3 * _MS2, ls, so="gbr")
    a("Martin M3", 0x24, 320, 128, _Msync, _MP, _Msync + 4 * _MP + 3 * _MS1, ls, so="gbr")
    a("Martin M4", 0x20, 160, 128, _Msync, _MP, _Msync + 4 * _MP + 3 * _MS2, ls, so="gbr")
    a("Scottie S1", 0x3C, 320, 256, _Ssync, _SP, _Ssync + 6 * _SP + 3 * _SS1, br, so="gbr")
    a("Scottie S2", 0x38, 160, 256, _Ssync, _SP, _Ssync + 6 * _SP + 3 * _SS2, br, so="gbr")
    a("Scottie DX", 0x4C, 320, 256, _Ssync, _SP, _Ssync + 6 * _SP + 3 * _SDX, br, so="gbr")
    a("Scottie S3", 0x34, 320, 128, _Ssync, _SP, _Ssync + 6 * _SP + 3 * _SS1, br, so="gbr")
    a("Scottie S4", 0x30, 160, 128, _Ssync, _SP, _Ssync + 6 * _SP + 3 * _SS2, br, so="gbr")
    a("Robot 36", 0x08, 320, 240, _R36_SYNC, _R36_SP,
      _R36_SYNC + _R36_SP + _R36_Y + _R36_G + _R36_P + _R36_C, ls, so="ycbcr")
    a("PD-50", 0x5D, 320, 128, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD50, ls, 256)
    a("PD-90", 0x63, 320, 128, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD90, ls, 256)
    a("PD-120", 0x5F, 640, 248, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD120, ls, 496)
    a("PD-160", 0x62, 512, 200, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD160, ls, 400)
    a("PD-180", 0x60, 640, 248, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD180, ls, 496)
    a("PD-240", 0x61, 640, 248, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD240, ls, 496)
    a("PD-290", 0x5E, 800, 308, _PD_SYNC, _PD_P, _PD_SYNC + _PD_P + 4 * _PD290, ls, 616)
    a("SC2-120", 0x3F, 320, 256, _WS, _WP, _WS + _WP + 3 * _W120, ls)
    a("SC2-180", 0x37, 320, 256, _WS, _WP, _WS + _WP + 3 * _W180, ls)
    a("Pasokon P3", 0x71, 640, 496, _PP3S, _PP3G, _PP3S + 4 * _PP3G + 3 * _PP3, ls)
    a("Pasokon P5", 0x72, 640, 496, _PP5S, _PP5G, _PP5S + 4 * _PP5G + 3 * _PP5, ls)
    a("Pasokon P7", 0x73, 640, 496, _PP7S, _PP7G, _PP7S + 4 * _PP7G + 3 * _PP7, ls)
    return t


_MODE_TABLE = _build_mode_table()

# VIS constants
_VIS_SYNC_HZ = 1200.0
_VIS_BIT_DUR_S = 0.030
_VIS_DATA_BITS = 7


def _analytic_signal(x: NDArray) -> NDArray[np.complex128]:
    return signal.hilbert(np.asarray(x))


def _inst_freq(x: NDArray, fs: float) -> NDArray[np.float64]:
    z = _analytic_signal(x)
    phase = np.unwrap(np.angle(z))
    if phase.size < 2:
        return np.zeros_like(phase, dtype=np.float64)
    diffs = np.diff(phase) * (fs / (2.0 * np.pi))
    return np.concatenate([diffs, diffs[-1:]])


def _bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _bp(x: NDArray, fs: int) -> NDArray:
    if x.size < _BP_MIN_SAMPLES:
        return x
    try:
        sos = _bandpass_sos(_BP_LOW_HZ, _BP_HIGH_HZ, fs, _BP_ORDER)
    except ValueError:
        return x
    return sosfiltfilt(sos, x)


def _sanitize(arr: NDArray) -> NDArray:
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _find_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = (np.where(diff == 1)[0] + 1).tolist()
    ends = (np.where(diff == -1)[0] + 1).tolist()
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(int(mask.size))
    return list(zip(starts, ends, strict=True))


def _detect_vis(samples: NDArray, fs: int) -> tuple[int, int] | None:
    """Detect SSTV VIS header in audio samples.

    pysstv's VIS structure:
      1900 Hz leader × 300ms
      1200 Hz sync × 10ms
      1900 Hz second leader × 300ms
      1200 Hz start bit × 30ms
      7 × data bits × 30ms (1100 Hz = 1, 1300 Hz = 0)
      1200 Hz stop bit × 30ms

    Returns (vis_code, sync_end_sample) or None.
    """
    arr = np.asarray(samples)
    if arr.ndim != 1 or arr.size == 0:
        return None
    inst = _inst_freq(arr, fs)
    smooth_n = max(1, int(round(0.002 * fs)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        inst = np.convolve(inst, kernel, mode="same")

    leader_hz = 1900.0
    sync_hz = 1200.0
    leader_tol = 200.0
    min_leader_ms = 200.0
    min_sync_ms = 3.0
    max_sync_ms = 40.0
    bit_dur_ms = 30.0

    min_leader_samp = int(min_leader_ms * fs / 1000.0)
    min_sync_samp = int(min_sync_ms * fs / 1000.0)
    max_sync_samp = int(max_sync_ms * fs / 1000.0)
    bit_samp = int(bit_dur_ms * fs / 1000.0)

    is_leader = np.abs(inst - leader_hz) < leader_tol
    is_sync = np.abs(inst - sync_hz) < leader_tol

    leader_runs = _find_runs(is_leader)
    for l_start, l_end in leader_runs:
        if (l_end - l_start) < min_leader_samp:
            continue

        search_end = min(l_end + max_sync_samp * 3, inst.size)
        sync_region = is_sync[l_end:search_end]
        sync_runs = _find_runs(sync_region)

        for s_rel_start, s_rel_end in sync_runs:
            sync_len = s_rel_end - s_rel_start
            if sync_len < min_sync_samp or sync_len > max_sync_samp:
                continue

            sync_end = l_end + s_rel_end

            second_leader_start = sync_end
            second_leader_region = is_leader[second_leader_start:
                second_leader_start + min_leader_samp + int(200 * fs / 1000)]
            sl_runs = _find_runs(second_leader_region)
            if not sl_runs:
                continue
            sl_start, sl_end_abs = sl_runs[0]
            if (sl_end_abs - sl_start) < min_leader_samp:
                continue

            vis_bit_start = second_leader_start + sl_end_abs
            n_bits = _VIS_DATA_BITS + 2
            vis_end = vis_bit_start + n_bits * bit_samp
            if vis_end > inst.size:
                continue

            raw_bits = []
            for bit_i in range(n_bits):
                b_start = vis_bit_start + bit_i * bit_samp
                b_end = b_start + bit_samp
                bit_freq = float(np.mean(inst[b_start:b_end]))
                raw_bits.append(bit_freq)

            data_bits = []
            for bit_freq in raw_bits[1:1 + _VIS_DATA_BITS]:
                data_bits.append(1 if bit_freq < 1200.0 else 0)

            vis_code = 0
            for i, b in enumerate(data_bits):
                vis_code |= b << i

            parity_bit = 1 if raw_bits[1 + _VIS_DATA_BITS] < 1200.0 else 0
            parity = data_bits.count(1) + parity_bit
            if parity % 2 != 0:
                _debug.debug("VIS parity failed for code 0x%02X", vis_code)
                continue

            if vis_code in _MODE_TABLE:
                mode = _MODE_TABLE[vis_code]
                _debug.debug(
                    "VIS detected: 0x%02X (%s) at sample %d, vis_end=%d",
                    vis_code, mode.name, l_start, vis_end,
                )
                return (vis_code, vis_end)
            _debug.debug("VIS code 0x%02X not in mode table", vis_code)

    return None


def _find_sync_candidates(
    inst: NDArray[np.float64], fs: int, mode: _ModeSpec, start: int
) -> list[int]:
    threshold = 1400.0
    min_gap = int(mode.line_time_ms * 0.80 * fs / 1000.0)
    max_gap = int(mode.line_time_ms * 1.20 * fs / 1000.0)
    pulse_samp = max(1, int(mode.sync_pulse_ms * 0.5 * fs / 1000.0))
    max_run_samp = int(mode.sync_pulse_ms * 3.0 * fs / 1000.0)
    is_sync = inst[start:] < threshold
    runs = _find_runs(is_sync)
    candidates = []
    for run_start, run_end in runs:
        run_len = run_end - run_start
        if run_len < pulse_samp:
            continue
        if run_len > max_run_samp:
            continue
        abs_pos = start + run_end
        if candidates:
            gap = abs_pos - candidates[-1]
            if min_gap <= gap <= max_gap:
                candidates.append(abs_pos)
            elif gap < min_gap and candidates:
                candidates[-1] = abs_pos
        else:
            candidates.append(abs_pos)
    return candidates


def _walk_sync_grid(
    candidates: list[int], fs: int, line_time_ms: float, tolerance: float = 0.15
) -> list[int]:
    if len(candidates) < 3:
        return candidates
    diffs = np.diff(candidates).astype(np.float64)
    median_dt = float(np.median(diffs))
    expected_dt = line_time_ms * fs / 1000.0
    if abs(median_dt - expected_dt) / expected_dt > tolerance:
        _debug.debug(
            "Sync grid mismatch: median=%.1f expected=%.1f", median_dt, expected_dt
        )
        return candidates
    best_start = 0
    best_score = 0
    for i in range(min(len(candidates), 5)):
        score = 0
        pos = candidates[i]
        for j in range(i, len(candidates)):
            expected = pos + (j - i) * median_dt
            actual = candidates[j]
            err = abs(actual - expected) / expected_dt
            if err < tolerance:
                score += 1
        if score > best_score:
            best_score = score
            best_start = i
    aligned = []
    ref = candidates[best_start]
    for k in range(best_score):
        expected = ref + k * median_dt
        actual = candidates[best_start + k]
        if abs(actual - expected) < expected_dt * 0.5:
            aligned.append(actual)
        else:
            aligned.append(int(round(expected)))
    return aligned


def _fit_line_timing(
    line_starts: list[int], line_time_ms: float, fs: int
) -> tuple[float, float]:
    if len(line_starts) < 2:
        return (line_time_ms, 0.0)
    expected_dt = line_time_ms * fs / 1000.0
    times = np.array(line_starts, dtype=np.float64)
    indices = np.arange(len(times), dtype=np.float64)
    if len(times) >= 3:
        coeffs = np.polyfit(indices, times, 1)
        fitted_dt = coeffs[0]
        slant_ratio = (fitted_dt - expected_dt) / expected_dt
        return (fitted_dt, slant_ratio)
    return (expected_dt, 0.0)


def _slant_corrected_line_starts(
    inst: NDArray[np.float64], fs: int, mode: _ModeSpec, start: int
) -> tuple[list[int], float]:
    candidates = _find_sync_candidates(inst, fs, mode, start)
    aligned = _walk_sync_grid(candidates, fs, mode.line_time_ms)
    fitted_dt, slant = _fit_line_timing(aligned, mode.line_time_ms, fs)
    if abs(slant) > 0.01:
        _debug.debug("Slant correction: %.4f%% over %d lines", slant * 100, len(aligned))
    corrected = []
    for i, pos in enumerate(aligned):
        expected = aligned[0] + i * fitted_dt
        corrected.append(int(round(expected)))
    return corrected, fitted_dt


def _sample_pixel(
    inst: NDArray[np.float64], center: int, width: int
) -> float:
    lo = max(0, center - width // 2)
    hi = min(inst.size, center + width // 2)
    if hi <= lo:
        return 0.0
    segment = inst[lo:hi]
    return float(np.mean(segment))


def _sample_scan(
    inst: NDArray[np.float64], start: int, end: int, npix: int
) -> NDArray[np.float64]:
    if end <= start or npix <= 0:
        return np.zeros(npix, dtype=np.float64)
    end = min(end, inst.size)
    start = max(0, start)
    if end <= start:
        return np.zeros(npix, dtype=np.float64)
    if end - start < npix:
        vals = np.interp(
            np.linspace(0, 1, npix),
            np.linspace(0, 1, end - start),
            inst[start:end],
        )
        return vals
    edges = np.linspace(start, end, npix + 1, dtype=np.int64)
    vals = np.zeros(npix, dtype=np.float64)
    for i in range(npix):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi > lo:
            vals[i] = float(np.mean(inst[lo:hi]))
        elif lo < inst.size:
            vals[i] = float(inst[lo])
    return vals


def _sample_pixels(
    inst: NDArray[np.float64], start: int, end: int, npix: int
) -> NDArray[np.float64]:
    return _sample_scan(inst, start, end, npix)


def _freq_to_luminance(freq: float) -> float:
    return np.clip((freq - COLOR_BLACK_HZ) / COLOR_RANGE * 255.0, 0, 255)


def _freq_to_chroma(freq: float) -> float:
    return np.clip((freq - COLOR_BLACK_HZ) / COLOR_RANGE * 255.0, 0, 255)


def _ycbcr_to_rgb(y: NDArray, cb: NDArray, cr: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    yf = y.astype(np.float64) / 255.0
    cbf = (cb.astype(np.float64) / 255.0) - 0.5
    crf = (cr.astype(np.float64) / 255.0) - 0.5
    r = yf + 1.402 * crf
    g = yf - 0.344136 * cbf - 0.714136 * crf
    b = yf + 1.772 * cbf
    return (
        np.clip(r * 255, 0, 255).astype(np.uint8),
        np.clip(g * 255, 0, 255).astype(np.uint8),
        np.clip(b * 255, 0, 255).astype(np.uint8),
    )


def _decode_martin_rgb(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w, h = mode.width, mode.display_height
    img = np.zeros((h, 3, w), dtype=np.uint8)
    total_line = mode.line_time_ms
    scan_width_ms = (total_line - mode.sync_pulse_ms - 4 * mode.sync_porch_ms) / 3.0
    ch_block_ms = scan_width_ms + mode.sync_porch_ms
    for line_i, line_samp in enumerate(line_starts[:h]):
        if line_i >= h:
            break
        for ch_idx, ch_name in enumerate(mode.scan_order):
            ch_offset_ms = (mode.sync_pulse_ms + mode.sync_porch_ms +
                           ch_idx * ch_block_ms)
            scan_start = line_samp + int(ch_offset_ms * fs / 1000.0)
            scan_end = scan_start + int(scan_width_ms * fs / 1000.0)
            pixels = _sample_scan(inst, scan_start, scan_end, w)
            lum = _freq_to_luminance(pixels)
            img[line_i, ch_idx] = lum.astype(np.uint8)
    rgb_img = np.transpose(img, (0, 2, 1))
    r_idx = mode.scan_order.index("r")
    g_idx = mode.scan_order.index("g")
    b_idx = mode.scan_order.index("b")
    r = rgb_img[:, :, r_idx]
    g = rgb_img[:, :, g_idx]
    b = rgb_img[:, :, b_idx]
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _decode_scottie_rgb(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w, h = mode.width, mode.display_height
    img = np.zeros((h, 3, w), dtype=np.uint8)
    scan_w_ms = (mode.line_time_ms - mode.sync_pulse_ms - 6 * mode.sync_porch_ms) / 3.0
    ch_block_ms = scan_w_ms + 2 * mode.sync_porch_ms
    # Physical channel order from sync_end for BEFORE_RED: R, G, B
    physical_order = "rgb"
    for line_i, line_samp in enumerate(line_starts[:h]):
        if line_i >= h:
            break
        for ch_idx, ch_name in enumerate(mode.scan_order):
            phys_idx = physical_order.index(ch_name)
            ch_offset_ms = mode.sync_porch_ms + phys_idx * ch_block_ms
            scan_start = line_samp + int(ch_offset_ms * fs / 1000.0)
            scan_end = scan_start + int(scan_w_ms * fs / 1000.0)
            pixels = _sample_scan(inst, scan_start, scan_end, w)
            lum = _freq_to_luminance(pixels)
            img[line_i, ch_idx] = lum.astype(np.uint8)
    rgb_img = np.transpose(img, (0, 2, 1))
    r_idx = mode.scan_order.index("r")
    g_idx = mode.scan_order.index("g")
    b_idx = mode.scan_order.index("b")
    r = rgb_img[:, :, r_idx]
    g = rgb_img[:, :, g_idx]
    b = rgb_img[:, :, b_idx]
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _decode_pd(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w = mode.width
    total_height = mode.display_height
    trans_lines = len(line_starts[:mode.height])
    img = np.zeros((total_height, 3, w), dtype=np.uint8)
    scan_time_ms = (mode.line_time_ms - mode.sync_pulse_ms - mode.sync_porch_ms) / 4.0
    scan_samp = int(scan_time_ms * fs / 1000.0)
    for line_i, line_samp in enumerate(line_starts[:mode.height]):
        pos = line_samp + int(mode.sync_pulse_ms * fs / 1000.0)
        pos += int(mode.sync_porch_ms * fs / 1000.0)
        y0_pixels = _sample_scan(inst, pos, pos + scan_samp, w)
        pos += scan_samp
        cr_half = _sample_scan(inst, pos, pos + scan_samp, w // 2)
        pos += scan_samp
        cb_half = _sample_scan(inst, pos, pos + scan_samp, w // 2)
        pos += scan_samp
        y1_pixels = _sample_scan(inst, pos, pos + scan_samp, w)
        pos += scan_samp
        cb_full = np.interp(np.arange(w), np.arange(w // 2) * 2 + 1, cb_half)
        cr_full = np.interp(np.arange(w), np.arange(w // 2) * 2 + 1, cr_half)
        y0_val = _freq_to_luminance(y0_pixels)
        y1_val = _freq_to_luminance(y1_pixels)
        cb_val = _freq_to_chroma(cb_full)
        cr_val = _freq_to_chroma(cr_full)
        r0, g0, b0 = _ycbcr_to_rgb(y0_val, cb_val, cr_val)
        r1, g1, b1 = _ycbcr_to_rgb(y1_val, cb_val, cr_val)
        row_even = line_i * 2
        row_odd = row_even + 1
        if row_even < total_height:
            img[row_even, 0] = r0
            img[row_even, 1] = g0
            img[row_even, 2] = b0
        if row_odd < total_height:
            img[row_odd, 0] = r1
            img[row_odd, 1] = g1
            img[row_odd, 2] = b1
    return Image.fromarray(np.transpose(img, (0, 2, 1)), "RGB")


def _decode_robot36(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w, h = mode.width, mode.display_height
    img = np.zeros((h, 3, w), dtype=np.uint8)
    for line_i, line_samp in enumerate(line_starts[:h]):
        pos = line_samp
        sync_end = pos + int(mode.sync_pulse_ms * fs / 1000.0)
        pos = sync_end + int(mode.sync_porch_ms * fs / 1000.0)
        y_end = pos + int(_R36_Y * fs / 1000.0)
        y_pixels = _sample_scan(inst, pos, y_end, w)
        lum = _freq_to_luminance(y_pixels)
        img[line_i, 0] = lum.astype(np.uint8)
        pos = y_end + int(_R36_G * fs / 1000.0)
        pos += int(_R36_P * fs / 1000.0)
        if line_i % 2 == 0:
            c_end = pos + int(_R36_C * fs / 1000.0)
            cr_pixels = _sample_scan(inst, pos, c_end, w // 2)
            cr_full = np.interp(np.arange(w), np.arange(w // 2) * 2 + 1, cr_pixels)
            img[line_i, 1] = 128
            img[line_i, 2] = _freq_to_chroma(cr_full).astype(np.uint8)
        else:
            c_end = pos + int(_R36_C * fs / 1000.0)
            cb_pixels = _sample_scan(inst, pos, c_end, w // 2)
            cb_full = np.interp(np.arange(w), np.arange(w // 2) * 2 + 1, cb_pixels)
            img[line_i, 1] = _freq_to_chroma(cb_full).astype(np.uint8)
            img[line_i, 2] = 128
    y_ch = img[:, 0, :]
    cb_ch = img[:, 1, :]
    cr_ch = img[:, 2, :]
    r, g, b = _ycbcr_to_rgb(y_ch, cb_ch, cr_ch)
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _decode_wraase_rgb(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w, h = mode.width, mode.display_height
    img = np.zeros((h, 3, w), dtype=np.uint8)
    for line_i, line_samp in enumerate(line_starts[:h]):
        pos = line_samp
        sync_end = pos + int(mode.sync_pulse_ms * fs / 1000.0)
        pos = sync_end + int(mode.sync_porch_ms * fs / 1000.0)
        total = mode.line_time_ms - mode.sync_pulse_ms - mode.sync_porch_ms
        ch_w = total / 3.0
        for ch_idx, ch_name in enumerate(mode.scan_order):
            ch_end = pos + int(ch_w * fs / 1000.0)
            pixels = _sample_scan(inst, pos, ch_end, w)
            lum = _freq_to_luminance(pixels)
            img[line_i, ch_idx] = lum.astype(np.uint8)
            pos = ch_end
    rgb_img = np.transpose(img, (0, 2, 1))
    r_idx = mode.scan_order.index("r")
    g_idx = mode.scan_order.index("g")
    b_idx = mode.scan_order.index("b")
    r = rgb_img[:, :, r_idx]
    g = rgb_img[:, :, g_idx]
    b = rgb_img[:, :, b_idx]
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _decode_pasokon_rgb(
    inst: NDArray[np.float64], line_starts: list[int], fitted_dt: float,
    mode: _ModeSpec, fs: int
) -> Image.Image:
    w, h = mode.width, mode.display_height
    img = np.zeros((h, 3, w), dtype=np.uint8)
    for line_i, line_samp in enumerate(line_starts[:h]):
        pos = line_samp
        sync_pulse_s = mode.sync_pulse_ms * fs / 1000.0
        pos += int(sync_pulse_s)
        porch_s = mode.sync_porch_ms * fs / 1000.0
        total_line_s = mode.line_time_ms * fs / 1000.0
        usable = total_line_s - sync_pulse_s - 4 * porch_s
        ch_w = usable / 3.0
        for ch_idx, ch_name in enumerate(mode.scan_order):
            pos += int(porch_s)
            ch_end = pos + int(ch_w)
            pixels = _sample_scan(inst, pos, ch_end, w)
            lum = _freq_to_luminance(pixels)
            img[line_i, ch_idx] = lum.astype(np.uint8)
            pos = ch_end
    rgb_img = np.transpose(img, (0, 2, 1))
    r_idx = mode.scan_order.index("r")
    g_idx = mode.scan_order.index("g")
    b_idx = mode.scan_order.index("b")
    r = rgb_img[:, :, r_idx]
    g = rgb_img[:, :, g_idx]
    b = rgb_img[:, :, b_idx]
    return Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")


def _decode_wav(
    samples: NDArray, fs: int, mode: _ModeSpec | None = None
) -> Image.Image | None:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        return None
    bp = _bp(arr, fs)
    inst = _inst_freq(bp, fs)
    inst = _sanitize(inst)
    smooth_n = max(1, int(round(0.002 * fs)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        inst = np.convolve(inst, kernel, mode="same")
    if mode is None:
        vis = _detect_vis(arr, fs)
        if vis is None:
            _debug.debug("No VIS detected")
            return None
        vis_code, sync_end = vis
        mode = _MODE_TABLE.get(vis_code)
        if mode is None:
            _debug.debug("Unknown VIS code 0x%02X", vis_code)
            return None
    else:
        sync_end = 0
    _debug.debug("Decoding %s at %d Hz", mode.name, fs)
    line_starts, fitted_dt = _slant_corrected_line_starts(inst, fs, mode, sync_end)
    if not line_starts:
        _debug.debug("No sync lines found")
        return None
    _debug.debug("Found %d lines (expected %d)", len(line_starts), mode.height)
    vis_code = mode.vis_code
    if vis_code in (0x2C, 0x28, 0x24, 0x20):
        img = _decode_martin_rgb(inst, line_starts, fitted_dt, mode, fs)
    elif vis_code in (0x3C, 0x38, 0x4C, 0x34, 0x30):
        img = _decode_scottie_rgb(inst, line_starts, fitted_dt, mode, fs)
    elif vis_code in (0x5D, 0x63, 0x5F, 0x62, 0x60, 0x61, 0x5E):
        img = _decode_pd(inst, line_starts, fitted_dt, mode, fs)
    elif vis_code == 0x08:
        img = _decode_robot36(inst, line_starts, fitted_dt, mode, fs)
    elif vis_code in (0x3F, 0x37):
        img = _decode_wraase_rgb(inst, line_starts, fitted_dt, mode, fs)
    elif vis_code in (0x71, 0x72, 0x73):
        img = _decode_pasokon_rgb(inst, line_starts, fitted_dt, mode, fs)
    else:
        _debug.debug("No decoder for VIS 0x%02X", vis_code)
        return None
    _debug.debug("Decoded image: %dx%d", img.width, img.height)
    return img


class SstvDecoder:
    def __init__(
        self,
        mode: str | None = None,
        sample_rate: int = 48000,
        callback: Callable[[NDArray, str], None] | None = None,
    ):
        self._sample_rate = sample_rate
        self._callback = callback
        self._mode_key = mode
        self._buf = np.array([], dtype=np.float64)
        self._mode_spec: _ModeSpec | None = None
        if mode and mode in MODES:
            self._mode_spec = _MODE_TABLE.get(MODES[mode].vis_code)
        self._decoded = False
        self._vis_found = False
        self._sync_end = 0

    def feed(self, pcm: NDArray) -> None:
        if self._decoded:
            return
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float64)])
        min_samples = 4800
        if self._buf.size < min_samples:
            return
        if not self._vis_found:
            self._try_detect()
        if self._vis_found:
            self._try_decode()

    def _try_detect(self) -> None:
        if self._mode_spec is not None:
            self._vis_found = True
            self._sync_end = 0
            return
        vis = _detect_vis(self._buf, self._sample_rate)
        if vis is not None:
            vis_code, sync_end = vis
            self._mode_spec = _MODE_TABLE.get(vis_code)
            if self._mode_spec is not None:
                self._vis_found = True
                self._sync_end = sync_end
                _debug.debug("Streaming VIS detected: %s", self._mode_spec.name)

    def _try_decode(self) -> None:
        mode = self._mode_spec
        if mode is None:
            return
        line_time_s = mode.line_time_ms / 1000.0
        needed_after_sync = line_time_s * (mode.height + 2)
        min_sample = self._sync_end + int(needed_after_sync * self._sample_rate)
        if self._buf.size < min_sample:
            return
        img = _decode_wav(self._buf, self._sample_rate, mode)
        if img is not None:
            self._decoded = True
            arr = np.array(img)
            if self._callback is not None:
                self._callback(arr, mode.name)
            _debug.debug("Streaming decode complete: %s", mode.name)

    def reset(self) -> None:
        self._buf = np.array([], dtype=np.float64)
        self._mode_spec = None
        self._decoded = False
        self._vis_found = False
        self._sync_end = 0
