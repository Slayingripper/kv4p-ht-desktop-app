"""
FFT-based spectrum analyzer and waterfall for audio-rate signals.

Processes PCM audio from the radio's RX path and produces:
  - Real-time FFT power spectrum (dB)
  - Waterfall history (scrolling spectrogram)
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

import numpy as np


class SpectrumAnalyzer:
    """Compute FFT-based power spectrum from streaming PCM audio."""

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 2048,
        averaging: int = 4,
        window_type: str = "hann",
        callback: Callable[[np.ndarray, np.ndarray, float], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.averaging = max(1, averaging)
        self._callback = callback

        self._window = self._make_window(window_type, fft_size)
        self._overlap = fft_size // 2
        self._buffer = np.zeros(0, dtype=np.float32)
        self._spectrum_ema: np.ndarray | None = None
        self._alpha = 2.0 / (self.averaging + 1)

        self._center_freq = 0.0
        self._span = float(sample_rate)
        self._ref_level = 0.0
        self._rbw = sample_rate / fft_size

        self._peak_hold: np.ndarray | None = None
        self._peak_decay = 0.995
        self._min_hold: np.ndarray | None = None

    @staticmethod
    def _make_window(name: str, n: int) -> np.ndarray:
        if name == "hann":
            return np.hanning(n).astype(np.float32)
        elif name == "hamming":
            return np.hamming(n).astype(np.float32)
        elif name == "blackman":
            return np.blackman(n).astype(np.float32)
        elif name == "blackmanharris":
            return np.blackmanharris(n).astype(np.float32)
        elif name == "kaiser":
            return np.kaiser(n, beta=14).astype(np.float32)
        return np.ones(n, dtype=np.float32)

    def set_center_freq(self, freq_mhz: float):
        self._center_freq = freq_mhz

    def set_span(self, span_hz: float):
        self._span = max(1.0, span_hz)

    def set_ref_level(self, db: float):
        self._ref_level = db

    def set_averaging(self, n: int):
        self.averaging = max(1, n)

    def set_fft_size(self, size: int):
        self.fft_size = max(64, min(16384, size))
        self._window = self._make_window("hann", self.fft_size)
        self._rbw = self.sample_rate / self.fft_size

    @property
    def rbw(self) -> float:
        return self._rbw

    @property
    def frequency_axis(self) -> np.ndarray:
        """Return frequency axis in Hz relative to center."""
        n = self.fft_size
        freqs = np.fft.fftfreq(n, 1.0 / self.sample_rate)
        return np.fft.fftshift(freqs)

    @property
    def frequency_axis_mhz(self) -> np.ndarray:
        return (self.frequency_axis / 1e6) + self._center_freq

    def feed(self, samples: np.ndarray):
        """Feed PCM float32 samples.  Calls callback with (freq_mhz, power_db, timestamp)."""
        if samples.ndim > 1:
            samples = samples.ravel()
        self._buffer = np.concatenate([self._buffer, samples])

        while len(self._buffer) >= self.fft_size:
            chunk = self._buffer[: self.fft_size]
            self._buffer = self._buffer[self.fft_size - self._overlap:]

            spectrum = self._compute(chunk)
            if self._spectrum_ema is None:
                self._spectrum_ema = spectrum
            else:
                self._spectrum_ema = self._alpha * spectrum + (1 - self._alpha) * self._spectrum_ema

            if self._peak_hold is None:
                self._peak_hold = self._spectrum_ema.copy()
            else:
                self._peak_hold = np.maximum(self._peak_hold * self._peak_decay, self._spectrum_ema)

            if self._min_hold is None:
                self._min_hold = self._spectrum_ema.copy()
            else:
                self._min_hold = np.minimum(self._min_hold, self._spectrum_ema)

            if self._callback:
                freq_mhz = self.frequency_axis_mhz
                self._callback(freq_mhz, self._spectrum_ema.copy(), time.monotonic())

    def _compute(self, chunk: np.ndarray) -> np.ndarray:
        windowed = chunk * self._window
        fft_result = np.fft.fft(windowed, n=self.fft_size)
        mag = np.abs(fft_result)
        mag = np.maximum(mag, 1e-20)
        db = 20.0 * np.log10(mag) - self._ref_level
        return np.fft.fftshift(db)

    def get_spectrum(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._spectrum_ema is None:
            return None
        return self.frequency_axis_mhz.copy(), self._spectrum_ema.copy()

    def get_peak_hold(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._peak_hold is None:
            return None
        return self.frequency_axis_mhz, self._peak_hold

    def get_min_hold(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._min_hold is None:
            return None
        return self.frequency_axis_mhz, self._min_hold

    def reset_peak_hold(self):
        self._peak_hold = None

    def reset_min_hold(self):
        self._min_hold = None

    def get_waterfall_row(self) -> np.ndarray | None:
        if self._spectrum_ema is None:
            return None
        return self._spectrum_ema.copy()


class WaterfallBuffer:
    """Circular buffer of spectrum rows for waterfall display."""

    def __init__(self, max_rows: int = 256, num_bins: int = 0):
        self.max_rows = max_rows
        self._rows: deque = deque(maxlen=max_rows)
        self._num_bins = num_bins

    def push(self, row: np.ndarray):
        if self._num_bins == 0:
            self._num_bins = len(row)
        self._rows.append(row)

    def get_matrix(self) -> np.ndarray | None:
        if not self._rows:
            return None
        rows = list(self._rows)
        return np.array(rows, dtype=np.float32)

    @property
    def num_rows(self) -> int:
        return len(self._rows)

    @property
    def num_bins(self) -> int:
        return self._num_bins

    def clear(self):
        self._rows.clear()
        self._num_bins = 0
