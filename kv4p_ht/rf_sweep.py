"""
RF spectrum sweep engine.

Rapidly steps through a frequency range, reads RSSI at each step,
and produces power-vs-frequency data for spectrum display.

Uses the SA818's GROUP command to change frequency and reads RSSI
back from the ESP32's SMETER reports.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np


class RfSweeper:
    """Sweeps a frequency range and measures RSSI at each step."""

    def __init__(
        self,
        set_freq_callback: Callable[[float], None],
        log_fn: Callable[[str], None] | None = None,
    ):
        self._set_freq = set_freq_callback
        self._log_fn = log_fn

        self._start_mhz = 144.0
        self._stop_mhz = 148.0
        self._step_khz = 25.0
        self._settle_ms = 30

        self._sweeping = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._last_rssi: float = 0.0
        self._rssi_lock = threading.Lock()

        self._on_sweep_point: Callable[[float, float], None] | None = None
        self._on_sweep_complete: Callable[[np.ndarray, np.ndarray], None] | None = None
        self._on_sweep_progress: Callable[[int, int], None] | None = None

    @property
    def start_mhz(self) -> float:
        return self._start_mhz

    @start_mhz.setter
    def start_mhz(self, v: float):
        self._start_mhz = v

    @property
    def stop_mhz(self) -> float:
        return self._stop_mhz

    @stop_mhz.setter
    def stop_mhz(self, v: float):
        self._stop_mhz = v

    @property
    def step_khz(self) -> float:
        return self._step_khz

    @step_khz.setter
    def step_khz(self, v: float):
        self._step_khz = max(1.0, v)

    @property
    def settle_ms(self) -> int:
        return self._settle_ms

    @settle_ms.setter
    def settle_ms(self, v: int):
        self._settle_ms = max(5, v)

    @property
    def on_sweep_point(self):
        return self._on_sweep_point

    @on_sweep_point.setter
    def on_sweep_point(self, cb):
        self._on_sweep_point = cb

    @property
    def on_sweep_complete(self):
        return self._on_sweep_complete

    @on_sweep_complete.setter
    def on_sweep_complete(self, cb):
        self._on_sweep_complete = cb

    @property
    def on_sweep_progress(self):
        return self._on_sweep_progress

    @on_sweep_progress.setter
    def on_sweep_progress(self, cb):
        self._on_sweep_progress = cb

    def set_rssi(self, rssi: float):
        with self._rssi_lock:
            self._last_rssi = rssi

    def _read_rssi(self) -> float:
        with self._rssi_lock:
            return self._last_rssi

    @property
    def is_sweeping(self) -> bool:
        return self._sweeping

    @property
    def num_steps(self) -> int:
        span = abs(self._stop_mhz - self._start_mhz)
        return max(1, int(span * 1000 / self._step_khz) + 1)

    @property
    def estimated_sweep_time(self) -> float:
        return self.num_steps * self._settle_ms / 1000.0

    def build_freq_axis(self) -> np.ndarray:
        return np.linspace(self._start_mhz, self._stop_mhz, self.num_steps, dtype=np.float64)

    def start(self):
        if self._sweeping:
            return
        self._sweeping = True
        self._paused = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sweep_loop, daemon=True)
        self._thread.start()
        self._log(f"RF sweep started: {self._start_mhz:.3f}-{self._stop_mhz:.3f} MHz, "
                   f"step={self._step_khz:.1f} kHz, ~{self.estimated_sweep_time:.1f}s per sweep")

    def stop(self):
        self._sweeping = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._log("RF sweep stopped")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _sweep_loop(self):
        while not self._stop_event.is_set():
            if self._paused:
                time.sleep(0.05)
                continue

            freq_axis = self.build_freq_axis()
            num_steps = len(freq_axis)
            rssi_values = np.zeros(num_steps, dtype=np.float64)

            for i, freq in enumerate(freq_axis):
                if self._stop_event.is_set():
                    return

                self._set_freq(freq)

                time.sleep(self._settle_ms / 1000.0)

                rssi_values[i] = self._read_rssi()

                if self._on_sweep_point:
                    self._on_sweep_point(freq, rssi_values[i])

                if self._on_sweep_progress:
                    self._on_sweep_progress(i + 1, num_steps)

            if self._on_sweep_complete and not self._stop_event.is_set():
                self._on_sweep_complete(freq_axis, rssi_values)

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)


def rssi_to_dbm(rssi: float) -> float:
    """Convert SA818 RSSI raw value to approximate dBm."""
    if rssi <= 0:
        return -120.0
    return 10.0 * np.log10(max(1.0, rssi)) - 120.0
