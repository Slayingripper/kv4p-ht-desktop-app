import threading
import time
from collections.abc import Callable
from typing import ClassVar


class BandPlan:
    BANDS: ClassVar[dict[str, tuple[int, int]]] = {
        "6m": (50, 54),
        "2m": (144, 148),
        "1.25m": (222, 225),
        "70cm": (420, 450),
        "33cm": (902, 928),
    }

    PRESETS: ClassVar[dict[str, list[float]]] = {
        "2m": [
            144.800, 144.850, 144.900, 144.950,
            145.000, 145.100, 145.200, 145.300,
            145.400, 145.500, 145.600, 145.700,
            145.800, 145.900, 146.000, 146.100,
            146.200, 146.300, 146.400, 146.500,
            146.520, 146.550, 146.600, 146.700,
            146.800, 146.850, 146.900, 146.940,
            146.970, 147.000, 147.030, 147.060,
            147.090, 147.120, 147.150, 147.180,
            147.210, 147.240, 147.270, 147.300,
            147.330, 147.360, 147.390, 147.420,
            147.450, 147.480, 147.510, 147.540,
            147.570, 147.600,
        ],
        "70cm": [
            420.000, 421.000, 422.000, 423.000,
            424.000, 425.000, 426.000, 427.000,
            428.000, 429.000, 430.000, 431.000,
            432.000, 433.000, 434.000, 435.000,
            436.000, 437.000, 438.000, 439.000,
            440.000, 441.000, 442.000, 443.000,
            444.000, 445.000, 446.000, 447.000,
            448.000, 449.000,
        ],
    }

    @classmethod
    def get_preset_list(cls, band: str = "2m") -> list:
        return list(cls.PRESETS.get(band, []))


class FrequencyScanner:
    def __init__(
        self,
        set_freq_callback: Callable[[float], None],
        on_signal_callback: Callable[[float, float], None],
    ):
        self._set_freq = set_freq_callback
        self._on_signal = on_signal_callback
        self._on_freq_change: Callable[[float], None] | None = None
        self._frequencies: list[float] = []
        self._dwell_ms: int = 500
        self._squelch_threshold: float = 3.0
        self._hold_seconds: float = 3.0
        self._scanning = False
        self._paused = False
        self._is_paused = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def on_freq_change(self) -> Callable[[float], None] | None:
        return self._on_freq_change

    @on_freq_change.setter
    def on_freq_change(self, cb: Callable[[float], None] | None) -> None:
        self._on_freq_change = cb

    @property
    def on_signal(self) -> Callable[[float, float], None] | None:
        return self._on_signal

    @on_signal.setter
    def on_signal(self, cb: Callable[[float, float], None] | None) -> None:
        self._on_signal = cb

    def start_scan(
        self,
        frequencies: list[float],
        dwell_ms: int = 500,
        squelch_threshold: float = 3.0,
    ) -> None:
        self.stop_scan()
        self._frequencies = list(frequencies)
        self._dwell_ms = dwell_ms
        self._squelch_threshold = squelch_threshold
        self._stop_event.clear()
        self._paused = False
        self._is_paused = False
        self._scanning = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def stop_scan(self) -> None:
        self._scanning = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def pause(self) -> None:
        self._paused = True
        self._is_paused = True

    def resume(self) -> None:
        self._paused = False
        self._is_paused = False

    def add_frequency(self, freq_mhz: float) -> None:
        with self._lock:
            if freq_mhz not in self._frequencies:
                self._frequencies.append(freq_mhz)

    def remove_frequency(self, freq_mhz: float) -> None:
        with self._lock:
            if freq_mhz in self._frequencies:
                self._frequencies.remove(freq_mhz)

    def get_scan_list(self) -> list[float]:
        with self._lock:
            return list(self._frequencies)

    def is_scanning(self) -> bool:
        return self._scanning

    def _scan_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._paused:
                time.sleep(0.05)
                continue
            with self._lock:
                freqs = list(self._frequencies)
            if not freqs:
                time.sleep(0.1)
                continue
            for freq in freqs:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    if freq not in self._frequencies:
                        continue
                self._set_freq(freq)
                if self._on_freq_change:
                    self._on_freq_change(freq)
                time.sleep(self._dwell_ms / 1000.0)
                if self._stop_event.is_set():
                    return
                rssi = self._read_rssi()
                if rssi is not None and rssi > self._squelch_threshold:
                    if self._on_signal:
                        self._on_signal(freq, rssi)
                    self._hold(freq)
                if self._stop_event.is_set():
                    return

    def _read_rssi(self) -> float | None:
        """Read current RSSI from the radio. Returns value or None if unavailable."""
        return getattr(self, '_last_rssi', None)

    def set_rssi(self, rssi: float):
        """Called by the host to update the scanner's RSSI reading."""
        self._last_rssi = rssi

    def _hold(self, freq: float) -> None:
        self._set_freq(freq)
        if self._on_freq_change:
            self._on_freq_change(freq)
        elapsed = 0.0
        while elapsed < self._hold_seconds and not self._stop_event.is_set():
            time.sleep(0.1)
            elapsed += 0.1
