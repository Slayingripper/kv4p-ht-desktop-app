from __future__ import annotations

import time

from kv4p_ht.scanner import BandPlan, FrequencyScanner


class TestBandPlan:
    def test_bands(self):
        assert BandPlan.BANDS["2m"] == (144, 148)
        assert BandPlan.BANDS["70cm"] == (420, 450)
        assert BandPlan.BANDS["6m"] == (50, 54)

    def test_preset_list_returns_copy(self):
        lst = BandPlan.get_preset_list("2m")
        lst.append(999.0)
        assert 999.0 not in BandPlan.get_preset_list("2m")
        assert len(BandPlan.get_preset_list("2m")) == 50

    def test_unknown_band_empty(self):
        assert BandPlan.get_preset_list("nope") == []


class TestFrequencyScanner:
    def make_scanner(self):
        set_freq = []
        on_signal = []
        s = FrequencyScanner(
            set_freq_callback=set_freq.append,
            on_signal_callback=lambda f, r: on_signal.append((f, r)),
        )
        return s, set_freq, on_signal

    def test_start_and_stop(self):
        s, set_freq, _ = self.make_scanner()
        s.start_scan([144.390, 145.000], dwell_ms=10)
        assert s.is_scanning() is True
        time.sleep(0.2)
        s.stop_scan()
        assert s.is_scanning() is False
        assert 144.390 in set_freq
        assert 145.000 in set_freq

    def test_add_remove_frequency(self):
        s, _, _ = self.make_scanner()
        s.add_frequency(144.800)
        s.add_frequency(144.800)
        assert s.get_scan_list() == [144.800]
        s.remove_frequency(144.800)
        assert s.get_scan_list() == []

    def test_pause_resume(self):
        s, _, _ = self.make_scanner()
        assert s.is_paused is False
        s.pause()
        assert s.is_paused is True
        s.resume()
        assert s.is_paused is False

    def test_on_freq_change_callback(self):
        s, _, _ = self.make_scanner()
        changes = []
        s.on_freq_change = changes.append
        s.start_scan([144.390], dwell_ms=10)
        time.sleep(0.15)
        s.stop_scan()
        assert changes
        assert changes[0] == 144.390

    def test_no_frequencies_no_crash(self):
        s, set_freq, _ = self.make_scanner()
        s.start_scan([], dwell_ms=5)
        time.sleep(0.05)
        s.stop_scan()
        assert set_freq == []

    def test_on_signal_callback_and_hold(self):
        s, _, on_signal = self.make_scanner()
        rssi_values = iter([3.0, 3.0, 5.0, 5.0, 5.0])
        s._read_rssi = lambda: next(rssi_values)
        s._hold_seconds = 0.2
        s._on_signal = lambda f, r: on_signal.append((f, r))
        s.start_scan([144.390], dwell_ms=10, squelch_threshold=4.0)
        time.sleep(0.5)
        s.stop_scan()
        assert on_signal, "expected a signal to be detected"
