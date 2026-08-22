from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from kv4p_ht.app import MainWindow
from kv4p_ht.aprs import encode_ax25_ui
from kv4p_ht.channels import Channel


@pytest.fixture
def window(qtbot, monkeypatch):
    # Prevent worker threads and audio from starting during construction
    monkeypatch.setattr("kv4p_ht.radio.SerialWorker.run", lambda self: None)
    monkeypatch.setattr("kv4p_ht.radio.AudioWorker.run", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_settings", lambda self: None)
    monkeypatch.setattr(MainWindow, "_save_settings", lambda self: None)

    w = MainWindow()
    w.callsign = "KV4PHT"
    yield w
    for timer in (w._rssi_timer, w._settings_timer, w._beacon_timer):
        timer.stop()
    if w._serial:
        w._serial.stop()
    if w._audio:
        w._audio.stop()


def test_window_construction(window):
    assert window.connected is False
    assert window.callsign
    assert window._freq_edit.text() == "144.390"
    assert window._ptt_btn is not None


def test_set_frequency_updates_display(window):
    window._freq_edit.setText("146.520")
    window._offset_edit.setText("0.600")
    window._set_frequency()
    assert window.freq_rx == pytest.approx(146.520)
    assert window.freq_tx == pytest.approx(147.120)


def test_toggle_ptt_requires_connection(window):
    window.connected = False
    window._toggle_ptt()
    assert window.ptt is False


def test_on_wsjtx_status_packet(window, monkeypatch):
    logs = []
    monkeypatch.setattr(window, "log", logs.append)
    window._on_wsjtx_packet({"type": 1, "frequency": 145_000_000, "mode": "FT8"})
    assert any("FT8" in m and "145.000" in m for m in logs)


def test_on_wsjtx_decode_packet(window, monkeypatch):
    logs = []
    monkeypatch.setattr(window, "log", logs.append)
    window._on_wsjtx_packet({"type": 2, "message": "CQ K1ABC", "snr": -8})
    assert any("CQ K1ABC" in m and "-8" in m for m in logs)


def test_on_fldigi_packet(window, monkeypatch):
    logs = []
    monkeypatch.setattr(window, "log", logs.append)
    window._on_fldigi_packet({"type": "status", "data": {"freq": "144.390"}})
    assert any("freq=144.390" in m for m in logs)


def test_on_direwolf_packet(window):
    window._on_direwolf_packet({"type": "aprs", "line": "K1ABC>APZ010:>hi"})
    assert "[Dire Wolf]" in window._aprs_log.toPlainText()


def test_on_ax25_status_packet(window, monkeypatch):
    logs = []
    monkeypatch.setattr(window, "log", logs.append)
    frame = encode_ax25_ui("K1ABC", "APZ010", [], b">test status")
    window._on_ax25(0, frame + b"\x00\x00")
    assert any("test status" in m for m in logs)


def test_on_ax25_message_autoack(window, monkeypatch):
    acked = []
    monkeypatch.setattr(MainWindow, "_tx_afsk_waveform", lambda self, w: None)
    monkeypatch.setattr(window, "_send_aprs_rf", lambda *a, **k: acked.append(a))
    frame = encode_ax25_ui("K1ABC", "APZ010", [], b":KV4PHT:hello{AB")
    window._on_ax25(0, frame + b"\x00\x00")
    assert acked, "expected auto-ACK to be transmitted"


def test_on_smeter_sets_state(window):
    window._on_smeter(600)
    assert window.rssi == 600
    assert 1 <= window.s_meter <= 9


def test_send_group_queues_frame(window):
    import queue

    class FakeSerial:
        def __init__(self):
            self.cmd_queue = queue.SimpleQueue()
            self._port = True

        def stop(self):
            pass

    fake_serial = FakeSerial()
    window._serial = fake_serial
    window._send_group()
    frame = fake_serial.cmd_queue.get_nowait()
    assert frame[:4] == b"\xDE\xAD\xBE\xEF"
    assert frame[4] == 0x03


def test_igate_thread_callbacks_are_marshaled(window, qtbot, monkeypatch):
    """Regression: IGate runs on its own thread; its callbacks must not touch
    widgets directly (crash). They must be marshaled to the GUI thread."""
    import threading

    class FakeIGate(threading.Thread):
        def __init__(self, callsign, rf_tx_callback=None, aprs_is_callback=None, **kw):
            super().__init__(daemon=True)
            self._rf_cb = rf_tx_callback
            self._is_cb = aprs_is_callback

        def run(self):
            self._is_cb("Connected to APRS-IS as TEST")
            self._is_cb("N0CALL>APRS:!4903.50N07201.75W>test pos")
            self._rf_cb(encode_ax25_ui("N0CALL", "CQ", [], b">hello"), True)

        def stop(self):
            pass

    monkeypatch.setattr("kv4p_ht.app.IGate", FakeIGate)
    window._toggle_igate(True)
    try:
        qtbot.waitUntil(lambda: "[IS POS] N0CALL" in window._aprs_log.toPlainText(),
                        timeout=5000)
        assert "APRS-IS: Connected to APRS-IS as TEST" in window._log.toPlainText()
    finally:
        window._toggle_igate(False)


def test_kiss_frame_from_thread_is_marshaled(window, qtbot):
    """KISS TNC reader thread must not touch widgets directly."""
    import threading

    frame = encode_ax25_ui("N0CALL", "CQ", [], b">via kiss")

    def emit_from_thread():
        window._aprs_sig.kiss_frame.emit(frame)

    t = threading.Thread(target=emit_from_thread)
    t.start()
    t.join()
    qtbot.waitUntil(lambda: "[KISS]" in window._aprs_log.toPlainText(), timeout=5000)


def test_udp_packet_from_thread_is_marshaled(window, qtbot):
    """UDP listener thread must not touch widgets directly."""
    import threading

    def emit_from_thread():
        window._udp_sig.wsjt.emit({"type": 1, "frequency": 145_000_000, "mode": "FT8"})

    t = threading.Thread(target=emit_from_thread)
    t.start()
    t.join()
    qtbot.waitUntil(lambda: "FT8 on 145.000 MHz" in window._log.toPlainText(), timeout=5000)


def test_ch_select_populates_fields(window):
    """Regression: _ch_select/_ch_save referenced self._ch_name_edit but the
    widget was stored as ch_name_edit — selecting a channel crashed the app."""
    window._channel_bank.add(
        Channel(name="TESTCH", freq_rx=145.500, offset=0.0, mode="FM"))
    window._refresh_channel_list()
    window._ch_list.setCurrentIndex(len(window._channel_bank.channels) - 1)
    assert window._ch_name_edit.text() == "TESTCH"
    assert window._ch_freq_edit.text() == "145.5000"


def test_send_group_packs_ctcss_index_not_tenths(window):
    """Regression: combo data is tenths (885 = 88.5 Hz); packing it raw
    overflowed struct 'B' and aborted the app. Wire byte must be the
    tone-table index."""
    import queue
    import struct as _struct

    class FakeSerial:
        def __init__(self):
            self.cmd_queue = queue.SimpleQueue()
            self._port = True

        def stop(self):
            pass

    window._serial = FakeSerial()
    window.ctcss_tx = 885   # 88.5 Hz
    window.ctcss_rx = 2035  # 203.5 Hz
    window._send_group()
    frame = window._serial.cmd_queue.get_nowait()
    payload = frame[7:7 + 12]  # delim(4)+cmd(1)+len(2)
    bw, ftx, frx, ctx, sq, crx = _struct.unpack('<BffBBB', payload)
    from kv4p_ht.protocol import CTCSS_TONES, ctcss_to_index
    assert ctx == ctcss_to_index(885) == CTCSS_TONES.index(88.5) + 1
    assert crx == ctcss_to_index(2035)
