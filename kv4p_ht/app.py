"""
Main application window with radio and APRS panels.
Full-featured ham radio application: FM transceiver, spectrum analysis,
APRS, SSTV, Morse/CW, digital modes, AX.25 file transfer, scanner.
"""
from __future__ import annotations

import logging
import queue
import struct
import time

from PyQt6.QtCore import QObject, QSettings, Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .afsk import build_tx_waveform
from .aprs import (
    Digipeater,
    IGate,
    decode_ax25_frame,
    format_ack,
    format_beacon,
    format_message,
    parse_aprs,
)
from .ax25_file_transfer import (
    CMD_ACK, CMD_NAK, CMD_ABORT, PROTO_ID,
    FileTransferReceiver, FileTransferSender,
)
from .hamlib import RigCtlD
from .kiss import KissTnc
from .morse import MorseDecoder, MorseKeyer, PracticeGenerator
from .protocol import (
    DELIMITER, FrameSender, rssi_to_s_meter, CTCSS_TONES, ctcss_to_index,
)
from .radio import AudioWorker, SerialWorker
from .scanner import BandPlan, FrequencyScanner
from .spectrum import SpectrumAnalyzer, WaterfallBuffer
from .spectrum_widget import SpectrumControls, SpectrumDisplay, WaterfallDisplay
from .sstv import MODES as SSTV_MODES, SstvDecoder, SstvEncoder
from .udp_broadcast import UdpBroadcastRx
from .channels import Channel, ChannelBank
from .radio_face import RadioFaceplate
from .rf_sweep import RfSweeper, rssi_to_dbm

log = logging.getLogger(__name__)
debug_log = logging.getLogger(__name__ + ".debug")

# ── Debug frame hex dump helper ───────────────────────────────────

def _hex_dump(data: bytes, max_bytes: int = 64) -> str:
    """Pretty-print bytes as hex + ASCII."""
    if not data:
        return "(empty)"
    parts = []
    for i in range(0, min(len(data), max_bytes), 16):
        chunk = data[i:i + 16]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        parts.append(f"  {i:04x}: {hex_str:<48s} {ascii_str}")
    if len(data) > max_bytes:
        parts.append(f"  ... ({len(data)} bytes total)")
    return '\n'.join(parts)

CTCSS_TONES = list(CTCSS_TONES)

RADIO_MODES = ['FM']

SSTV_MODE_NAMES = list(SSTV_MODES.keys())


def _try_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except ValueError:
        return default


class _FtSignals(QObject):
    """Thread-safe signal bridge for file transfer callbacks."""
    log_msg = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    complete = pyqtSignal(bool, str)


class _AprsSignals(QObject):
    """Thread-safe signal bridge for callbacks from non-GUI threads (IGate, KISS TNC)."""
    is_line = pyqtSignal(str)
    rf_tx = pyqtSignal(bytes, bool)
    kiss_frame = pyqtSignal(bytes)


class _UdpSignals(QObject):
    """Thread-safe signal bridge for UDP broadcast listener callbacks."""
    log_line = pyqtSignal(str)
    wsjt = pyqtSignal(object)
    direwolf = pyqtSignal(object)
    fldigi = pyqtSignal(object)


class _ScanSignals(QObject):
    """Thread-safe signal bridge for scanner, RF-sweeper and rigctld callbacks."""
    scan_set_freq = pyqtSignal(float)
    scan_on_signal = pyqtSignal(float, float)
    sweep_set_freq = pyqtSignal(float)
    sweep_log = pyqtSignal(str)
    sweep_complete = pyqtSignal(object, object)
    sweep_progress = pyqtSignal(int, int)
    rig_freq = pyqtSignal(float)
    rig_ptt = pyqtSignal(bool)


class TxWorker(QThread):
    """Transmit a waveform (AFSK, SSTV, CW) on a background thread to avoid blocking the UI."""
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, waveform, cmd_queue, parent=None):
        super().__init__(parent)
        self._waveform = waveform
        self._cmd_queue = cmd_queue

    def run(self):
        import numpy as np
        try:
            import opuslib
        except ImportError:
            self.finished.emit(False, "opuslib not installed")
            return

        waveform = self._waveform
        frame_size = 1920
        enc = opuslib.Encoder(48000, 1, opuslib.APPLICATION_AUDIO)
        opus_frames = []
        offset = 0
        while offset < len(waveform):
            chunk = waveform[offset:offset + frame_size]
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            opus = enc.encode(pcm, frame_size)
            opus_frames.append(opus)
            offset += frame_size

        # PTT down
        ptt_down = DELIMITER + bytes([0x01]) + struct.pack('<H', 0)
        self._cmd_queue.put(ptt_down)
        time.sleep(0.3)

        frame_interval = frame_size / 48000  # 0.040s for 1920 @ 48kHz
        for opus in opus_frames:
            if self.isInterruptionRequested():
                ptt_up = DELIMITER + bytes([0x02]) + struct.pack('<H', 0)
                self._cmd_queue.put(ptt_up)
                self.finished.emit(False, "TX interrupted")
                return
            t0 = time.monotonic()
            frame = DELIMITER + bytes([0x07]) + struct.pack('<H', len(opus)) + opus
            self._cmd_queue.put(frame)
            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        total_playback_s = len(opus_frames) * frame_interval
        total_queue_s = len(opus_frames) * frame_interval  # now paced at real-time
        ptt_delay = max(0.3, 0.5)  # 0.5s tail after last frame
        time.sleep(ptt_delay)

        # PTT up
        ptt_up = DELIMITER + bytes([0x02]) + struct.pack('<H', 0)
        self._cmd_queue.put(ptt_up)
        self.finished.emit(True, f"TX complete ({len(opus_frames)} frames)")


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KV4P-Desktop")
        self.resize(1400, 900)
        self.setMinimumSize(900, 600)

        # ── State ────────────────────────────────────────────────
        self.callsign = "N0CALL"
        self.freq_tx = 144.390
        self.freq_rx = 144.390
        self.offset = 0.0
        self.squelch = 3
        self.bandwidth = 0
        self.ptt = False
        self.high_power = True
        self.rssi = 0
        self.s_meter = 1
        self.connected = False
        self.firmware_ver = 0
        self.module_type = ""
        self.radio_found = False
        self.radio_mode = 'FM'

        # ── CTCSS ────────────────────────────────────────────────
        self.ctcss_tx = 0
        self.ctcss_rx = 0

        # ── Audio levels ─────────────────────────────────────────
        self.mic_gain = 1.0
        self.speaker_volume = 1.0

        # ── APRS state ───────────────────────────────────────────
        self.aprs_beacon_on = False
        self.aprs_beacon_interval = 300
        self.aprs_igate_on = False
        self.aprs_digi_on = False
        self.aprs_lat = 0.0
        self.aprs_lon = 0.0
        self.aprs_symbol = '/-'
        self.aprs_path = "WIDE1-1"
        self._igate: IGate | None = None
        self._digipeater: Digipeater | None = None
        self._igate_stats = {'rf_to_is': 0, 'is_to_rf': 0, 'errors': 0}

        # ── Integration state ────────────────────────────────────
        self._rigctld: RigCtlD | None = None
        self._kiss_tnc: KissTnc | None = None
        self._udp_rx: UdpBroadcastRx | None = None
        self._scanner: FrequencyScanner | None = None
        self.scanning = False

        # ── Spectrum state ───────────────────────────────────────
        self._spectrum = SpectrumAnalyzer(
            sample_rate=48000, fft_size=2048, averaging=4,
            callback=self._on_spectrum_data
        )
        self._waterfall = WaterfallBuffer(max_rows=200)
        self._rf_sweeper: RfSweeper | None = None
        self._rf_sweep_mode = False

        # ── SSTV state ───────────────────────────────────────────
        self._sstv_encoder: SstvEncoder | None = None
        self._sstv_decoder: SstvDecoder | None = None
        self._sstv_image: np.ndarray | None = None

        # ── Morse state ──────────────────────────────────────────
        self._morse_keyer = MorseKeyer(wpm=20)
        self._morse_decoder = MorseDecoder(wpm=20)
        self._practice_gen = PracticeGenerator()

        # ── File transfer state ──────────────────────────────────
        self._ft_sig = _FtSignals()
        self._ft_sig.log_msg.connect(self._ft_log_ui)
        self._ft_sig.progress.connect(self._ft_progress_ui)
        self._ft_sig.complete.connect(self._ft_complete_ui)
        self._file_sender: FileTransferSender | None = None
        self._file_receiver: FileTransferReceiver | None = None

        # ── APRS / digital-modes thread bridges ──────────────────
        # IGate/KISS/UDP callbacks arrive on foreign threads; marshal to GUI thread.
        self._aprs_sig = _AprsSignals()
        self._aprs_sig.is_line.connect(self._on_aprs_is)
        self._aprs_sig.rf_tx.connect(self._rf_tx_callback)
        self._aprs_sig.kiss_frame.connect(self._on_kiss_ax25)

        self._udp_sig = _UdpSignals()
        self._udp_sig.log_line.connect(self._on_udp_log)
        self._udp_sig.wsjt.connect(self._on_wsjtx_packet)
        self._udp_sig.direwolf.connect(self._on_direwolf_packet)
        self._udp_sig.fldigi.connect(self._on_fldigi_packet)

        # Scanner / RF sweeper / rigctld callbacks arrive on foreign threads.
        self._scan_sig = _ScanSignals()
        self._scan_sig.scan_set_freq.connect(self._scan_set_freq)
        self._scan_sig.scan_on_signal.connect(self._scan_on_signal)
        self._scan_sig.sweep_set_freq.connect(self._rf_sweep_set_freq)
        self._scan_sig.sweep_log.connect(self.log)
        self._scan_sig.sweep_complete.connect(self._on_rf_sweep_complete)
        self._scan_sig.sweep_progress.connect(self._on_rf_sweep_progress)
        self._scan_sig.rig_freq.connect(self._on_rigctld_freq)
        self._scan_sig.rig_ptt.connect(self._on_rigctld_ptt)

        # ── TX worker ───────────────────────────────────────────
        self._tx_worker: TxWorker | None = None

        # ── Channel memory ──────────────────────────────────────
        self._channel_bank = ChannelBank()
        self._radio_face: RadioFaceplate | None = None

        # ── Workers ──────────────────────────────────────────────
        self._serial: SerialWorker | None = None
        self._audio: AudioWorker | None = None
        self._sender: FrameSender | None = None

        # ── Debug state ──────────────────────────────────────────
        self._tx_frame_count = 0
        self._rx_frame_count = 0
        self._tx_byte_count = 0
        self._rx_byte_count = 0
        self._debug_tx_log: list[str] = []
        self._debug_rx_log: list[str] = []
        self._debug_audio_level = 0.0
        self._morse_decoder_active = False
        self._spectrum_pcm_queue: queue.SimpleQueue | None = None

        # ── Load settings ────────────────────────────────────────
        self._load_settings()

        # ── GUI ──────────────────────────────────────────────────
        self._build_ui()
        self._build_menu()

        # ── Start workers ────────────────────────────────────────
        self._start_workers()

        # ── Timers ───────────────────────────────────────────────
        self._beacon_timer = QTimer(self)
        self._beacon_timer.timeout.connect(self._send_aprs_beacon)
        self._beacon_remaining = 0

        self._rssi_timer = QTimer(self)
        self._rssi_timer.timeout.connect(self._update_smeter_ui)
        self._rssi_timer.start(200)

        self._settings_timer = QTimer(self)
        self._settings_timer.timeout.connect(self._auto_save_settings)
        self._settings_timer.start(30000)

        self._spectrum_timer = QTimer(self)
        self._spectrum_timer.timeout.connect(self._update_spectrum_display)
        self._spectrum_timer.start(100)

        self._debug_timer = QTimer(self)
        self._debug_timer.timeout.connect(self._refresh_debug_counters)
        self._debug_timer.start(2000)

    # ── Settings persistence ─────────────────────────────────────

    def _load_settings(self):
        s = QSettings("kv4p", "kv4p-desktop")
        self.callsign = s.value("callsign", "N0CALL")
        self.freq_rx = float(s.value("freq_rx", "144.390"))
        self.offset = float(s.value("offset", "0.0"))
        self.freq_tx = self.freq_rx + self.offset
        self.squelch = int(s.value("squelch", "3"))
        self.bandwidth = int(s.value("bandwidth", "0"))
        self.high_power = s.value("high_power", "true") == "true"
        self.ctcss_tx = int(s.value("ctcss_tx", "0"))
        self.ctcss_rx = int(s.value("ctcss_rx", "0"))
        self.aprs_lat = float(s.value("aprs_lat", "0.0"))
        self.aprs_lon = float(s.value("aprs_lon", "0.0"))
        self.aprs_beacon_interval = int(s.value("aprs_beacon_interval", "300"))
        self.aprs_symbol = s.value("aprs_symbol", "/-")
        self.aprs_path = s.value("aprs_path", "WIDE1-1")
        self.mic_gain = float(s.value("mic_gain", "1.0"))
        self.speaker_volume = float(s.value("speaker_volume", "1.0"))
        self.radio_mode = s.value("radio_mode", "FM")

    def _save_settings(self):
        s = QSettings("kv4p", "kv4p-desktop")
        s.setValue("callsign", self.callsign)
        s.setValue("freq_rx", str(self.freq_rx))
        s.setValue("offset", str(self.offset))
        s.setValue("squelch", str(self.squelch))
        s.setValue("bandwidth", str(self.bandwidth))
        s.setValue("high_power", "true" if self.high_power else "false")
        s.setValue("ctcss_tx", str(self.ctcss_tx))
        s.setValue("ctcss_rx", str(self.ctcss_rx))
        s.setValue("aprs_lat", str(self.aprs_lat))
        s.setValue("aprs_lon", str(self.aprs_lon))
        s.setValue("aprs_beacon_interval", str(self.aprs_beacon_interval))
        s.setValue("aprs_symbol", self.aprs_symbol)
        s.setValue("aprs_path", self.aprs_path)
        s.setValue("mic_gain", str(self.mic_gain))
        s.setValue("speaker_volume", str(self.speaker_volume))
        s.setValue("radio_mode", self.radio_mode)

    def _auto_save_settings(self):
        if self.connected:
            self._save_settings()

    # ── Menu bar ─────────────────────────────────────────────────

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        save_act = QAction("&Save Settings", self)
        save_act.triggered.connect(self._save_settings)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        tx_menu = menu.addMenu("&TX")
        self._tx_mode_actions = {}
        for mode in RADIO_MODES:
            act = QAction(f"&{mode}", self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked, m=mode: self._set_radio_mode(m))
            tx_menu.addAction(act)
            self._tx_mode_actions[mode] = act
        if self.radio_mode in self._tx_mode_actions:
            self._tx_mode_actions[self.radio_mode].setChecked(True)

        window_menu = menu.addMenu("&Window")
        spectrum_act = QAction("&Spectrum Analyzer", self)
        spectrum_act.triggered.connect(lambda: self._main_tabs.setCurrentIndex(1))
        window_menu.addAction(spectrum_act)
        sstv_act = QAction("&SSTV", self)
        sstv_act.triggered.connect(lambda: self._main_tabs.setCurrentIndex(4))
        window_menu.addAction(sstv_act)

        help_menu = menu.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("About KV4P-Desktop")
        dlg.resize(450, 300)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            "<h2>KV4P-Desktop</h2>"
            "<p>Full-featured ham radio desktop application.</p>"
            "<p>FM transceiver with APRS, SSTV, Morse/CW</p>"
            "<p>Spectrum analysis, digital modes, file transfer</p>"
            "<p>License: GPL v3</p>"
            "<p><a href='https://kv4p.com'>https://kv4p.com</a></p>"
        ))
        if self.firmware_ver:
            layout.addWidget(QLabel(f"Firmware: v{self.firmware_ver}"))
        if self.module_type:
            layout.addWidget(QLabel(f"Module: {self.module_type}"))
        layout.addWidget(QLabel(f"Callsign: {self.callsign}"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    # ── UI Construction ───────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        top_splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Top area: main tabs ──
        self._main_tabs = QTabWidget()
        self._main_tabs.addTab(self._build_radio_tab(), "Radio")
        self._main_tabs.addTab(self._build_channels_tab(), "Channels")
        self._main_tabs.addTab(self._build_spectrum_tab(), "Spectrum")
        self._main_tabs.addTab(self._build_aprs_tab(), "APRS")
        self._main_tabs.addTab(self._build_digital_tab(), "Digital Modes")
        self._main_tabs.addTab(self._build_sstv_tab(), "SSTV")
        self._main_tabs.addTab(self._build_morse_tab(), "Morse / CW")
        self._main_tabs.addTab(self._build_file_transfer_tab(), "File Transfer")
        self._main_tabs.addTab(self._build_scanner_tab(), "Scanner")
        self._main_tabs.addTab(self._build_settings_tab(), "Settings")
        self._main_tabs.addTab(self._build_debug_tab(), "Debug")
        top_splitter.addWidget(self._main_tabs)

        # ── Bottom area: log tabs ──
        bottom_tabs = QTabWidget()
        bottom_tabs.setMaximumHeight(200)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Monospace", 9))
        self._log.setMaximumHeight(180)
        bottom_tabs.addTab(self._log, "Event Log")

        self._aprs_log = QTextEdit()
        self._aprs_log.setReadOnly(True)
        self._aprs_log.setFont(QFont("Monospace", 9))
        self._aprs_log.setMaximumHeight(180)
        bottom_tabs.addTab(self._aprs_log, "APRS Packets")

        self._igate_stats_te = QTextEdit()
        self._igate_stats_te.setReadOnly(True)
        self._igate_stats_te.setFont(QFont("Monospace", 9))
        self._igate_stats_te.setMaximumHeight(180)
        bottom_tabs.addTab(self._igate_stats_te, "iGate Stats")

        top_splitter.addWidget(bottom_tabs)
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 1)

        layout.addWidget(top_splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Disconnected")
        self._status.addPermanentWidget(self._status_label)

    # ── Radio tab ────────────────────────────────────────────────

    def _build_radio_tab(self) -> QWidget:
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(4)

        # Toggle bar: switch between desktop and radio-face mode
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        self._face_toggle = QPushButton("Radio Face")
        self._face_toggle.setCheckable(True)
        self._face_toggle.setFixedWidth(100)
        self._face_toggle.setStyleSheet(
            "QPushButton { border: 1px solid #555; border-radius: 3px; padding: 3px; }"
            "QPushButton:checked { background-color: #2d7d2d; color: white; }"
        )
        self._face_toggle.clicked.connect(self._toggle_radio_face)
        toggle_row.addWidget(self._face_toggle)
        toggle_row.addStretch()
        main_layout.addLayout(toggle_row)

        # Desktop mode: split layout
        self._radio_desktop = QWidget()
        desktop_layout = QHBoxLayout(self._radio_desktop)
        desktop_layout.setSpacing(8)
        desktop_layout.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(450)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(4)
        left_layout.addWidget(self._radio_control_group())
        left_layout.addWidget(self._audio_group())
        left_layout.addStretch()
        desktop_layout.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._smeter_group())
        right_layout.addWidget(self._mode_display_group())
        right_layout.addStretch()
        desktop_layout.addWidget(right)

        main_layout.addWidget(self._radio_desktop, stretch=1)

        # Radio face mode: faceplate widget (hidden by default)
        self._radio_face_container = QWidget()
        face_layout = QVBoxLayout(self._radio_face_container)
        face_layout.setContentsMargins(0, 0, 0, 0)
        self._radio_face = RadioFaceplate()
        self._radio_face.ptt_clicked.connect(self._toggle_ptt)
        self._radio_face.mode_changed.connect(self._set_radio_mode)
        self._radio_face.channel_changed.connect(self._on_face_channel)
        self._radio_face.channel_added.connect(self._on_face_channel_add)
        self._radio_face.squelch_changed.connect(self._set_squelch)
        face_layout.addWidget(self._radio_face)
        self._radio_face_container.setVisible(False)
        main_layout.addWidget(self._radio_face_container, stretch=1)

        return tab

    def _toggle_radio_face(self, on: bool):
        self._radio_desktop.setVisible(not on)
        self._radio_face_container.setVisible(on)
        if on:
            self._sync_faceplate()

    def _sync_faceplate(self):
        if not self._radio_face:
            return
        self._radio_face.set_frequency(self.freq_rx, "RX")
        self._radio_face.set_mode(self.radio_mode)
        self._radio_face.set_squelch(self.squelch)
        channels = []
        for i, ch in enumerate(self._channel_bank.channels):
            channels.append({'name': ch.name, 'freq_rx': ch.freq_rx, 'index': i})
        self._radio_face.set_channels(channels)

    def _on_face_channel(self, idx: int):
        if 0 <= idx < len(self._channel_bank.channels):
            ch = self._channel_bank.channels[idx]
            self.freq_rx = ch.freq_rx
            self.offset = ch.offset
            self.freq_tx = ch.freq_rx + ch.offset
            self.radio_mode = ch.mode
            self._freq_edit.setText(f"{ch.freq_rx:.3f}")
            self._freq_display.setText(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
            self._radio_face.set_frequency(self.freq_rx, ch.name)
            self._radio_face.set_mode(ch.mode)
            self._set_radio_mode(ch.mode)
            self._send_group()

    def _on_face_channel_add(self):
        ch = Channel(name=f"CH-{len(self._channel_bank.channels)+1}",
                     freq_rx=self.freq_rx, offset=self.offset,
                     mode=self.radio_mode, ctcss_tx=self.ctcss_tx,
                     ctcss_rx=self.ctcss_rx, squelch=self.squelch)
        self._channel_bank.add(ch)
        self._sync_faceplate()
        self._refresh_channel_list()

    # ── Channels tab ────────────────────────────────────────────

    def _build_channels_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(8)

        # Left: channel list
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setSpacing(4)

        toolbar = QHBoxLayout()
        self._ch_add_btn = QPushButton("+ Add")
        self._ch_add_btn.clicked.connect(self._ch_add)
        toolbar.addWidget(self._ch_add_btn)
        self._ch_del_btn = QPushButton("- Remove")
        self._ch_del_btn.clicked.connect(self._ch_remove)
        toolbar.addWidget(self._ch_del_btn)
        toolbar.addStretch()
        self._ch_import_btn = QPushButton("Import CSV")
        self._ch_import_btn.clicked.connect(self._ch_import)
        toolbar.addWidget(self._ch_import_btn)
        self._ch_export_btn = QPushButton("Export CSV")
        self._ch_export_btn.clicked.connect(self._ch_export)
        toolbar.addWidget(self._ch_export_btn)
        left_l.addLayout(toolbar)

        self._ch_list = QComboBox()
        self._ch_list.setMinimumHeight(300)
        self._ch_list.setStyleSheet(
            "QComboBox { font-size: 13px; }"
            "QComboBox QAbstractItemView { min-height: 280px; }"
        )
        self._ch_list.currentIndexChanged.connect(self._ch_select)
        left_l.addWidget(self._ch_list, stretch=1)

        layout.addWidget(left, stretch=1)

        # Right: edit selected channel
        right = QGroupBox("Channel Details")
        right_l = QVBoxLayout(right)

        for label_text, attr, widget_type, extra in [
            ("Name:", "_ch_name_edit", "line", {}),
            ("Freq RX (MHz):", "_ch_freq_edit", "line", {}),
            ("Offset (MHz):", "_ch_offset_edit", "line", {}),
            ("Mode:", "_ch_mode_combo", "combo", {"items": ["FM"]}),
            ("CTCSS TX:", "_ch_ctcss_tx_combo", "ctcss", {}),
            ("CTCSS RX:", "_ch_ctcss_rx_combo", "ctcss", {}),
            ("Notes:", "_ch_notes_edit", "line", {}),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(90)
            row.addWidget(lbl)
            if widget_type == "line":
                w = QLineEdit()
                setattr(self, attr, w)
                row.addWidget(w)
            elif widget_type == "combo":
                w = QComboBox()
                w.addItems(extra["items"])
                setattr(self, attr, w)
                row.addWidget(w)
            elif widget_type == "ctcss":
                w = QComboBox()
                w.addItem("None", 0)
                for t in CTCSS_TONES:
                    w.addItem(f"{t:.1f} Hz", int(t * 10))
                setattr(self, attr, w)
                row.addWidget(w)
            right_l.addLayout(row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self._ch_save)
        btn_row.addWidget(save_btn)
        tune_btn = QPushButton("Tune to Channel")
        tune_btn.clicked.connect(self._ch_tune)
        btn_row.addWidget(tune_btn)
        right_l.addLayout(btn_row)
        right_l.addStretch()

        layout.addWidget(right, stretch=1)

        self._refresh_channel_list()
        return tab

    def _refresh_channel_list(self):
        self._ch_list.clear()
        for ch in self._channel_bank.channels:
            self._ch_list.addItem(
                f"{ch.name}  {ch.freq_rx:.3f} MHz  {ch.mode}", len(self._ch_list)
            )
        self._sync_faceplate()

    def _ch_select(self, idx: int):
        if idx < 0 or idx >= len(self._channel_bank.channels):
            return
        ch = self._channel_bank.channels[idx]
        self._ch_name_edit.setText(ch.name)
        self._ch_freq_edit.setText(f"{ch.freq_rx:.4f}")
        self._ch_offset_edit.setText(f"{ch.offset:.4f}")
        self._ch_mode_combo.setCurrentText(ch.mode)
        self._ch_ctcss_tx_combo.setCurrentIndex(
            next((i for i in range(self._ch_ctcss_tx_combo.count())
                  if self._ch_ctcss_tx_combo.itemData(i) == ch.ctcss_tx), 0))
        self._ch_ctcss_rx_combo.setCurrentIndex(
            next((i for i in range(self._ch_ctcss_rx_combo.count())
                  if self._ch_ctcss_rx_combo.itemData(i) == ch.ctcss_rx), 0))
        self._ch_notes_edit.setText(ch.notes)

    def _ch_add(self):
        ch = Channel(
            name=f"CH-{len(self._channel_bank.channels)+1}",
            freq_rx=self.freq_rx, offset=self.offset,
            mode=self.radio_mode, ctcss_tx=self.ctcss_tx,
            ctcss_rx=self.ctcss_rx, squelch=self.squelch,
        )
        self._channel_bank.add(ch)
        self._refresh_channel_list()
        self._ch_list.setCurrentIndex(len(self._channel_bank.channels) - 1)

    def _ch_remove(self):
        idx = self._ch_list.currentIndex()
        if 0 <= idx < len(self._channel_bank.channels):
            name = self._channel_bank.channels[idx].name
            self._channel_bank.remove(idx)
            self._refresh_channel_list()
            self.log(f"Channel '{name}' removed")

    def _ch_save(self):
        idx = self._ch_list.currentIndex()
        if idx < 0 or idx >= len(self._channel_bank.channels):
            return
        ch = Channel(
            name=self._ch_name_edit.text().strip() or f"CH-{idx+1}",
            freq_rx=float(self._ch_freq_edit.text() or "144.390"),
            offset=float(self._ch_offset_edit.text() or "0.0"),
            mode=self._ch_mode_combo.currentText(),
            ctcss_tx=self._ch_ctcss_tx_combo.currentData() or 0,
            ctcss_rx=self._ch_ctcss_rx_combo.currentData() or 0,
            notes=self._ch_notes_edit.text().strip(),
        )
        self._channel_bank.update(idx, ch)
        self._refresh_channel_list()
        self._ch_list.setCurrentIndex(idx)
        self.log(f"Channel '{ch.name}' saved")

    def _ch_tune(self):
        idx = self._ch_list.currentIndex()
        if 0 <= idx < len(self._channel_bank.channels):
            ch = self._channel_bank.channels[idx]
            self.freq_rx = ch.freq_rx
            self.offset = ch.offset
            self.freq_tx = ch.freq_rx + ch.offset
            self._freq_edit.setText(f"{ch.freq_rx:.3f}")
            self._freq_display.setText(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
            self._set_radio_mode(ch.mode)
            self._send_group()
            self.log(f"Tuned to {ch.name}: {ch.freq_rx:.3f} MHz {ch.mode}")

    def _ch_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Channels", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            count = self._channel_bank.import_csv(path)
            self._refresh_channel_list()
            self.log(f"Imported {count} channels from {path}")

    def _ch_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Channels", "channels.csv", "CSV Files (*.csv)"
        )
        if path:
            self._channel_bank.export_csv(path)
            self.log(f"Exported {len(self._channel_bank.channels)} channels to {path}")

    def _radio_control_group(self) -> QGroupBox:
        g = QGroupBox("Radio Control")
        v = QVBoxLayout(g)

        # Callsign
        h = QHBoxLayout()
        h.addWidget(QLabel("Callsign:"))
        self._callsign_edit = QLineEdit(self.callsign)
        self._callsign_edit.textChanged.connect(lambda t: setattr(self, 'callsign', t.upper()))
        h.addWidget(self._callsign_edit)
        v.addLayout(h)

        # Frequency
        h = QHBoxLayout()
        h.addWidget(QLabel("Freq (MHz):"))
        self._freq_edit = QLineEdit(f"{self.freq_rx:.3f}")
        h.addWidget(self._freq_edit)
        self._set_freq_btn = QPushButton("Set")
        self._set_freq_btn.clicked.connect(self._set_frequency)
        h.addWidget(self._set_freq_btn)
        v.addLayout(h)

        # Offset
        h = QHBoxLayout()
        h.addWidget(QLabel("Offset (MHz):"))
        self._offset_edit = QLineEdit(f"{self.offset:.3f}")
        h.addWidget(self._offset_edit)
        v.addLayout(h)

        # Frequency display
        self._freq_display = QLabel(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
        font = self._freq_display.font()
        font.setPointSize(14)
        font.setBold(True)
        self._freq_display.setFont(font)
        v.addWidget(self._freq_display)

        # PTT button
        self._ptt_btn = QPushButton("PTT (click to TX)")
        self._ptt_btn.setMinimumHeight(50)
        self._ptt_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; font-size: 16px; }"
            "QPushButton:hover { background-color: #3a9e3a; }"
        )
        self._ptt_btn.clicked.connect(self._toggle_ptt)
        v.addWidget(self._ptt_btn)

        # Squelch
        h = QHBoxLayout()
        h.addWidget(QLabel("Squelch:"))
        self._squelch_slider = QSlider(Qt.Orientation.Horizontal)
        self._squelch_slider.setRange(0, 8)
        self._squelch_slider.setValue(self.squelch)
        self._squelch_slider.valueChanged.connect(self._set_squelch)
        h.addWidget(self._squelch_slider)
        self._squelch_label = QLabel(str(self.squelch))
        h.addWidget(self._squelch_label)
        v.addLayout(h)

        # Bandwidth
        h = QHBoxLayout()
        h.addWidget(QLabel("BW:"))
        self._bw_combo = QComboBox()
        self._bw_combo.addItems(["12.5 kHz", "25 kHz"])
        self._bw_combo.setCurrentIndex(self.bandwidth)
        self._bw_combo.currentIndexChanged.connect(self._set_bw)
        h.addWidget(self._bw_combo)
        v.addLayout(h)

        # CTCSS
        h = QHBoxLayout()
        h.addWidget(QLabel("CTCSS TX:"))
        self._ctcss_tx_combo = QComboBox()
        self._ctcss_tx_combo.addItem("None", 0)
        for t in CTCSS_TONES:
            self._ctcss_tx_combo.addItem(f"{t:.1f} Hz", int(t * 10))
        self._ctcss_tx_combo.setCurrentIndex(0)
        self._ctcss_tx_combo.currentIndexChanged.connect(self._set_ctcss)
        h.addWidget(self._ctcss_tx_combo)
        h.addWidget(QLabel("RX:"))
        self._ctcss_rx_combo = QComboBox()
        self._ctcss_rx_combo.addItem("None", 0)
        for t in CTCSS_TONES:
            self._ctcss_rx_combo.addItem(f"{t:.1f} Hz", int(t * 10))
        self._ctcss_rx_combo.setCurrentIndex(0)
        self._ctcss_rx_combo.currentIndexChanged.connect(self._set_ctcss)
        h.addWidget(self._ctcss_rx_combo)
        v.addLayout(h)

        # Power
        h = QHBoxLayout()
        h.addWidget(QLabel("Power:"))
        self._hp_btn = QPushButton("High (1W)")
        self._hp_btn.setCheckable(True)
        self._hp_btn.setChecked(self.high_power)
        self._hp_btn.clicked[bool].connect(self._set_power)
        h.addWidget(self._hp_btn)
        v.addLayout(h)

        return g

    def _audio_group(self) -> QGroupBox:
        g = QGroupBox("Audio")
        v = QVBoxLayout(g)

        h = QHBoxLayout()
        h.addWidget(QLabel("Mic Gain:"))
        self._mic_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._mic_gain_slider.setRange(0, 200)
        self._mic_gain_slider.setValue(int(self.mic_gain * 100))
        self._mic_gain_slider.valueChanged.connect(self._set_mic_gain)
        h.addWidget(self._mic_gain_slider)
        self._mic_gain_label = QLabel(f"{self.mic_gain:.1f}x")
        h.addWidget(self._mic_gain_label)
        v.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Volume:"))
        self._spk_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._spk_vol_slider.setRange(0, 200)
        self._spk_vol_slider.setValue(int(self.speaker_volume * 100))
        self._spk_vol_slider.valueChanged.connect(self._set_speaker_volume)
        h.addWidget(self._spk_vol_slider)
        self._spk_vol_label = QLabel(f"{self.speaker_volume:.1f}x")
        h.addWidget(self._spk_vol_label)
        v.addLayout(h)

        return g

    def _smeter_group(self) -> QGroupBox:
        g = QGroupBox("Signal Meter")
        v = QVBoxLayout(g)

        self._smeter_label = QLabel("S1")
        font = self._smeter_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self._smeter_label.setFont(font)
        v.addWidget(self._smeter_label)

        self._smeter_bar = QFrame()
        self._smeter_bar.setFrameShape(QFrame.Shape.Box)
        self._smeter_bar.setMinimumHeight(24)
        self._smeter_bar.setStyleSheet("background-color: #1a1a1a;")
        self._smeter_fill = QWidget(self._smeter_bar)
        self._smeter_fill.setGeometry(0, 0, 0, 24)
        self._smeter_fill.setStyleSheet("background-color: green;")
        v.addWidget(self._smeter_bar)

        self._dbm_label = QLabel("dBm: --")
        v.addWidget(self._dbm_label)

        return g

    def _mode_display_group(self) -> QGroupBox:
        g = QGroupBox("Mode")
        v = QVBoxLayout(g)

        h = QHBoxLayout()
        self._mode_buttons = {}
        for mode in RADIO_MODES:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda checked, m=mode: self._set_radio_mode(m))
            h.addWidget(btn)
            self._mode_buttons[mode] = btn
        v.addLayout(h)

        if self.radio_mode in self._mode_buttons:
            self._mode_buttons[self.radio_mode].setChecked(True)

        return g

    # ── Spectrum tab ─────────────────────────────────────────────

    def _build_spectrum_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        self._spectrum_display = SpectrumDisplay()
        self._spectrum_display.setMinimumHeight(200)
        self._spectrum_display.spectrum_clicked.connect(self._on_spectrum_click)
        layout.addWidget(self._spectrum_display, stretch=2)

        self._waterfall_display = WaterfallDisplay()
        self._waterfall_display.setMinimumHeight(150)
        layout.addWidget(self._waterfall_display, stretch=2)

        self._spectrum_controls = SpectrumControls()
        self._spectrum_controls.settings_changed.connect(self._on_spectrum_settings_changed)
        self._spectrum_controls.reset_peaks_clicked.connect(self._on_reset_peaks)
        self._spectrum_controls.rf_sweep_toggled.connect(self._on_rf_sweep_toggled)
        self._spectrum_controls.rf_sweep_start.connect(self._on_rf_sweep_start)
        self._spectrum_controls.rf_sweep_stop.connect(self._on_rf_sweep_stop)
        layout.addWidget(self._spectrum_controls)

        info_layout = QHBoxLayout()
        self._spec_info_label = QLabel("Center: 144.390 MHz  Span: 48 kHz  RBW: 23 Hz")
        info_layout.addWidget(self._spec_info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        return tab

    # ── APRS tab ─────────────────────────────────────────────────

    def _build_aprs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left = QWidget()
        left.setMinimumWidth(350)
        left.setMaximumWidth(500)
        ll = QVBoxLayout(left)

        # ── Toggles ──
        g = QGroupBox("APRS Controls")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        self._igate_btn = QPushButton("iGate OFF")
        self._igate_btn.setCheckable(True)
        self._igate_btn.clicked[bool].connect(self._toggle_igate)
        h.addWidget(self._igate_btn)
        self._digi_btn = QPushButton("Digi OFF")
        self._digi_btn.setCheckable(True)
        self._digi_btn.clicked[bool].connect(self._toggle_digi)
        h.addWidget(self._digi_btn)
        self._beacon_btn = QPushButton("Beacon OFF")
        self._beacon_btn.setCheckable(True)
        self._beacon_btn.clicked[bool].connect(self._toggle_beacon)
        h.addWidget(self._beacon_btn)
        gv.addLayout(h)

        # ── Beacon config ──
        h = QHBoxLayout()
        h.addWidget(QLabel("Interval:"))
        self._beacon_spin = QSpinBox()
        self._beacon_spin.setRange(60, 3600)
        self._beacon_spin.setValue(self.aprs_beacon_interval)
        self._beacon_spin.setSuffix(" s")
        self._beacon_spin.valueChanged.connect(lambda v: setattr(self, 'aprs_beacon_interval', v))
        h.addWidget(self._beacon_spin)
        gv.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Lat:"))
        self._lat_edit = QLineEdit(f"{self.aprs_lat:.4f}")
        self._lat_edit.setFixedWidth(80)
        self._lat_edit.textChanged.connect(self._on_latlon_changed)
        h.addWidget(self._lat_edit)
        h.addWidget(QLabel("Lon:"))
        self._lon_edit = QLineEdit(f"{self.aprs_lon:.4f}")
        self._lon_edit.setFixedWidth(80)
        self._lon_edit.textChanged.connect(self._on_latlon_changed)
        h.addWidget(self._lon_edit)
        gv.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Path:"))
        self._aprs_path_edit = QLineEdit(self.aprs_path)
        self._aprs_path_edit.textChanged.connect(lambda t: setattr(self, 'aprs_path', t))
        h.addWidget(self._aprs_path_edit)
        gv.addLayout(h)

        # ── IGate config ──
        self._igate_filter = QLineEdit()
        self._igate_filter.setPlaceholderText("APRS-IS filter (e.g. m/200)")
        gv.addWidget(self._igate_filter)

        h = QHBoxLayout()
        self._igate_tx = QCheckBox("IS->RF TX")
        self._igate_tx.setChecked(True)
        h.addWidget(self._igate_tx)
        self._igate_status = QLineEdit()
        self._igate_status.setPlaceholderText("iGate status")
        h.addWidget(self._igate_status)
        gv.addLayout(h)

        ll.addWidget(g)

        # ── Message compose ──
        g = QGroupBox("Send Message")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        h.addWidget(QLabel("To:"))
        self._aprs_to = QLineEdit()
        self._aprs_to.setPlaceholderText("CALL")
        h.addWidget(self._aprs_to)
        gv.addLayout(h)
        h = QHBoxLayout()
        self._aprs_msg = QLineEdit()
        self._aprs_msg.setPlaceholderText("Message text, Enter to send")
        self._aprs_msg.returnPressed.connect(self._aprs_send_message)
        h.addWidget(self._aprs_msg)
        self._aprs_send_btn = QPushButton("Send")
        self._aprs_send_btn.clicked.connect(self._aprs_send_message)
        h.addWidget(self._aprs_send_btn)
        gv.addLayout(h)
        ll.addWidget(g)

        # ── Test button ──
        h = QHBoxLayout()
        self._test_decode_btn = QPushButton("Test Decode")
        self._test_decode_btn.clicked.connect(self._aprs_test_decode)
        h.addWidget(self._test_decode_btn)
        self._test_status = QLabel("")
        h.addWidget(self._test_status)
        ll.addLayout(h)
        ll.addStretch()
        layout.addWidget(left)

        # Right: APRS log
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("APRS Packets"))
        self._aprs_detail_log = QTextEdit()
        self._aprs_detail_log.setReadOnly(True)
        self._aprs_detail_log.setFont(QFont("Monospace", 9))
        rl.addWidget(self._aprs_detail_log)
        layout.addWidget(right)

        return tab

    # ── Digital Modes tab ────────────────────────────────────────

    def _build_digital_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g = QGroupBox("External Digital Mode Software")
        gv = QVBoxLayout(g)

        # Hamlib rigctld
        hl = QHBoxLayout()
        hl.addWidget(QLabel("RigCtlD (Hamlib):"))
        self._rigctl_host = QLineEdit("localhost")
        self._rigctl_host.setFixedWidth(100)
        hl.addWidget(self._rigctl_host)
        self._rigctl_port = QLineEdit("4532")
        self._rigctl_port.setFixedWidth(60)
        hl.addWidget(self._rigctl_port)
        self._rigctl_btn = QPushButton("Connect")
        self._rigctl_btn.setCheckable(True)
        self._rigctl_btn.clicked[bool].connect(self._toggle_rigctld)
        hl.addWidget(self._rigctl_btn)
        gv.addLayout(hl)

        # KISS TNC
        hl = QHBoxLayout()
        hl.addWidget(QLabel("KISS TNC (Dire Wolf):"))
        self._kiss_host = QLineEdit("localhost")
        self._kiss_host.setFixedWidth(100)
        hl.addWidget(self._kiss_host)
        self._kiss_port = QLineEdit("8001")
        self._kiss_port.setFixedWidth(60)
        hl.addWidget(self._kiss_port)
        self._kiss_btn = QPushButton("Connect")
        self._kiss_btn.setCheckable(True)
        self._kiss_btn.clicked[bool].connect(self._toggle_kiss)
        hl.addWidget(self._kiss_btn)
        gv.addLayout(hl)

        # UDP Broadcast
        hl = QHBoxLayout()
        self._udp_cb = QCheckBox("UDP Broadcast Listener (WSJT-X / FLDigi / Dire Wolf)")
        self._udp_cb.toggled.connect(self._toggle_udp)
        hl.addWidget(self._udp_cb)
        gv.addLayout(hl)

        # Status
        self._digital_status = QLabel("Not connected to any digital mode software")
        gv.addWidget(self._digital_status)

        layout.addWidget(g)

        # Supported modes info
        info = QGroupBox("Supported Digital Modes via Integration")
        iv = QVBoxLayout(info)
        modes_text = (
            "<b>WSJT-X / JTDX:</b> FT8, FT4, JT65, JT9, WSPR, MSK144<br>"
            "<b>FLDigi:</b> RTTY, PSK31/64/125, Olivia, Contestia, CW, MT63, Thor, DominoEX, etc.<br>"
            "<b>JS8Call:</b> JS8 (QRP digital messaging)<br>"
            "<b>Dire Wolf:</b> APRS, packet radio, AX.25<br>"
            "<b>Pat (Winlink):</b> Email over radio<br>"
            "<b>Hamlib:</b> CAT control for any supported radio<br><br>"
            "These applications connect via UDP broadcast, KISS TNC, or rigctld TCP."
        )
        iv.addWidget(QLabel(modes_text))
        layout.addWidget(info)

        layout.addStretch()
        return tab

    # ── SSTV tab ─────────────────────────────────────────────────

    def _build_sstv_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left: controls
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(400)
        ll = QVBoxLayout(left)

        g = QGroupBox("SSTV Mode")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        h.addWidget(QLabel("Mode:"))
        self._sstv_mode_combo = QComboBox()
        self._sstv_mode_combo.addItems(SSTV_MODE_NAMES)
        h.addWidget(self._sstv_mode_combo)
        gv.addLayout(h)
        ll.addWidget(g)

        # Encode
        g = QGroupBox("Encode (TX)")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        self._sstv_load_btn = QPushButton("Load Image")
        self._sstv_load_btn.clicked.connect(self._sstv_load_image)
        h.addWidget(self._sstv_load_btn)
        self._sstv_image_label = QLabel("No image loaded")
        h.addWidget(self._sstv_image_label)
        gv.addLayout(h)
        self._sstv_preview = QLabel()
        self._sstv_preview.setMinimumHeight(120)
        self._sstv_preview.setMaximumHeight(200)
        self._sstv_preview.setStyleSheet("background-color: #1a1a1a;")
        self._sstv_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gv.addWidget(self._sstv_preview)
        self._sstv_tx_btn = QPushButton("Transmit SSTV")
        self._sstv_tx_btn.setMinimumHeight(40)
        self._sstv_tx_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; }"
        )
        self._sstv_tx_btn.clicked.connect(self._sstv_transmit)
        gv.addWidget(self._sstv_tx_btn)
        ll.addWidget(g)

        # Decode
        g = QGroupBox("Decode (RX)")
        gv = QVBoxLayout(g)
        self._sstv_decode_btn = QPushButton("Start SSTV Decode")
        self._sstv_decode_btn.setCheckable(True)
        self._sstv_decode_btn.clicked[bool].connect(self._sstv_toggle_decode)
        gv.addWidget(self._sstv_decode_btn)
        self._sstv_decode_status = QLabel("Idle")
        gv.addWidget(self._sstv_decode_status)
        self._sstv_rx_preview = QLabel()
        self._sstv_rx_preview.setMinimumHeight(120)
        self._sstv_rx_preview.setMaximumHeight(200)
        self._sstv_rx_preview.setStyleSheet("background-color: #1a1a1a;")
        self._sstv_rx_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gv.addWidget(self._sstv_rx_preview)
        ll.addWidget(g)
        ll.addStretch()

        layout.addWidget(left)

        # Right: decoded images gallery
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("SSTV Images"))
        self._sstv_gallery = QTextEdit()
        self._sstv_gallery.setReadOnly(True)
        self._sstv_gallery.setFont(QFont("Monospace", 9))
        rl.addWidget(self._sstv_gallery)
        layout.addWidget(right)

        return tab

    # ── Morse / CW tab ───────────────────────────────────────────

    def _build_morse_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left: keyer controls
        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(450)
        ll = QVBoxLayout(left)

        g = QGroupBox("CW Keyer")
        gv = QVBoxLayout(g)

        h = QHBoxLayout()
        h.addWidget(QLabel("WPM:"))
        self._cw_wpm_spin = QSpinBox()
        self._cw_wpm_spin.setRange(5, 60)
        self._cw_wpm_spin.setValue(20)
        self._cw_wpm_spin.valueChanged.connect(self._morse_set_wpm)
        h.addWidget(self._cw_wpm_spin)

        h.addWidget(QLabel("Tone:"))
        self._cw_tone_spin = QSpinBox()
        self._cw_tone_spin.setRange(400, 1000)
        self._cw_tone_spin.setValue(700)
        self._cw_tone_spin.setSuffix(" Hz")
        self._cw_tone_spin.valueChanged.connect(self._morse_set_tone)
        h.addWidget(self._cw_tone_spin)
        gv.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Mode:"))
        self._cw_mode_combo = QComboBox()
        self._cw_mode_combo.addItems(["Iambic A", "Iambic B", "Straight Key"])
        self._cw_mode_combo.currentIndexChanged.connect(self._morse_set_mode)
        h.addWidget(self._cw_mode_combo)
        gv.addLayout(h)

        self._cw_tx_btn = QPushButton("Send CW Tone")
        self._cw_tx_btn.setMinimumHeight(36)
        self._cw_tx_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; }"
        )
        self._cw_tx_btn.clicked.connect(self._morse_send_tone)
        gv.addWidget(self._cw_tx_btn)

        ll.addWidget(g)

        # CW input
        g = QGroupBox("CW Input")
        gv = QVBoxLayout(g)
        self._cw_input = QLineEdit()
        self._cw_input.setPlaceholderText("Type text to send as Morse...")
        self._cw_input.returnPressed.connect(self._morse_send_text)
        gv.addWidget(self._cw_input)
        h = QHBoxLayout()
        self._cw_send_btn = QPushButton("Send")
        self._cw_send_btn.clicked.connect(self._morse_send_text)
        h.addWidget(self._cw_send_btn)
        self._cw_clear_btn = QPushButton("Clear")
        self._cw_clear_btn.clicked.connect(lambda: self._cw_input.clear())
        h.addWidget(self._cw_clear_btn)
        gv.addLayout(h)
        ll.addWidget(g)

        # Practice
        g = QGroupBox("Practice Mode")
        gv = QVBoxLayout(g)
        self._cw_practice_btn = QPushButton("Generate Practice QSO")
        self._cw_practice_btn.clicked.connect(self._morse_generate_practice)
        gv.addWidget(self._cw_practice_btn)
        self._cw_practice_text = QTextEdit()
        self._cw_practice_text.setMaximumHeight(60)
        self._cw_practice_text.setReadOnly(True)
        gv.addWidget(self._cw_practice_text)
        self._cw_play_practice_btn = QPushButton("Play Practice")
        self._cw_play_practice_btn.clicked.connect(self._morse_play_practice)
        gv.addWidget(self._cw_play_practice_btn)
        ll.addWidget(g)

        # Reference
        g = QGroupBox("Morse Reference")
        gv = QVBoxLayout(g)
        ref_text = QTextEdit()
        ref_text.setReadOnly(True)
        ref_text.setMaximumHeight(120)
        ref_text.setFont(QFont("Monospace", 9))
        ref_lines = []
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            from .morse import CHAR_TO_MORSE
            ref_lines.append(f"{ch}: {CHAR_TO_MORSE.get(ch, '')}")
        ref_text.setPlainText("  ".join(ref_lines))
        gv.addWidget(ref_text)
        ll.addWidget(g)
        ll.addStretch()

        layout.addWidget(left)

        # Right: decoder display
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("Morse Decoder"))

        self._cw_decode_display = QTextEdit()
        self._cw_decode_display.setReadOnly(True)
        self._cw_decode_display.setFont(QFont("Monospace", 14))
        self._cw_decode_display.setStyleSheet("color: #00ff00; background-color: #0a0a0a;")
        rl.addWidget(self._cw_decode_display)

        h = QHBoxLayout()
        self._cw_decoder_btn = QPushButton("Start Decoder")
        self._cw_decoder_btn.setCheckable(True)
        self._cw_decoder_btn.clicked[bool].connect(self._morse_toggle_decoder)
        h.addWidget(self._cw_decoder_btn)
        self._cw_decoder_status = QLabel("Stopped")
        h.addWidget(self._cw_decoder_status)
        rl.addLayout(h)

        self._cw_element_display = QLabel("Current: ")
        font = self._cw_element_display.font()
        font.setPointSize(12)
        self._cw_element_display.setFont(font)
        rl.addWidget(self._cw_element_display)

        layout.addWidget(right)

        return tab

    # ── File Transfer tab ────────────────────────────────────────

    def _build_file_transfer_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g = QGroupBox("AX.25 File Transfer")
        gv = QVBoxLayout(g)

        h = QHBoxLayout()
        h.addWidget(QLabel("Destination:"))
        self._ft_dest = QLineEdit()
        self._ft_dest.setPlaceholderText("Destination callsign")
        h.addWidget(self._ft_dest)
        h.addWidget(QLabel("Path:"))
        self._ft_path = QLineEdit("WIDE1-1")
        h.addWidget(self._ft_path)
        gv.addLayout(h)

        h = QHBoxLayout()
        self._ft_send_btn = QPushButton("Send File")
        self._ft_send_btn.setMinimumHeight(36)
        self._ft_send_btn.setStyleSheet(
            "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; }"
        )
        self._ft_send_btn.clicked.connect(self._ft_send_file)
        h.addWidget(self._ft_send_btn)

        self._ft_receive_btn = QPushButton("Start Receiver")
        self._ft_receive_btn.setCheckable(True)
        self._ft_receive_btn.setMinimumHeight(36)
        self._ft_receive_btn.clicked[bool].connect(self._ft_toggle_receiver)
        h.addWidget(self._ft_receive_btn)
        gv.addLayout(h)

        self._ft_progress = QProgressBar()
        self._ft_progress.setValue(0)
        gv.addWidget(self._ft_progress)

        self._ft_status = QLabel("Idle")
        gv.addWidget(self._ft_status)

        layout.addWidget(g)

        # Log
        layout.addWidget(QLabel("Transfer Log"))
        self._ft_log = QTextEdit()
        self._ft_log.setReadOnly(True)
        self._ft_log.setFont(QFont("Monospace", 9))
        layout.addWidget(self._ft_log)

        return tab

    # ── Scanner tab ──────────────────────────────────────────────

    def _build_scanner_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g = QGroupBox("Frequency Scanner")
        gv = QVBoxLayout(g)

        h = QHBoxLayout()
        h.addWidget(QLabel("Band:"))
        self._scan_band = QComboBox()
        self._scan_band.addItems(["2m", "70cm", "Custom"])
        h.addWidget(self._scan_band)
        h.addWidget(QLabel("Dwell:"))
        self._scan_dwell = QSpinBox()
        self._scan_dwell.setRange(100, 5000)
        self._scan_dwell.setValue(500)
        self._scan_dwell.setSuffix(" ms")
        h.addWidget(self._scan_dwell)
        gv.addLayout(h)

        h = QHBoxLayout()
        self._scan_btn = QPushButton("Start Scan")
        self._scan_btn.setMinimumHeight(36)
        self._scan_btn.clicked.connect(self._toggle_scan)
        h.addWidget(self._scan_btn)
        self._scan_pause_btn = QPushButton("Pause")
        self._scan_pause_btn.setEnabled(False)
        self._scan_pause_btn.clicked.connect(self._pause_scan)
        h.addWidget(self._scan_pause_btn)
        gv.addLayout(h)

        self._scan_status = QLabel("Idle")
        gv.addWidget(self._scan_status)

        layout.addWidget(g)
        layout.addStretch()
        return tab

    # ── Debug tab ─────────────────────────────────────────────────

    def _build_debug_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g = QGroupBox("Frame I/O Counters")
        gv = QVBoxLayout(g)
        self._debug_counters_label = QLabel("TX: 0 frames (0 bytes) | RX: 0 frames (0 bytes)")
        gv.addWidget(self._debug_counters_label)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_debug_counters)
        gv.addWidget(refresh_btn)
        layout.addWidget(g)

        g2 = QGroupBox("Last 50 TX Frames")
        self._debug_tx_te = QTextEdit()
        self._debug_tx_te.setReadOnly(True)
        self._debug_tx_te.setFont(QFont("Monospace", 9))
        gv2 = QVBoxLayout(g2)
        gv2.addWidget(self._debug_tx_te)
        layout.addWidget(g2)

        g3 = QGroupBox("Last 50 RX Frames")
        self._debug_rx_te = QTextEdit()
        self._debug_rx_te.setReadOnly(True)
        self._debug_rx_te.setFont(QFont("Monospace", 9))
        gv3 = QVBoxLayout(g3)
        gv3.addWidget(self._debug_rx_te)
        layout.addWidget(g3)

        g4 = QGroupBox("Audio Levels")
        gv4 = QVBoxLayout(g4)
        self._debug_audio_level_label = QLabel("RX Level: -- | Mic Level: --")
        gv4.addWidget(self._debug_audio_level_label)
        layout.addWidget(g4)

        g5 = QGroupBox("Hex Dump Log")
        self._debug_hex_te = QTextEdit()
        self._debug_hex_te.setReadOnly(True)
        self._debug_hex_te.setFont(QFont("Monospace", 9))
        self._debug_hex_te.setMaximumHeight(200)
        gv5 = QVBoxLayout(g5)
        gv5.addWidget(self._debug_hex_te)
        layout.addWidget(g5)

        layout.addStretch()
        return tab

    def _refresh_debug_counters(self):
        self._debug_counters_label.setText(
            f"TX: {self._tx_frame_count} frames ({self._tx_byte_count} bytes) | "
            f"RX: {self._rx_frame_count} frames ({self._rx_byte_count} bytes)"
        )
        self._debug_tx_te.setPlainText('\n'.join(reversed(self._debug_tx_log[-50:])))
        self._debug_rx_te.setPlainText('\n'.join(reversed(self._debug_rx_log[-50:])))

    # ── Settings tab ─────────────────────────────────────────────

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        g = QGroupBox("Audio Devices")
        gv = QVBoxLayout(g)
        self._audio_devices_label = QLabel("Default audio devices")
        gv.addWidget(self._audio_devices_label)
        layout.addWidget(g)

        g = QGroupBox("Application Settings")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        h.addWidget(QLabel("Grid timeout:"))
        self._grid_timeout_spin = QSpinBox()
        self._grid_timeout_spin.setRange(1, 60)
        self._grid_timeout_spin.setValue(30)
        self._grid_timeout_spin.setSuffix(" min")
        gv.addLayout(h)
        layout.addWidget(g)

        g = QGroupBox("About")
        gv = QVBoxLayout(g)
        gv.addWidget(QLabel("KV4P-Desktop — Ham Radio App"))
        gv.addWidget(QLabel("License: GPL v3"))
        gv.addWidget(QLabel("https://kv4p.com"))
        layout.addWidget(g)

        layout.addStretch()
        return tab

    # ── Centralized frame I/O with debug logging ─────────────────

    def _send_frame(self, cmd: int, payload: bytes = b'', label: str = ''):
        """Send a frame to the ESP32 with debug logging."""
        port = getattr(self._serial, '_port', None) if self._serial else None
        if not self._serial or not port:
            debug_log.warning(f"TX BLOCKED (no serial): {label} cmd=0x{cmd:02X}")
            return
        frame = DELIMITER + bytes([cmd]) + struct.pack('<H', len(payload)) + payload
        self._serial.cmd_queue.put(frame)
        self._tx_frame_count += 1
        self._tx_byte_count += len(frame)
        debug_log.info(f"TX frame #{self._tx_frame_count}: cmd=0x{cmd:02X} "
                        f"len={len(payload)} {label}")
        if debug_log.isEnabledFor(logging.DEBUG):
            debug_log.debug(f"TX hex:\n{_hex_dump(frame)}")
        # Keep last 50 TX frames in debug buffer
        entry = f"[0x{cmd:02X}] {label} ({len(payload)}B)"
        self._debug_tx_log.append(entry)
        if len(self._debug_tx_log) > 50:
            self._debug_tx_log.pop(0)
        # Append hex dump to debug hex display
        if hasattr(self, '_debug_hex_te') and cmd in (0x01, 0x02, 0x07):
            self._debug_hex_te.append(
                f"TX 0x{cmd:02X} {label}\n{_hex_dump(frame)}\n"
            )

    def _log_rx_frame(self, cmd: int, payload: bytes, label: str = ''):
        """Log received frame for debug."""
        self._rx_frame_count += 1
        self._rx_byte_count += len(payload)
        debug_log.info(f"RX frame #{self._rx_frame_count}: cmd=0x{cmd:02X} "
                        f"len={len(payload)} {label}")
        if debug_log.isEnabledFor(logging.DEBUG):
            debug_log.debug(f"RX hex:\n{_hex_dump(payload)}")
        entry = f"[0x{cmd:02X}] {label} ({len(payload)}B)"
        self._debug_rx_log.append(entry)
        if len(self._debug_rx_log) > 50:
            self._debug_rx_log.pop(0)
        if hasattr(self, '_debug_hex_te'):
            self._debug_hex_te.append(
                f"RX 0x{cmd:02X} {label}\n{_hex_dump(payload)}\n"
            )

    # ── Worker management ────────────────────────────────────────

    def _start_workers(self):
        self._serial = SerialWorker()
        self._serial.hello_received.connect(self._on_hello)
        self._serial.version_received.connect(self._on_version)
        self._serial.smeter.connect(self._on_smeter)
        self._serial.ax25_packet.connect(self._on_ax25)
        self._serial.debug_msg.connect(self._on_debug)
        self._serial.connected.connect(self._on_connected)
        self._serial.tx_cmd.connect(self._on_tx_cmd)
        self._serial.phys_ptt_changed.connect(self._on_phys_ptt)
        self._serial.start()

        self._audio = AudioWorker()
        self._serial.rx_opus_queue = self._audio.rx_opus_queue

        # Spectrum PCM queue: AudioWorker decoded PCM goes to spectrum analyzer
        self._spectrum_pcm_queue = queue.SimpleQueue()
        self._audio.spectrum_pcm_queue = self._spectrum_pcm_queue

        self._audio.opus_for_tx.connect(self._on_opus_from_mic)
        self._audio.aprs_packet.connect(self._on_aprs_demodulated)
        self._audio.morse_samples.connect(self._on_morse_audio)
        self._audio.debug_msg.connect(lambda m: self.log(f"[AUDIO] {m}"))
        self._audio.start()

    # ── Serial signals ────────────────────────────────────────────

    def _on_connected(self, ok: bool):
        self.connected = ok
        self._update_status()
        if ok:
            self.log("Connected to KV4P-HT board")
        else:
            self.log("Disconnected — searching...")

    def _on_hello(self):
        self.log("Board booted")

    def _on_version(self, info: dict):
        self.firmware_ver = info['ver']
        self.module_type = "VHF" if info['module_type'] == 0 else "UHF"
        self.radio_found = info['radio_status'] == '1'
        self._log_rx_frame(0x08, b'', f"VERSION v{info['ver']} {self.module_type}")
        self.log(f"Firmware v{info['ver']}, {self.module_type}, "
                 f"radio={'OK' if self.radio_found else 'MISSING'}, "
                 f"window={info['window_size']}, "
                 f"AFSK={'yes' if info.get('has_esp32_afsk') else 'no'}, "
                 f"HL={'yes' if info.get('has_hl') else 'no'}")
        self._update_status()

    def _on_morse_audio(self, pcm_data):
        """PCM audio from RX path → feed to morse decoder + SSTV decoder."""
        if self._morse_decoder_active:
            try:
                # pcm_data is float32 ndarray normalized to [-1, 1]
                level = float(np.abs(pcm_data).mean())
                is_down = level > 0.02
                now = time.monotonic()
                self._morse_decoder.process_key(is_down, timestamp=now)
                text = self._morse_decoder.get_text()
                if text and hasattr(self, '_cw_decode_display'):
                    self._cw_decode_display.append(text)
                    self._morse_decoder.reset()
            except Exception as e:
                debug_log.debug(f"Morse decode error: {e}")
        if self._sstv_decoder is not None:
            try:
                self._sstv_decoder.feed(pcm_data)
            except Exception as e:
                debug_log.debug(f"SSTV decode error: {e}")

    def _on_smeter(self, rssi: int):
        self.rssi = rssi
        self.s_meter = rssi_to_s_meter(rssi)
        if self._scanner:
            self._scanner.set_rssi(float(rssi))
        if self._rf_sweeper:
            self._rf_sweeper.set_rssi(float(rssi))

    def _update_smeter_ui(self):
        s = self.s_meter
        color = "green" if s <= 3 else "yellow" if s <= 5 else "orange" if s <= 7 else "red"
        width = int((s / 9) * self._smeter_bar.width())
        if width > 0:
            self._smeter_fill.setGeometry(0, 0, width, 24)
            self._smeter_fill.setStyleSheet(f"background-color: {color};")
        self._smeter_label.setText(f"S{s}")
        dbm = None
        if self.rssi > 0:
            import math
            dbm = 10 * math.log10(max(1, self.rssi)) - 120
            self._dbm_label.setText(f"dBm: {dbm:.0f}")
        if self._radio_face:
            self._radio_face.set_signal(s, dbm)

    def _update_igate_stats(self):
        self._igate_stats_te.clear()
        stats = self._igate_stats
        self._igate_stats_te.append(f"RF -> IS:   {stats.get('rf_to_is', 0)}")
        self._igate_stats_te.append(f"IS -> RF:   {stats.get('is_to_rf', 0)}")
        self._igate_stats_te.append(f"Errors:     {stats.get('errors', 0)}")
        if self._igate:
            self._igate_stats_te.append("Status:     Connected")
        else:
            self._igate_stats_te.append("Status:     Disconnected")

    def _on_aprs_is(self, line: str):
        self.log(f"APRS-IS: {line[:120]}")
        if line.startswith('#'):
            return
        self._igate_stats['is_to_rf'] = self._igate_stats.get('is_to_rf', 0) + 1
        self._update_igate_stats()
        if ':' in line:
            try:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    header = parts[0]
                    src = header.split('>')[0] if '>' in header else '?'
                    info = parts[1]
                    if info.startswith(':'):
                        dst_msg = info[1:]
                        colon = dst_msg.find(':')
                        if colon >= 0:
                            addr = dst_msg[:colon].strip()
                            text = dst_msg[colon + 1:]
                            self._aprs_log.append(f"[IS MSG] {src} -> {addr}: {text[:80]}")
                    elif info.startswith(('=', '!', '@', '/')):
                        self._aprs_log.append(f"[IS POS] {src}")
                    else:
                        self._aprs_log.append(f"[IS] {src}: {info[:80]}")
            except Exception:
                pass

    def _on_ax25(self, decoder_id: int, payload: bytes):
        self._log_rx_frame(0x0A, payload, f"AX25 decoder={decoder_id}")
        self.log(f"RX AX.25 decoder={decoder_id} len={len(payload)}")
        try:
            if len(payload) >= 2:
                payload = payload[:-2]
            decoded = decode_ax25_frame(payload)
            src = decoded.get('source', '?')
            dst = decoded.get('destination', '?')
            info = decoded.get('info', b'').decode('ascii', errors='replace')
            digis = ', '.join(decoded.get('digipeaters', []))
            path = f" via {digis}" if digis else ""

            parsed = parse_aprs(info, source=src)
            ptype = parsed.get('type', '?')

            self.log(f"AX.25 {src}->{dst}{path} [{ptype}] {info[:80]}")

            if ptype == 'message':
                addr = parsed.get('addressee', '')
                msg = parsed.get('text', info)
                msg_id = parsed.get('msg_id', '')
                self._aprs_log.append(
                    f"[MSG] {src} -> {addr}: {msg}"
                    + (f"  {{id={msg_id}" if msg_id else "")
                )
                if addr.upper() == self.callsign.upper() and msg_id:
                    ack_text = format_ack(src, msg_id)
                    self._send_aprs_rf(self.callsign, src, ack_text.encode())
                    self.log(f"Auto-ACK to {src} id={msg_id}")
            elif ptype == 'position':
                comment = parsed.get('comment', '')
                self._aprs_log.append(f"[POS] {src}: {comment}" if comment else f"[POS] {src}")
            elif ptype == 'status':
                self._aprs_log.append(f"[STS] {src}: {parsed.get('text', info)}")
            elif ptype == 'ack':
                self._aprs_log.append(f"[ACK] {src}: {info[:60]}")
            else:
                self._aprs_log.append(f"[{ptype.upper()}] {src}: {info[:80]}")

            if self.aprs_igate_on and self._igate:
                self._igate.send_to_is(src, info)
                self._igate_stats['rf_to_is'] = self._igate_stats.get('rf_to_is', 0) + 1
                self._update_igate_stats()

            if self.aprs_digi_on and self._digipeater:
                self._digipeater.process(payload)

            info_bytes = decoded.get('info', b'')

            if info_bytes[:2] == PROTO_ID:
                if self._file_sender:
                    from .ax25_file_transfer import FileTransferPacket
                    pkt = FileTransferPacket.unpack(info_bytes)
                    if pkt and pkt.cmd in (CMD_ACK, CMD_NAK):
                        self._file_sender.receive_ack(pkt.cmd, pkt.seq)
                        self.log(f"File transfer {'ACK' if pkt.cmd == CMD_ACK else 'NAK'} seq={pkt.seq}")
                if self._file_receiver:
                    self._file_receiver.process_packet(info_bytes)
            elif self._file_receiver:
                self._file_receiver.process_packet(info_bytes)

            if self._kiss_tnc:
                try:
                    self._kiss_tnc.send_ax25(payload)
                except Exception:
                    pass
        except Exception as e:
            self.log(f"AX.25 error: {e}")

    def _on_tx_cmd(self, cmd: int):
        names = {0x01: "PTT_DOWN", 0x02: "PTT_UP", 0x03: "GROUP", 0x06: "CONFIG",
                 0x07: "TX_AUDIO", 0x08: "HL", 0x0A: "AX25"}
        name = names.get(cmd, f"0x{cmd:02X}")
        if cmd in (0x0A, 0x01, 0x02):
            self.log(f"TX -> {name}")

    def _on_phys_ptt(self, pressed: bool):
        self.ptt = pressed
        if pressed:
            self._ptt_btn.setText("PTT (click to release)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #8b0000; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #a00000; }"
            )
        else:
            self._ptt_btn.setText("PTT (click to TX)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #3a9e3a; }"
            )
        if self._radio_face:
            self._radio_face.set_ptt(pressed)

    _demod_pkt_count = 0

    def _on_aprs_demodulated(self, parsed: dict):
        self.__class__._demod_pkt_count += 1
        n = self.__class__._demod_pkt_count
        self.log(f"Demod packet #{n}")
        src = parsed.get('source', '?')
        ptype = parsed.get('type', '?')
        info = parsed.get('raw', '')

        if ptype == 'message':
            addr = parsed.get('addressee', '')
            msg = parsed.get('text', info)
            msg_id = parsed.get('msg_id', '')
            self._aprs_log.append(
                f"[RX] {src} -> {addr}: {msg}"
                + (f"  {{id={msg_id}" if msg_id else "")
            )
            self.log(f"APRS msg from {src}: {msg}")
            if addr.upper() == self.callsign.upper() and msg_id:
                ack_text = format_ack(src, msg_id)
                self._send_aprs_rf(self.callsign, src, ack_text.encode())
                self.log(f"Auto-ACK to {src} id={msg_id}")
        elif ptype == 'position':
            comment = parsed.get('comment', '')
            self._aprs_log.append(f"[RX POS] {src}: {comment}" if comment else f"[RX POS] {src}")
        elif ptype == 'status':
            self._aprs_log.append(f"[RX STS] {src}: {parsed.get('text', info)}")
        elif ptype == 'beacon':
            self._aprs_log.append(f"[RX BCN] {src}: {parsed.get('text', info)[:80]}")
        else:
            self._aprs_log.append(f"[RX {ptype.upper()}] {src}: {info[:80]}")

        if self.aprs_igate_on and self._igate:
            self._igate.send_to_is(src, info)
            self._igate_stats['rf_to_is'] = self._igate_stats.get('rf_to_is', 0) + 1
            self._update_igate_stats()

        if self.aprs_digi_on and self._digipeater:
            body = parsed.get('raw_frame')
            if body:
                self._digipeater.process(body)

        raw_frame = parsed.get('raw_frame')
        if raw_frame:
            try:
                info_bytes = decode_ax25_frame(raw_frame).get('info', b'')
                if info_bytes[:2] == PROTO_ID:
                    if self._file_sender:
                        from .ax25_file_transfer import FileTransferPacket
                        pkt = FileTransferPacket.unpack(info_bytes)
                        if pkt and pkt.cmd in (CMD_ACK, CMD_NAK):
                            self._file_sender.receive_ack(pkt.cmd, pkt.seq)
                            self.log(f"FT {'ACK' if pkt.cmd == CMD_ACK else 'NAK'} seq={pkt.seq} (host demod)")
                    if self._file_receiver:
                        self._file_receiver.process_packet(info_bytes)
            except Exception:
                pass

    def _on_debug(self, level: int, text: str):
        level_names = {1: 'INFO', 2: 'ERROR', 3: 'WARN', 4: 'DEBUG', 5: 'TRACE'}
        tag = level_names.get(level, f'L{level}')
        self.log(f"[ESP {tag}] {text}")
        debug_log.log(max(10, min(40, level * 10)), f"[ESP] {text}")

    def _on_opus_from_mic(self, opus_frame: bytes):
        port = getattr(self._serial, '_port', None) if self._serial else None
        if self.ptt and self._serial and port:
            self._send_frame(0x07, opus_frame, f"TX_AUDIO ({len(opus_frame)}B)")
        elif self.ptt:
            debug_log.warning("PTT active but serial lost — dropping opus frame")

    # ── Radio control actions ────────────────────────────────────

    def _set_frequency(self):
        try:
            self.freq_rx = float(self._freq_edit.text())
            offset = float(self._offset_edit.text())
            self.freq_tx = self.freq_rx + offset
            self._freq_display.setText(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
            self._send_group()
            self.log(f"RX: {self.freq_rx} MHz, TX: {self.freq_tx} MHz")
            self._spectrum.set_center_freq(self.freq_rx)
            if hasattr(self, '_spectrum_display'):
                self._spectrum_display.set_center_freq(self.freq_rx)
        except ValueError:
            self.log("Invalid frequency")

    def _send_group(self):
        if self._serial:
            from .protocol import pack_group
            payload = pack_group(self.bandwidth, self.freq_tx, self.freq_rx,
                                 ctcss_to_index(self.ctcss_tx), self.squelch,
                                 ctcss_to_index(self.ctcss_rx))
            self._send_frame(0x03, payload, f"GROUP freq_tx={self.freq_tx:.3f}")

    def _toggle_ptt(self):
        if not self.connected:
            self.log("Board not connected — plug in KV4P-HT via USB")
            return
        self.ptt = not self.ptt
        if self._serial:
            cmd = 0x01 if self.ptt else 0x02
            self._send_frame(cmd, label="PTT_DOWN" if self.ptt else "PTT_UP")

        if self.ptt:
            self._ptt_btn.setText("PTT (click to release)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #8b0000; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #a00000; }"
            )
            self.log("PTT pressed")
        else:
            self._ptt_btn.setText("PTT (click to TX)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #3a9e3a; }"
            )
            self.log("PTT released")

        if self._radio_face:
            self._radio_face.set_ptt(self.ptt)

    def _set_squelch(self, value: int):
        self.squelch = value
        self._squelch_label.setText(str(value))
        self._send_group()

    def _set_bw(self, idx: int):
        self.bandwidth = idx
        self._send_group()

    def _set_power(self, high: bool):
        self.high_power = high
        self._hp_btn.setText("High (1W)" if high else "Low")
        if self._serial:
            from .protocol import pack_hl
            payload = pack_hl(high)
            self._send_frame(0x08, payload, f"HL power={'HIGH' if high else 'LOW'}")

    def _set_radio_mode(self, mode: str):
        self.radio_mode = "FM"
        for m, btn in self._mode_buttons.items():
            btn.setChecked(True)
        if self._rigctld:
            self._rigctld.set_mode("FM")
        if self._serial:
            from .protocol import pack_filters
            payload = pack_filters(True, False, False)
            self._send_frame(0x04, payload, "FILTERS mode=FM pre=True")
        self.log("Radio mode: FM")

    # ── CTCSS & Audio ─────────────────────────────────────────────

    def _set_ctcss(self):
        self.ctcss_tx = self._ctcss_tx_combo.currentData()
        self.ctcss_rx = self._ctcss_rx_combo.currentData()
        self._send_group()

    def _set_mic_gain(self, value: int):
        self.mic_gain = value / 100.0
        self._mic_gain_label.setText(f"{self.mic_gain:.1f}x")
        if self._audio:
            self._audio.mic_gain = self.mic_gain

    def _set_speaker_volume(self, value: int):
        self.speaker_volume = value / 100.0
        self._spk_vol_label.setText(f"{self.speaker_volume:.1f}x")
        if self._audio:
            self._audio.speaker_volume = self.speaker_volume

    # ── Spectrum actions ──────────────────────────────────────────

    def _on_spectrum_data(self, freqs, power_db, timestamp):
        self._waterfall.push(power_db)

    def _update_spectrum_display(self):
        if self._rf_sweep_mode:
            return
        fed = False
        if self._spectrum_pcm_queue is not None:
            try:
                while True:
                    pcm_data = self._spectrum_pcm_queue.get_nowait()
                    self._spectrum.feed(pcm_data)
                    fed = True
            except queue.Empty:
                pass

        if not fed and self._spectrum.get_spectrum() is None:
            import numpy as np
            noise = np.random.randn(self._spectrum.fft_size).astype(np.float32) * 0.001
            self._spectrum.feed(noise)

        result = self._spectrum.get_spectrum()
        if result is not None:
            freqs, power = result
            self._spectrum_display.update_spectrum(freqs, power)
            if self._spectrum_controls.show_peak_hold:
                peak = self._spectrum.get_peak_hold()
                if peak is not None:
                    self._spectrum_display.update_peak_hold(*peak)
            else:
                self._spectrum_display.update_peak_hold(None, None)
            if self._spectrum_controls.show_min_hold:
                mn = self._spectrum.get_min_hold()
                if mn is not None:
                    self._spectrum_display.update_min_hold(*mn)
            else:
                self._spectrum_display.update_min_hold(None, None)

        wf_matrix = self._waterfall.get_matrix()
        if wf_matrix is not None:
            self._waterfall_display.update_waterfall(wf_matrix)

    def _on_spectrum_settings_changed(self):
        if self._rf_sweep_mode:
            return
        span = self._spectrum_controls.span_hz
        fft_size = self._spectrum_controls.fft_size
        self._spectrum.set_span(span)
        self._spectrum.set_fft_size(fft_size)
        self._spectrum_display.set_span(span)
        self._spectrum_display._show_peak_hold = self._spectrum_controls.show_peak_hold
        self._spectrum_display._show_min_hold = self._spectrum_controls.show_min_hold
        if self._spectrum_controls.show_peak_hold:
            self._spectrum.reset_peak_hold()
        if self._spectrum_controls.show_min_hold:
            self._spectrum.reset_min_hold()

    def _on_reset_peaks(self):
        self._spectrum.reset_peak_hold()
        self._spectrum.reset_min_hold()
        self.log("Spectrum peak holds reset")

    def _on_spectrum_click(self, freq_mhz):
        self._freq_edit.setText(f"{freq_mhz:.3f}")
        self._set_frequency()

    # ── RF Sweep ───────────────────────────────────────────────

    def _make_rf_sweeper(self) -> RfSweeper:
        sweeper = RfSweeper(
            set_freq_callback=self._scan_sig.sweep_set_freq.emit,
            log_fn=self._scan_sig.sweep_log.emit,
        )
        sweeper.on_sweep_complete = (
            lambda f, r: self._scan_sig.sweep_complete.emit(f, r))
        sweeper.on_sweep_progress = (
            lambda c, t: self._scan_sig.sweep_progress.emit(int(c), int(t)))
        return sweeper

    def _on_rf_sweep_toggled(self, on: bool):
        self._rf_sweep_mode = on
        if on and self._rf_sweeper is None:
            self._rf_sweeper = self._make_rf_sweeper()
        elif not on:
            if self._rf_sweeper and self._rf_sweeper.is_sweeping:
                self._rf_sweeper.stop()
            self._rf_sweeper = None
            self._spectrum_display.set_range(-100, 0)
            self._spectrum_controls.set_sweep_status("")
        self._spec_info_label.setText(
            "RF Sweep" if on else "Audio FFT"
        )

    def _on_rf_sweep_start(self):
        if not self.connected:
            self.log("Board not connected — cannot sweep RF")
            return
        if self._rf_sweeper is None:
            self._rf_sweeper = self._make_rf_sweeper()
        sweeper = self._rf_sweeper
        sweeper.start_mhz = self._spectrum_controls.sweep_start_mhz
        sweeper.stop_mhz = self._spectrum_controls.sweep_stop_mhz
        sweeper.step_khz = self._spectrum_controls.sweep_step_khz
        self._spectrum_display.set_range(-120, 0)
        self._spectrum_display.set_center_freq(
            (sweeper.start_mhz + sweeper.stop_mhz) / 2
        )
        self._spectrum_display.set_span(
            (sweeper.stop_mhz - sweeper.start_mhz) * 1e6
        )
        sweeper.start()

    def _on_rf_sweep_stop(self):
        if self._rf_sweeper:
            self._rf_sweeper.stop()

    def _rf_sweep_set_freq(self, freq_mhz: float):
        if not self._serial:
            return
        freq_rx = freq_mhz
        freq_tx = freq_mhz + self.offset
        from .protocol import pack_group
        payload = pack_group(0, freq_tx, freq_rx,
                             ctcss_to_index(self.ctcss_tx), self.squelch,
                             ctcss_to_index(self.ctcss_rx))
        self._send_frame(0x03, payload, f"RF_SWEEP freq={freq_mhz:.3f}")

    def _on_rf_sweep_complete(self, freq_axis, rssi_values):
        import numpy as np
        dbm = np.array([rssi_to_dbm(r) for r in rssi_values], dtype=np.float64)
        self._spectrum_display.update_spectrum(freq_axis, dbm)
        wf_row = dbm.copy()
        self._waterfall.push(wf_row)
        wf_matrix = self._waterfall.get_matrix()
        if wf_matrix is not None:
            self._waterfall_display.update_waterfall(wf_matrix)
        self._spectrum_controls.set_sweep_status("sweep done")

    def _on_rf_sweep_progress(self, current, total):
        self._spectrum_controls.set_sweep_status(f"{current}/{total}")

    # ── External integration ─────────────────────────────────────

    def _toggle_rigctld(self, on: bool):
        if on:
            host = self._rigctl_host.text().strip() or "localhost"
            port = int(self._rigctl_port.text().strip() or "4532")
            self._rigctld = RigCtlD(
                host,
                port,
                freq_callback=self._scan_sig.rig_freq.emit,
                ptt_callback=lambda on: self._scan_sig.rig_ptt.emit(bool(on)),
                log_fn=lambda m: self._scan_sig.sweep_log.emit(f"[RIGCTLD] {m}"),
            )
            if self._rigctld.connect():
                self._rigctl_btn.setText("Connected")
                self.log(f"RigCtlD connected to {host}:{port}")
                self._digital_status.setText(f"Connected to rigctld at {host}:{port}")
            else:
                self._rigctl_btn.setChecked(False)
                self.log(f"RigCtlD connection to {host}:{port} failed")
                self._rigctld = None
        else:
            if self._rigctld:
                self._rigctld.disconnect()
                self._rigctld = None
            self._rigctl_btn.setText("Connect")
            self._digital_status.setText("Disconnected")
            self.log("RigCtlD disconnected")

    def _on_rigctld_freq(self, freq_mhz: float):
        self.freq_rx = freq_mhz
        self.freq_tx = freq_mhz + self.offset
        self._freq_edit.setText(f"{freq_mhz:.3f}")
        self._freq_display.setText(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
        self._send_group()
        self._spectrum.set_center_freq(freq_mhz)
        self._spectrum_display.set_center_freq(freq_mhz)
        self.log(f"[RIGCTLD] Frequency: {freq_mhz:.3f} MHz")

    def _on_rigctld_ptt(self, on: bool):
        if on == self.ptt:
            return
        self.ptt = on
        if self._serial:
            cmd = 0x01 if on else 0x02
            self._send_frame(cmd, label="PTT_DOWN" if on else "PTT_UP")
        if on:
            self._ptt_btn.setText("PTT (click to release)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #8b0000; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #a00000; }"
            )
            self.log("[RIGCTLD] PTT ON")
        else:
            self._ptt_btn.setText("PTT (click to TX)")
            self._ptt_btn.setStyleSheet(
                "QPushButton { background-color: #2d7d2d; color: white; font-weight: bold; font-size: 16px; }"
                "QPushButton:hover { background-color: #3a9e3a; }"
            )
            self.log("[RIGCTLD] PTT OFF")

    def _toggle_kiss(self, on: bool):
        if on:
            host = self._kiss_host.text().strip() or "localhost"
            port = int(self._kiss_port.text().strip() or "8001")
            self._kiss_tnc = KissTnc(host=host, tcp_port=port,
                                     callback=self._aprs_sig.kiss_frame.emit,
                                     log_fn=lambda m: self.log(f"[KISS] {m}"))
            if self._kiss_tnc.connect():
                self._kiss_tnc.start()
                self._kiss_btn.setText("Connected")
                self.log(f"KISS TNC connected to {host}:{port}")
            else:
                self._kiss_btn.setChecked(False)
                self.log(f"KISS TNC connection to {host}:{port} failed")
                self._kiss_tnc = None
        else:
            if self._kiss_tnc:
                self._kiss_tnc.stop()
                self._kiss_tnc.disconnect()
                self._kiss_tnc = None
            self._kiss_btn.setText("Connect")
            self.log("KISS TNC disconnected")

    def _on_kiss_ax25(self, frame_body: bytes):
        try:
            decoded = decode_ax25_frame(frame_body)
            src = decoded.get('source', '?')
            dst = decoded.get('destination', '?')
            info = decoded.get('info', b'').decode('ascii', errors='replace')
            self._aprs_log.append(f"[KISS] {src}->{dst}: {info[:80]}")
            if self.aprs_igate_on and self._igate:
                self._igate.send_to_is(src, info)
            if self.connected and self._serial:
                self._send_frame(0x0A, frame_body, f"TX_AX25 via KISS {src}->{dst}")
        except Exception as e:
            self.log(f"[KISS] Decode error: {e}")

    def _toggle_udp(self, on: bool):
        if on:
            self._udp_rx = UdpBroadcastRx(log_fn=self._udp_sig.log_line.emit)
            self._udp_rx.set_callback('wsjt-x', self._udp_sig.wsjt.emit)
            self._udp_rx.set_callback('direwolf', self._udp_sig.direwolf.emit)
            self._udp_rx.set_callback('fldigi', self._udp_sig.fldigi.emit)
            self._udp_rx.start()
            self.log("UDP broadcast listener started")
            self._digital_status.setText("UDP listener active")
        else:
            if self._udp_rx:
                self._udp_rx.stop()
                self._udp_rx = None
            self.log("UDP broadcast listener stopped")

    def _on_udp_log(self, msg: str):
        self.log(f"[UDP] {msg}")

    def _on_wsjtx_packet(self, pkt: dict):
        ptype = pkt.get('type', 0)
        if ptype == 1:
            freq = pkt.get('frequency', 0)
            mode = pkt.get('mode', '?')
            self.log(f"[WSJT-X] {mode} on {freq / 1e6:.3f} MHz")
        elif ptype == 2:
            msg = pkt.get('message', '')
            snr = pkt.get('snr', 0)
            self.log(f"[WSJT-X] Decode: {msg} (SNR {snr} dB)")

    def _on_direwolf_packet(self, pkt: dict):
        line = pkt.get('line', '')
        if line:
            self._aprs_log.append(f"[Dire Wolf] {line[:120]}")

    def _on_fldigi_packet(self, pkt: dict):
        fields = pkt.get('data', {})
        raw = pkt.get('raw', '')
        if fields:
            summary = "; ".join(f"{k}={v}" for k, v in fields.items())
            self.log(f"[FLD] {summary[:80]}")
        elif raw:
            self.log(f"[FLD] {raw[:80]}")

    # ── Scanner ────────────────────────────────────────────────────

    def _toggle_scan(self):
        if not self.scanning:
            band = self._scan_band.currentText()
            if band == "Custom":
                freqs = [self.freq_rx]
            else:
                freqs = BandPlan.get_preset_list(band.lower())
            if not freqs:
                self.log("No frequencies to scan")
                return
            dwell = self._scan_dwell.value()
            self._scanner = FrequencyScanner(
                set_freq_callback=self._scan_sig.scan_set_freq.emit,
                on_signal_callback=self._scan_sig.scan_on_signal.emit,
            )
            self._scanner.start_scan(freqs, dwell_ms=dwell)
            self.scanning = True
            self._scan_btn.setText("Stop Scan")
            self._scan_pause_btn.setEnabled(True)
            self._scan_pause_btn.setText("Pause")
            self._scan_status.setText(f"Scanning {len(freqs)} frequencies")
            self.log(f"Scanner started: {band}, {len(freqs)} freqs, {dwell}ms dwell")
        else:
            if self._scanner:
                self._scanner.stop_scan()
                self._scanner = None
            self.scanning = False
            self._scan_btn.setText("Start Scan")
            self._scan_pause_btn.setEnabled(False)
            self._scan_status.setText("Idle")
            self.log("Scanner stopped")

    def _pause_scan(self):
        if self._scanner and self.scanning:
            if self._scanner.is_paused:
                self._scanner.resume()
                self._scan_pause_btn.setText("Pause")
                self._scan_status.setText("Scanning (resumed)")
            else:
                self._scanner.pause()
                self._scan_pause_btn.setText("Resume")
                self._scan_status.setText("Scanning (paused)")

    def _scan_set_freq(self, freq_mhz: float):
        self.freq_rx = freq_mhz
        self.freq_tx = freq_mhz + self.offset
        self._freq_edit.setText(f"{freq_mhz:.3f}")
        self._freq_display.setText(f"TX: {self.freq_tx:.3f} MHz  RX: {self.freq_rx:.3f} MHz")
        if self._serial:
            self._send_group()

    def _scan_on_signal(self, freq_mhz: float, rssi: int):
        self._scan_status.setText(f"Signal on {freq_mhz:.3f} MHz (RSSI {rssi})")
        self.log(f"Scanner: signal on {freq_mhz:.3f} MHz, RSSI={rssi}")

    # ── SSTV actions ──────────────────────────────────────────────

    def _sstv_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image for SSTV", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All (*)"
        )
        if not path:
            return
        try:
            import numpy as np
            from PIL import Image
            img = Image.open(path)
            img = img.convert('RGB')
            self._sstv_image = np.array(img)
            preview = QPixmap(path)
            scaled = preview.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
            self._sstv_preview.setPixmap(scaled)
            self._sstv_image_label.setText(f"{img.size[0]}x{img.size[1]}")
            self.log(f"SSTV image loaded: {path}")
        except Exception as e:
            self.log(f"SSTV image load error: {e}")

    def _sstv_transmit(self):
        if self._sstv_image is None:
            self.log("No image loaded for SSTV TX")
            return
        if not self.connected:
            self.log("Board not connected")
            return

        mode_name = self._sstv_mode_combo.currentText()
        self.log(f"Encoding SSTV {mode_name}...")

        try:
            encoder = SstvEncoder(mode=mode_name, sample_rate=48000)
            waveform = encoder.encode_image(self._sstv_image)
            self.log(f"SSTV encoded: {len(waveform)} samples")
            self._tx_afsk_waveform(waveform)
        except Exception as e:
            self.log(f"SSTV encode error: {e}")

    def _sstv_toggle_decode(self, on: bool):
        if self._audio:
            self._audio.sstv_decoder_active = on
        if on:
            mode_name = self._sstv_mode_combo.currentText()
            self._sstv_decoder = SstvDecoder(
                mode=mode_name,
                sample_rate=48000,
                callback=self._on_sstv_decoded,
            )
            self._sstv_decode_btn.setText("Stop SSTV Decode")
            self._sstv_decode_status.setText(f"Decoding {mode_name}...")
            self.log(f"SSTV decoder started ({mode_name})")
        else:
            self._sstv_decoder = None
            self._sstv_decode_btn.setText("Start SSTV Decode")
            self._sstv_decode_status.setText("Idle")
            self.log("SSTV decoder stopped")

    def _on_sstv_decoded(self, image: np.ndarray, mode: str):
        import numpy as np
        from PIL import Image
        from io import BytesIO
        import base64

        h, w = image.shape[:2]
        img = Image.fromarray(image)
        buf = BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        self._sstv_rx_preview.setPixmap(
            QPixmap.fromImage(QImage.fromData(buf.getvalue())).scaled(
                200, 150, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        timestamp = time.strftime("%H:%M:%S")
        self._sstv_gallery.append(
            f"[{timestamp}] SSTV {mode}: {w}x{h} pixels"
        )
        self.log(f"SSTV image decoded: {w}x{h} ({mode})")

    # ── Morse / CW actions ────────────────────────────────────────

    def _morse_set_wpm(self, wpm: int):
        self._morse_keyer.set_wpm(wpm)
        self._morse_decoder.set_wpm(wpm)

    def _morse_set_tone(self, hz: int):
        self._morse_keyer.tone_hz = hz

    def _morse_set_mode(self, idx: int):
        modes = ["iambic", "iambic", "straight"]
        self._morse_keyer.mode = modes[idx]

    def _morse_send_text(self):
        text = self._cw_input.text().strip()
        if not text:
            return
        if not self.connected:
            self.log("Board not connected — cannot TX CW")
            return
        self.log(f"CW: {text}")

        waveform = self._morse_keyer.generate_tone_array(text)
        if len(waveform) > 0:
            self._cw_decode_display.append(f"TX: {text}")
            self._tx_morse_waveform(waveform)

    def _morse_send_tone(self):
        self.log("CW tone test")
        waveform = self._morse_keyer.generate_tone_array("TEST")
        if len(waveform) > 0:
            self._cw_decode_display.append("TX: TEST")
            self._tx_morse_waveform(waveform)

    def _tx_morse_waveform(self, waveform):
        """TX a numpy float waveform as Opus frames over serial with PTT (background)."""
        self._start_tx_worker(waveform, "CW")

    def _tx_afsk_waveform(self, waveform):
        """TX a numpy float waveform as Opus frames over serial with PTT (background)."""
        self._start_tx_worker(waveform, "AFSK")

    def _start_tx_worker(self, waveform, label: str = "TX"):
        """Start a TxWorker to transmit a waveform without blocking the UI."""
        if self._tx_worker and self._tx_worker.isRunning():
            self.log(f"{label} TX already in progress")
            return
        port = getattr(self._serial, '_port', None) if self._serial else None
        if not self._serial or not port:
            self.log(f"Cannot {label} TX: serial not connected")
            return
        if not self._serial.cmd_queue:
            self.log(f"Cannot {label} TX: no command queue")
            return
        self._tx_worker = TxWorker(waveform, self._serial.cmd_queue)
        self._tx_worker.finished.connect(lambda ok, msg: self._on_tx_worker_done(ok, msg, label))
        self.log(f"{label} TX starting ({len(waveform)} samples)...")
        self._tx_worker.start()

    def _on_tx_worker_done(self, success: bool, msg: str, label: str):
        if success:
            self.log(f"{label} TX: {msg}")
        else:
            self.log(f"{label} TX failed: {msg}")

    def _morse_generate_practice(self):
        text = self._practice_gen.generate_exchange()
        self._cw_practice_text.setPlainText(text)
        self.log(f"CW Practice: {text}")

    def _morse_play_practice(self):
        text = self._cw_practice_text.toPlainText().strip()
        if text:
            waveform = self._morse_keyer.generate_tone_array(text)
            self.log(f"CW Playing practice: {text[:40]}...")
            if len(waveform) > 0:
                self._tx_morse_waveform(waveform)

    def _morse_toggle_decoder(self, on: bool):
        self._morse_decoder_active = on
        if self._audio:
            self._audio.morse_decoder_active = on
        if on:
            self._morse_decoder.reset()
            self._cw_decoder_btn.setText("Stop Decoder")
            self._cw_decoder_status.setText("Listening...")
            self.log("Morse decoder started")
        else:
            self._cw_decoder_btn.setText("Start Decoder")
            self._cw_decoder_status.setText("Stopped")
            self.log("Morse decoder stopped")

    # ── File transfer actions ─────────────────────────────────────

    def _ft_log_ui(self, msg: str):
        self._ft_log.append(msg)

    def _ft_progress_ui(self, sent: int, total: int):
        if total > 0:
            pct = int(sent * 100 / total)
            self._ft_progress.setValue(pct)
            self._ft_status.setText(f"{sent}/{total} packets ({pct}%)")

    def _ft_complete_ui(self, success: bool, msg: str):
        if success:
            self._ft_progress.setValue(100)
            self._ft_status.setText(f"Complete: {msg}")
            self.log(f"File transfer complete: {msg}")
        else:
            self._ft_status.setText(f"Failed: {msg}")
            self.log(f"File transfer failed: {msg}")

    def _ft_log_from_thread(self, msg: str):
        self._ft_sig.log_msg.emit(msg)

    def _ft_progress_from_thread(self, sent: int, total: int):
        self._ft_sig.progress.emit(sent, total)

    def _ft_complete_from_thread(self, success: bool, msg: str):
        self._ft_sig.complete.emit(success, msg)

    def _ft_send_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File for AX.25 Transfer")
        if not path:
            return
        dest = self._ft_dest.text().strip().upper()
        if not dest:
            self.log("Set destination callsign for file transfer")
            return

        path_list = [p.strip() for p in self._ft_path.text().split(',') if p.strip()]

        self._file_sender = FileTransferSender(
            source_call=self.callsign,
            dest_call=dest,
            tx_callback=self._ft_tx_ax25,
            log_fn=self._ft_log_from_thread,
            progress_callback=self._ft_progress_from_thread,
            on_complete=self._ft_complete_from_thread,
        )
        self._file_sender.send_file(path, path_list)
        self._ft_status.setText(f"Sending: {path.split('/')[-1]}")
        self.log(f"File transfer started: {path} -> {dest}")

    def _ft_tx_ax25(self, ax25_frame: bytes):
        if not self._serial or not self._serial.cmd_queue:
            return
        from .afsk import build_tx_waveform_from_body
        waveform = build_tx_waveform_from_body(ax25_frame)
        if self._audio:
            import numpy as np
            self._audio.inject_audio(waveform.copy())
        worker = TxWorker(waveform, self._serial.cmd_queue)
        worker.start()
        worker.wait(timeout=30)

    def _ft_toggle_receiver(self, on: bool):
        if on:
            dest = self._ft_dest.text().strip().upper() or self.callsign
            self._file_receiver = FileTransferReceiver(
                source_call=self.callsign,
                dest_call=dest,
                tx_callback=self._ft_tx_ax25,
                log_fn=self._ft_log_from_thread,
                progress_callback=self._ft_progress_from_thread,
                on_complete=self._ft_complete_from_thread,
            )
            self._file_receiver.start()
            self._ft_receive_btn.setText("Stop Receiver")
            self._ft_status.setText("Receiving...")
            self.log("AX.25 file receiver started")
        else:
            if self._file_receiver:
                self._file_receiver.stop()
                self._file_receiver = None
            self._ft_receive_btn.setText("Start Receiver")
            self._ft_status.setText("Idle")
            self.log("AX.25 file receiver stopped")

    # ── APRS actions ──────────────────────────────────────────────

    def _on_latlon_changed(self):
        self.aprs_lat = _try_float(self._lat_edit.text(), 0.0)
        self.aprs_lon = _try_float(self._lon_edit.text(), 0.0)
        if self._igate:
            self._igate.lat = self.aprs_lat
            self._igate.lon = self.aprs_lon

    def _toggle_igate(self, on: bool):
        self.aprs_igate_on = on
        self._igate_btn.setText("iGate ON" if on else "iGate OFF")
        if on:
            self._igate = IGate(
                self.callsign,
                rf_tx_callback=self._emit_rf_tx,
                aprs_is_callback=self._aprs_sig.is_line.emit,
                lat=self.aprs_lat, lon=self.aprs_lon,
                filter_str=self._igate_filter.text().strip(),
                tx_enabled=self._igate_tx.isChecked(),
                status_text=self._igate_status.text().strip(),
            )
            self._igate.start()
            self.log("APRS iGate started")
        else:
            if self._igate:
                self._igate.stop()
                self._igate = None
            self.log("APRS iGate stopped")

    def _toggle_digi(self, on: bool):
        self.aprs_digi_on = on
        self._digi_btn.setText("Digi ON" if on else "Digi OFF")
        if on:
            self._digipeater = Digipeater(self.callsign, self._rf_tx_callback)
            self.log("Digipeater started")
        else:
            self._digipeater = None
            self.log("Digipeater stopped")

    def _toggle_beacon(self, on: bool):
        self.aprs_beacon_on = on
        self._beacon_btn.setText("Beacon ON" if on else "Beacon OFF")
        if on:
            self._beacon_remaining = self.aprs_beacon_interval
            self._beacon_timer.start(1000)
            self._send_aprs_beacon()
            self.log(f"APRS beacon every {self.aprs_beacon_interval}s")
        else:
            self._beacon_timer.stop()
            self.log("APRS beacon stopped")

    def _aprs_send_message(self):
        dest = self._aprs_to.text().strip().upper()
        text = self._aprs_msg.text().strip()
        if not dest:
            self.log("Set destination callsign (To: field)")
            return
        if not text:
            self.log("Enter a message")
            return
        if not self.connected:
            self.log("Board not connected")
            return

        info = format_message(dest, text)
        self.log(f"Generating AFSK for {dest}...")
        waveform = build_tx_waveform(self.callsign, dest, [], info.encode())
        self._tx_afsk_waveform(waveform)
        if self.aprs_igate_on and self._igate:
            self._igate.send_to_is(self.callsign, info)
        self._aprs_log.append(f"[TX MSG] {self.callsign} -> {dest}: {text}")
        self.log(f"APRS message sent to {dest}")
        self._aprs_msg.clear()

    def _aprs_test_decode(self):
        if self._audio:
            self._test_status.setText("Testing...")
            self._audio.inject_selftest()
            QTimer.singleShot(500, lambda: self._test_status.setText("Injected — check log"))
        else:
            self._test_status.setText("Audio not ready")

    def _send_aprs_rf(self, source: str, dest: str, info: bytes,
                      digipeaters: list[str] | None = None):
        if digipeaters is None:
            digipeaters = [self.aprs_path] if self.aprs_path else []
        waveform = build_tx_waveform(source, dest, digipeaters, info)
        self._tx_afsk_waveform(waveform)
        self.log(f"APRS TX {source}->{dest} via {','.join(digipeaters) if digipeaters else 'direct'}")

    def _emit_rf_tx(self, ax25_body: bytes, from_igate: bool = False):
        """Called from the IGate thread — forward to GUI thread via signal."""
        self._aprs_sig.rf_tx.emit(bytes(ax25_body), bool(from_igate))

    def _rf_tx_callback(self, ax25_body: bytes, from_igate: bool = False):
        if not self._serial:
            return
        try:
            decoded = decode_ax25_frame(ax25_body)
            src = decoded.get('source', self.callsign)
            dst = decoded.get('destination', 'APZ010')
            info = decoded.get('info', b'')
            if from_igate:
                self._aprs_log.append(f"[IS TX] {info[:80].decode('ascii', errors='replace')}")
            self._send_aprs_rf(src, dst, info, decoded.get('digipeaters', []))
        except Exception as e:
            self.log(f"RF TX error: {e}")

    def _send_aprs_beacon(self):
        if not self.aprs_beacon_on:
            return
        self._beacon_remaining -= 1
        if self._beacon_remaining <= 0:
            self._beacon_remaining = self.aprs_beacon_interval
            info = format_beacon(self.callsign, self.aprs_lat, self.aprs_lon,
                                 comment="KV4P-Desktop", symbol=self.aprs_symbol)
            self._send_aprs_rf(self.callsign, "APZ010", info.encode())
            if self.aprs_igate_on and self._igate:
                self._igate.send_to_is(self.callsign, info)
            self.log(f"APRS beacon sent ({self.aprs_lat}, {self.aprs_lon})")

    # ── Misc ──────────────────────────────────────────────────────

    def _update_status(self):
        parts = []
        if self.connected:
            parts.append("CONNECTED")
        else:
            parts.append("DISCONNECTED")
        if self.firmware_ver:
            parts.append(f"FW v{self.firmware_ver}")
        if self.module_type:
            parts.append(self.module_type)
        parts.append(f"Mode:{self.radio_mode}")
        status = " | ".join(parts)
        self._status_label.setText(status)

    def log(self, msg: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")
        doc = self._log.document()
        if doc.blockCount() > 300:
            cursor = self._log.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        sb = self._log.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def closeEvent(self, event):
        self.log("Shutting down...")
        self._save_settings()
        if self._tx_worker and self._tx_worker.isRunning():
            self._tx_worker.requestInterruption()
            self._tx_worker.wait(3000)
        self._beacon_timer.stop()
        if self._scanner:
            self._scanner.stop_scan()
        if self._rf_sweeper:
            self._rf_sweeper.stop()
        if self._igate:
            self._igate.stop()
        if self._udp_rx:
            self._udp_rx.stop()
        if self._kiss_tnc:
            self._kiss_tnc.stop()
            self._kiss_tnc.disconnect()
        if self._rigctld:
            self._rigctld.disconnect()
        if self._serial:
            self._serial.stop()
        if self._audio:
            self._audio.stop()


# Need numpy import at module level for type hints
import numpy as np
