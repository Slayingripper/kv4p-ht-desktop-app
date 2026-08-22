"""
Radio-like faceplate widget — large frequency display, analog S-meter, channel knob.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, QLinearGradient,
    QRadialGradient, QPainterPath,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QSizePolicy, QFrame,
)

# ── Color palette ──────────────────────────────────────────────
_BG = QColor(20, 20, 25)
_PANEL = QColor(30, 30, 35)
_BEZEL = QColor(50, 50, 55)
_LCD_BG = QColor(10, 40, 10)
_LCD_FG = QColor(0, 220, 80)
_LCD_DIM = QColor(0, 60, 20)
_METER_BG = QColor(15, 15, 20)
_METER_GREEN = QColor(40, 200, 40)
_METER_YELLOW = QColor(220, 200, 40)
_METER_RED = QColor(220, 40, 40)
_BTN_TX = QColor(180, 30, 30)
_BTN_RX = QColor(30, 120, 30)
_TEXT = QColor(200, 200, 200)
_TEXT_DIM = QColor(120, 120, 120)


class AnalogSMeter(QWidget):
    """Draws an analog S-meter with a needle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._s_value = 0.0  # 0..9+
        self._dbm = None
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_signal(self, s_meter: int, dbm: float | None = None):
        self._s_value = max(0.0, min(9.0, float(s_meter)))
        self._dbm = dbm
        self.update()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.fillRect(0, 0, w, h, _METER_BG)

        cx = w / 2
        cy = h * 0.85
        radius = min(w * 0.42, h * 0.7)

        # Arc sweep: -150° to +30° (180° total) from 3 o'clock
        start_angle = 150  # degrees from 3 o'clock (CCW)
        span = -180

        # Draw colored arc segments
        arcs = [
            (0, 1, _METER_GREEN),
            (1, 3, _METER_GREEN),
            (3, 5, _METER_YELLOW),
            (5, 7, _METER_YELLOW),
            (7, 9, _METER_RED),
        ]
        for s_lo, s_hi, color in arcs:
            a1 = start_angle + span * (s_lo / 9.0)
            a2 = start_angle + span * (s_hi / 9.0)
            pen = QPen(color, 3)
            p.setPen(pen)
            p.drawArc(
                QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                int(a1 * 16), int((a2 - a1) * 16),
            )

        # Tick marks and labels
        p.setPen(QPen(_TEXT_DIM, 1))
        small_font = QFont("monospace", 8)
        p.setFont(small_font)
        for s in range(10):
            frac = s / 9.0
            angle = math.radians(start_angle + span * frac)
            inner = radius * 0.82
            outer = radius * 0.95
            x1 = cx + inner * math.cos(angle)
            y1 = cy - inner * math.sin(angle)
            x2 = cx + outer * math.cos(angle)
            y2 = cy - outer * math.sin(angle)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
            if s <= 9:
                lx = cx + (radius * 0.72) * math.cos(angle)
                ly = cy - (radius * 0.72) * math.sin(angle)
                p.drawText(QPointF(lx - 5, ly + 4), str(s))

        # "S" and "dBm" labels
        p.setPen(QPen(_TEXT, 1))
        p.setFont(QFont("sans-serif", 9))
        p.drawText(QPointF(cx - radius * 0.5, cy - radius * 0.15), "S")
        p.drawText(QPointF(cx + radius * 0.3, cy - radius * 0.15), "dB")

        # Needle
        frac = self._s_value / 9.0
        angle = math.radians(start_angle + span * frac)
        needle_len = radius * 0.88
        nx = cx + needle_len * math.cos(angle)
        ny = cy - needle_len * math.sin(angle)

        p.setPen(QPen(QColor(220, 40, 40), 2))
        p.drawLine(int(cx), int(cy), int(nx), int(ny))

        # Center dot
        p.setBrush(QBrush(QColor(180, 40, 40)))
        p.drawEllipse(QPointF(cx, cy), 4, 4)

        p.end()


class FrequencyDisplay(QWidget):
    """Large LCD-style frequency display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freq_mhz = 144.390
        self._label = "RX"
        self._mode = "FM"
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_frequency(self, freq_mhz: float, label: str = "RX"):
        self._freq_mhz = freq_mhz
        self._label = label
        self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # LCD panel with bezel
        margin = 6
        lcd_rect = QRectF(margin, margin, w - 2 * margin, h - 2 * margin)
        p.setPen(QPen(QColor(70, 70, 70), 2))
        p.setBrush(QBrush(_LCD_BG))
        p.drawRoundedRect(lcd_rect, 6, 6)

        # Frequency text
        freq_str = f"{self._freq_mhz:.4f}"
        font = QFont("monospace", max(14, h // 5), QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QPen(_LCD_FG))
        fm = QFontMetrics(font)
        text_rect = fm.boundingRect(freq_str)
        tx = margin + (w - 2 * margin - text_rect.width()) / 2
        ty = margin + (h - 2 * margin + text_rect.height()) / 2 - 8
        p.drawText(QPointF(tx, ty), freq_str)

        # "MHz" suffix
        small = QFont("monospace", max(9, h // 8))
        p.setFont(small)
        p.setPen(QPen(_LCD_DIM))
        p.drawText(QPointF(tx + text_rect.width() + 4, ty - 6), "MHz")

        # Mode badge (top right)
        mode_font = QFont("sans-serif", max(8, h // 10), QFont.Weight.Bold)
        p.setFont(mode_font)
        p.setPen(QPen(_LCD_DIM))
        p.drawText(QPointF(w - margin - 40, margin + 18), self._mode)

        # Label (top left)
        p.drawText(QPointF(margin + 8, margin + 18), self._label)

        p.end()


class RadioFaceplate(QWidget):
    """
    Full radio-like faceplate with frequency display, S-meter, PTT, mode buttons,
    channel selector, and squelch.
    """

    ptt_clicked = pyqtSignal()
    frequency_changed = pyqtSignal(float)
    mode_changed = pyqtSignal(str)
    channel_changed = pyqtSignal(int)
    channel_added = pyqtSignal()
    squelch_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG.name()};")
        self._ptt_on = False
        self._channels: list[dict] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # ── Top row: Channel selector + Add ──
        ch_row = QHBoxLayout()
        ch_row.setSpacing(4)
        lbl = QLabel("CH:")
        lbl.setStyleSheet(f"color: {_TEXT_DIM.name()}; font-weight: bold;")
        ch_row.addWidget(lbl)

        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(180)
        self._channel_combo.setStyleSheet(
            f"QComboBox {{ background-color: {_PANEL.name()}; color: {_TEXT.name()}; "
            f"border: 1px solid {_BEZEL.name()}; border-radius: 3px; padding: 3px 8px; font-size: 13px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background-color: {_PANEL.name()}; color: {_TEXT.name()}; }}"
        )
        self._channel_combo.currentIndexChanged.connect(self._on_channel_select)
        ch_row.addWidget(self._channel_combo, stretch=1)

        self._add_ch_btn = QPushButton("+")
        self._add_ch_btn.setFixedSize(28, 28)
        self._add_ch_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_PANEL.name()}; color: {_TEXT.name()}; "
            f"border: 1px solid {_BEZEL.name()}; border-radius: 3px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {_BEZEL.name()}; }}"
        )
        self._add_ch_btn.clicked.connect(self.channel_added)
        ch_row.addWidget(self._add_ch_btn)

        layout.addLayout(ch_row)

        # ── Frequency display ──
        self._freq_display = FrequencyDisplay()
        layout.addWidget(self._freq_display)

        # ── S-meter ──
        self._smeter = AnalogSMeter()
        layout.addWidget(self._smeter)

        # ── Signal readout ──
        sig_row = QHBoxLayout()
        self._s_readout = QLabel("S0")
        self._s_readout.setStyleSheet(f"color: {_LCD_FG.name()}; font-family: monospace; font-size: 13px;")
        sig_row.addWidget(self._s_readout)
        sig_row.addStretch()
        self._dbm_readout = QLabel("-- dBm")
        self._dbm_readout.setStyleSheet(f"color: {_LCD_DIM.name()}; font-family: monospace; font-size: 12px;")
        sig_row.addWidget(self._dbm_readout)
        layout.addLayout(sig_row)

        # ── Mode buttons ──
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode in ["FM"]:
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setMinimumSize(52, 32)
            btn.setStyleSheet(self._mode_btn_style(False))
            btn.clicked.connect(lambda checked, m=mode: self._on_mode(m))
            mode_row.addWidget(btn)
            self._mode_buttons[mode] = btn
        layout.addLayout(mode_row)

        # ── PTT button ──
        self._ptt_btn = QPushButton("PTT")
        self._ptt_btn.setMinimumHeight(56)
        self._ptt_btn.setStyleSheet(self._ptt_style(False))
        self._ptt_btn.clicked.connect(self.ptt_clicked)
        layout.addWidget(self._ptt_btn)

        # ── Bottom: squelch ──
        sq_row = QHBoxLayout()
        sq_lbl = QLabel("SQL:")
        sq_lbl.setStyleSheet(f"color: {_TEXT_DIM.name()}; font-size: 11px;")
        sq_row.addWidget(sq_lbl)
        self._sql_spin = QSpinBox()
        self._sql_spin.setRange(0, 8)
        self._sql_spin.setFixedWidth(40)
        self._sql_spin.setStyleSheet(
            f"QSpinBox {{ background-color: {_PANEL.name()}; color: {_TEXT.name()}; "
            f"border: 1px solid {_BEZEL.name()}; border-radius: 2px; font-size: 12px; }}"
        )
        self._sql_spin.valueChanged.connect(self.squelch_changed)
        sq_row.addWidget(self._sql_spin)
        sq_row.addStretch()
        layout.addLayout(sq_row)

    # ── Public API ─────────────────────────────────────────────

    def set_frequency(self, freq_mhz: float, label: str = "RX"):
        self._freq_display.set_frequency(freq_mhz, label)

    def set_mode(self, mode: str):
        self._freq_display.set_mode(mode)
        for m, btn in self._mode_buttons.items():
            btn.setChecked(m == mode)
            btn.setStyleSheet(self._mode_btn_style(m == mode))

    def set_signal(self, s_meter: int, dbm: float | None = None):
        self._smeter.set_signal(s_meter, dbm)
        self._s_readout.setText(f"S{s_meter}")
        if dbm is not None:
            self._dbm_readout.setText(f"{dbm:.0f} dBm")

    def set_ptt(self, on: bool):
        self._ptt_on = on
        self._ptt_btn.setStyleSheet(self._ptt_style(on))
        self._ptt_btn.setText("TRANSMIT" if on else "PTT")

    def set_squelch(self, value: int):
        self._sql_spin.blockSignals(True)
        self._sql_spin.setValue(value)
        self._sql_spin.blockSignals(False)

    def set_channels(self, channels: list[dict]):
        """channels: list of dicts with 'name' and 'freq_rx' keys."""
        self._channels = channels
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for ch in channels:
            self._channel_combo.addItem(
                f"{ch['name']}  {ch['freq_rx']:.3f}", ch.get('index', -1)
            )
        self._channel_combo.blockSignals(False)

    def set_channel_index(self, idx: int):
        """Highlight the given channel without emitting signal."""
        self._channel_combo.blockSignals(True)
        self._channel_combo.setCurrentIndex(idx)
        self._channel_combo.blockSignals(False)

    # ── Internal ───────────────────────────────────────────────

    def _on_mode(self, mode: str):
        self.set_mode(mode)
        self.mode_changed.emit(mode)

    def _on_channel_select(self, idx: int):
        if idx >= 0:
            self.channel_changed.emit(idx)

    @staticmethod
    def _mode_btn_style(active: bool) -> str:
        bg = _LCD_FG.name() if active else _PANEL.name()
        fg = _LCD_BG.name() if active else _TEXT.name()
        return (
            f"QPushButton {{ background-color: {bg}; color: {fg}; "
            f"border: 1px solid {_BEZEL.name()}; border-radius: 3px; font-weight: bold; font-size: 12px; }}"
            f"QPushButton:hover {{ border-color: {_LCD_FG.name()}; }}"
        )

    @staticmethod
    def _ptt_style(on: bool) -> str:
        bg = _BTN_TX.name() if on else _BTN_RX.name()
        fg = "white"
        return (
            f"QPushButton {{ background-color: {bg}; color: {fg}; "
            f"font-weight: bold; font-size: 18px; border-radius: 6px; "
            f"border: 2px solid {'#ff4444' if on else '#44aa44'}; }}"
            f"QPushButton:hover {{ border-color: {'#ff6666' if on else '#66cc66'}; }}"
        )
