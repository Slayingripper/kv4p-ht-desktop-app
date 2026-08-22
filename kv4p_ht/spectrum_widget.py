"""
PyQt6 widgets for spectrum and waterfall display.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SpectrumDisplay(QWidget):
    """Draws FFT spectrum as a curve plot."""

    spectrum_clicked = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._freqs = None
        self._power = None
        self._peak_hold = None
        self._min_hold = None
        self._ref_level = 0.0
        self._min_db = -100.0
        self._max_db = 0.0
        self._center_freq_mhz = 144.390
        self._span_hz = 48000.0
        self._marker_freq = None
        self._show_peak_hold = False
        self._show_min_hold = False

    def update_spectrum(self, freqs, power_db):
        self._freqs = freqs
        self._power = power_db
        self.update()

    def update_peak_hold(self, freqs, power_db):
        self._peak_hold = power_db
        self.update()

    def update_min_hold(self, freqs, power_db):
        self._min_hold = power_db
        self.update()

    def set_center_freq(self, freq_mhz: float):
        self._center_freq_mhz = freq_mhz

    def set_span(self, span_hz: float):
        self._span_hz = max(100, span_hz)

    def set_range(self, min_db: float, max_db: float):
        self._min_db = min_db
        self._max_db = max_db

    def set_marker(self, freq_mhz: float | None):
        self._marker_freq = freq_mhz
        self.update()

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor(0, 0, 0)
        painter.fillRect(0, 0, w, h, bg)

        grid_pen = QPen(QColor(40, 40, 40), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(5):
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)
        for i in range(9):
            x = int(w * i / 8)
            painter.drawLine(x, 0, x, h)

        label_pen = QPen(QColor(100, 100, 100))
        painter.setPen(label_pen)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for i in range(5):
            db = self._max_db - (self._max_db - self._min_db) * i / 4
            y = int(h * i / 4)
            painter.drawText(2, y + 12, f"{db:.0f}")

        freq_min = self._center_freq_mhz - self._span_hz / 2e6
        freq_max = self._center_freq_mhz + self._span_hz / 2e6
        for i in range(0, 9, 2):
            f = freq_min + (freq_max - freq_min) * i / 8
            x = int(w * i / 8)
            painter.drawText(x, h - 2, f"{f:.3f}")

        if self._freqs is None or self._power is None:
            painter.end()
            return

        import numpy as np

        def freq_to_x(freq_mhz):
            return int((freq_mhz - freq_min) / (freq_max - freq_min) * w)

        def db_to_y(db):
            db = max(self._min_db, min(self._max_db, db))
            return int((self._max_db - db) / (self._max_db - self._min_db) * h)

        if self._show_peak_hold and self._peak_hold is not None:
            peak_pen = QPen(QColor(255, 165, 0, 180), 1)
            painter.setPen(peak_pen)
            self._draw_curve(painter, self._freqs, self._peak_hold, w, h,
                             freq_to_x, db_to_y)

        if self._show_min_hold and self._min_hold is not None:
            min_pen = QPen(QColor(0, 200, 255, 180), 1)
            painter.setPen(min_pen)
            self._draw_curve(painter, self._freqs, self._min_hold, w, h,
                             freq_to_x, db_to_y)

        spec_pen = QPen(QColor(0, 255, 0), 1)
        painter.setPen(spec_pen)
        self._draw_curve(painter, self._freqs, self._power, w, h,
                         freq_to_x, db_to_y)

        if self._marker_freq is not None:
            mx = freq_to_x(self._marker_freq)
            marker_pen = QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine)
            painter.setPen(marker_pen)
            painter.drawLine(mx, 0, mx, h)

        painter.end()

    def _draw_curve(self, painter, freqs, power, w, h, freq_to_x, db_to_y):
        n = len(power)
        if n < 2:
            return
        step = max(1, n // w)
        points = []
        for i in range(0, n, step):
            x = int(i * w / n)
            y = db_to_y(power[i])
            y = max(0, min(h - 1, y))
            points.append((x, y))

        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1],
                             points[i + 1][0], points[i + 1][1])

    def mousePressEvent(self, event):
        if self._freqs is None:
            return
        x = event.position().x()
        w = self.width()
        freq_min = self._center_freq_mhz - self._span_hz / 2e6
        freq_max = self._center_freq_mhz + self._span_hz / 2e6
        freq = freq_min + (x / w) * (freq_max - freq_min)
        self.spectrum_clicked.emit(freq)


class WaterfallDisplay(QWidget):
    """Scrolling waterfall / spectrogram display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self._image = None
        self._colormap = 'turbo'

    def update_waterfall(self, matrix):
        if matrix is None or matrix.size == 0:
            return
        import numpy as np
        h, w = matrix.shape
        target_w = min(w, self.width())
        if target_w < 2:
            return
        if target_w < w:
            indices = np.linspace(0, w - 1, target_w).astype(int)
            matrix = matrix[:, indices]
        min_val = float(matrix.min())
        max_val = float(matrix.max())
        if max_val - min_val < 1:
            max_val = min_val + 1
        normalized = ((matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        colored = self._apply_colormap(normalized)
        self._image = QPixmap.fromImage(
            self._numpy_to_qimage(colored, target_w, h)
        )
        self.update()

    def _apply_colormap(self, gray):
        import numpy as np
        v = gray.astype(np.float32) / 255.0
        h, w = v.shape
        r = np.zeros((h, w), dtype=np.uint8)
        g = np.zeros((h, w), dtype=np.uint8)
        b = np.zeros((h, w), dtype=np.uint8)

        m0 = v < 0.25
        m1 = (v >= 0.25) & (v < 0.5)
        m2 = (v >= 0.5) & (v < 0.75)
        m3 = v >= 0.75

        b[m0] = 255
        g[m0] = (v[m0] * 4 * 255).astype(np.uint8)

        g[m1] = 255
        b[m1] = (255 * (1 - (v[m1] - 0.25) * 4)).astype(np.uint8)

        r[m2] = (255 * (v[m2] - 0.5) * 4).astype(np.uint8)
        g[m2] = 255

        r[m3] = 255
        g[m3] = (255 * (1 - (v[m3] - 0.75) * 4)).astype(np.uint8)

        rgb = np.stack([r, g, b], axis=-1)
        return rgb

    @staticmethod
    def _turbo(val: int) -> tuple[int, int, int]:
        v = val / 255.0
        if v < 0.25:
            r = 0
            g = int(255 * (v * 4))
            b = 255
        elif v < 0.5:
            r = 0
            g = 255
            b = int(255 * (1 - (v - 0.25) * 4))
        elif v < 0.75:
            r = int(255 * ((v - 0.5) * 4))
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (v - 0.75) * 4))
            b = 0
        return r, g, b

    @staticmethod
    def _numpy_to_qimage(arr, w, h):
        from PyQt6.QtGui import QImage
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr[:, :, :3].copy()
            bytes_per_line = w * 3
            return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QImage(w, h, QImage.Format.Format_RGB888)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._image is not None:
            scaled = self._image.scaled(self.width(), self.height(),
                                        Qt.AspectRatioMode.IgnoreAspectRatio,
                                        Qt.TransformationMode.FastTransformation)
            painter.drawPixmap(0, 0, scaled)
        else:
            painter.fillRect(0, 0, self.width(), self.height(), QColor(0, 0, 0))
        painter.end()


class SpectrumControls(QWidget):
    """Control bar for spectrum display settings."""

    settings_changed = pyqtSignal()
    reset_peaks_clicked = pyqtSignal()
    rf_sweep_toggled = pyqtSignal(bool)
    rf_sweep_start = pyqtSignal()
    rf_sweep_stop = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._mode_btn = QPushButton("Audio FFT")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setFixedWidth(90)
        self._mode_btn.setStyleSheet(
            "QPushButton { border: 1px solid #555; border-radius: 3px; padding: 2px 6px; }"
            "QPushButton:checked { background-color: #2d7d2d; color: white; }"
        )
        self._mode_btn.toggled.connect(self._on_mode_toggle)
        layout.addWidget(self._mode_btn)

        layout.addWidget(self._vline())

        # Audio FFT controls
        self._audio_controls = QWidget()
        ac = QHBoxLayout(self._audio_controls)
        ac.setContentsMargins(0, 0, 0, 0)
        ac.setSpacing(4)
        ac.addWidget(QLabel("Span:"))
        self._span_combo = self._make_combo(["6k", "12k", "24k", "48k", "96k", "192k"])
        self._span_combo.setCurrentText("48k")
        ac.addWidget(self._span_combo)
        ac.addWidget(QLabel("FFT:"))
        self._fft_combo = self._make_combo(["512", "1024", "2048", "4096", "8192"])
        self._fft_combo.setCurrentText("2048")
        ac.addWidget(self._fft_combo)
        layout.addWidget(self._audio_controls)

        # RF Sweep controls
        self._rf_controls = QWidget()
        rf = QHBoxLayout(self._rf_controls)
        rf.setContentsMargins(0, 0, 0, 0)
        rf.setSpacing(4)
        rf.addWidget(QLabel("Start:"))
        self._sweep_start_mhz = self._make_spin(144.0, 140.0, 150.0, 3)
        self._sweep_start_mhz.setFixedWidth(75)
        rf.addWidget(self._sweep_start_mhz)
        rf.addWidget(QLabel("Stop:"))
        self._sweep_stop_mhz = self._make_spin(148.0, 140.0, 150.0, 3)
        self._sweep_stop_mhz.setFixedWidth(75)
        rf.addWidget(self._sweep_stop_mhz)
        rf.addWidget(QLabel("Step:"))
        self._sweep_step_khz = self._make_spin(25.0, 1.0, 500.0, 1)
        self._sweep_step_khz.setFixedWidth(60)
        rf.addWidget(self._sweep_step_khz)
        rf.addWidget(QLabel("kHz"))
        self._sweep_btn = QPushButton("Sweep")
        self._sweep_btn.setCheckable(True)
        self._sweep_btn.setFixedWidth(65)
        self._sweep_btn.clicked.connect(self._on_sweep_btn)
        rf.addWidget(self._sweep_btn)
        self._sweep_status = QLabel("")
        self._sweep_status.setStyleSheet("color: #aaa; font-size: 10px;")
        rf.addWidget(self._sweep_status)
        rf.addStretch()
        layout.addWidget(self._rf_controls)
        self._rf_controls.setVisible(False)

        layout.addWidget(self._vline())

        self._peak_cb = QCheckBox("Peak Hold")
        layout.addWidget(self._peak_cb)

        self._min_cb = QCheckBox("Min Hold")
        layout.addWidget(self._min_cb)

        self._peak_reset_btn = QPushButton("Reset Peaks")
        self._peak_reset_btn.clicked.connect(self.reset_peaks_clicked.emit)
        layout.addWidget(self._peak_reset_btn)

        self._range_label = QLabel("Range: -100 to 0 dB")
        layout.addWidget(self._range_label)

        layout.addStretch()

        self._span_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        self._fft_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        self._peak_cb.toggled.connect(lambda: self.settings_changed.emit())
        self._min_cb.toggled.connect(lambda: self.settings_changed.emit())

    @staticmethod
    def _vline():
        from PyQt6.QtWidgets import QFrame
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet("color: #444;")
        return f

    def _on_mode_toggle(self, checked: bool):
        self._audio_controls.setVisible(not checked)
        self._rf_controls.setVisible(checked)
        self.rf_sweep_toggled.emit(checked)

    def _on_sweep_btn(self):
        if self._sweep_btn.isChecked():
            self._sweep_btn.setText("Stop")
            self._sweep_btn.setStyleSheet(
                "QPushButton { background-color: #8b0000; color: white; font-weight: bold; }"
            )
            self.rf_sweep_start.emit()
        else:
            self._sweep_btn.setText("Sweep")
            self._sweep_btn.setStyleSheet("")
            self.rf_sweep_stop.emit()

    def set_sweep_status(self, text: str):
        self._sweep_status.setText(text)

    @staticmethod
    def _make_combo(items):
        from PyQt6.QtWidgets import QComboBox
        combo = QComboBox()
        combo.addItems(items)
        return combo

    @staticmethod
    def _make_spin(value, lo, hi, decimals):
        from PyQt6.QtWidgets import QDoubleSpinBox
        s = QDoubleSpinBox()
        s.setValue(value)
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(0.1 if decimals <= 2 else 1.0)
        return s

    @property
    def span_hz(self) -> float:
        text = self._span_combo.currentText().replace('k', '000')
        try:
            return float(text)
        except ValueError:
            return 48000.0

    @property
    def fft_size(self) -> int:
        try:
            return int(self._fft_combo.currentText())
        except ValueError:
            return 2048

    @property
    def sweep_start_mhz(self) -> float:
        return self._sweep_start_mhz.value()

    @property
    def sweep_stop_mhz(self) -> float:
        return self._sweep_stop_mhz.value()

    @property
    def sweep_step_khz(self) -> float:
        return self._sweep_step_khz.value()

    @property
    def show_peak_hold(self) -> bool:
        return self._peak_cb.isChecked()

    @property
    def show_min_hold(self) -> bool:
        return self._min_cb.isChecked()
