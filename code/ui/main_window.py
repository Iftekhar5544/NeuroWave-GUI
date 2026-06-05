from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from app_theme import THEME_COLORS, apply_dark_title_bar, themed_button_style, themed_label_style
from config import (
    APP_TITLE,
    DEFAULT_CHANNEL_COUNT,
    DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_REDRAW_INTERVAL_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_SECONDS,
    DISPLAY_AUTOSCALE_MARGIN,
    DISPLAY_AUTOSCALE_MIN_UV,
    DISPLAY_FILTER_PRESET,
    DISPLAY_FIXED_SCALE_UV,
    DISPLAY_USER_PRESETS_FILE,
    PROJECT_ROOT,
    STATUS_CONNECTED,
    STATUS_READY,
    STATUS_STREAMING,
)
from core.board_service import BoardService
from core.display_pipeline import ActiveFilterPlan, DisplayPipeline, FilterConfig
from core.preset_store import PresetStore
from core.ring_buffer import RingBuffer
from core.serial_ports import list_serial_ports
from core.stream_worker import StreamWorker
from ui.eeg_ml_window import EegMLWindow


class FilterWindow(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


class StyledArrowComboBox(QtWidgets.QComboBox):
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()

        separator_pen = QtGui.QPen(QtGui.QColor(THEME_COLORS["border"]))
        separator_pen.setWidth(1)
        painter.setPen(separator_pen)
        separator_x = rect.right() - 24
        painter.drawLine(separator_x, 3, separator_x, rect.height() - 4)

        color = QtGui.QColor(THEME_COLORS["text"] if self.isEnabled() else THEME_COLORS["disabled"])
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.NoPen)

        center_x = rect.right() - 11
        center_y = rect.center().y() + 1
        triangle = QtGui.QPolygon(
            [
                QtCore.QPoint(center_x - 5, center_y - 3),
                QtCore.QPoint(center_x + 5, center_y - 3),
                QtCore.QPoint(center_x, center_y + 4),
            ]
        )
        painter.drawPolygon(triangle)


class RoundedGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        path = QtGui.QPainterPath()
        rect = QtCore.QRectF(self.rect())
        path.addRoundedRect(rect, 10, 10)
        self.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))


class ConnectPortDialog(QtWidgets.QDialog):
    refresh_requested = QtCore.pyqtSignal()
    connect_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect Board")
        self.setModal(True)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowFlag(QtCore.Qt.CustomizeWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowTitleHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, False)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(460, 210)
        self.resize(520, 220)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Select Board Port")
        title.setStyleSheet("font-size: 18px; font-weight: 700; background: transparent;")
        subtitle = QtWidgets.QLabel("Choose the OpenBCI receiver port or simulator endpoint, then connect.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(title)
        layout.addWidget(subtitle)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        combo_wrap = QtWidgets.QFrame()
        combo_wrap.setObjectName("connectPortWrap")
        combo_wrap_layout = QtWidgets.QHBoxLayout(combo_wrap)
        combo_wrap_layout.setContentsMargins(0, 0, 0, 0)
        combo_wrap_layout.setSpacing(0)

        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(False)
        self.port_combo.setMinimumWidth(250)
        self.port_combo.setMinimumHeight(36)
        self.port_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.port_combo.setStyleSheet(
            "QComboBox {"
            " background: transparent;"
            f" color: {THEME_COLORS['text']};"
            " border: none;"
            " border-radius: 0px;"
            " padding: 3px 2px 3px 10px;"
            " min-height: 34px;"
            "}"
            "QComboBox::drop-down {"
            " width: 0px;"
            " border: none;"
            " background: transparent;"
            "}"
        )
        self.combo_arrow_button = QtWidgets.QToolButton()
        self.combo_arrow_button.setAutoRaise(False)
        self.combo_arrow_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.combo_arrow_button.setFixedSize(18, 36)
        self.combo_arrow_button.setArrowType(QtCore.Qt.DownArrow)
        combo_wrap_layout.addWidget(self.port_combo, 1)
        combo_wrap_layout.addWidget(self.combo_arrow_button, 0)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        row.addWidget(combo_wrap, 1)
        row.addWidget(self.refresh_button, 0)
        row.addWidget(self.connect_button, 0)
        layout.addLayout(row)

        self.message_label = QtWidgets.QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(self.message_label)

        self.refresh_button.setStyleSheet(themed_button_style("muted"))
        self.connect_button.setStyleSheet(themed_button_style("accent"))
        self.setStyleSheet(
            f"#connectPortWrap {{"
            f" background-color: {THEME_COLORS['input_bg']};"
            f" border: 1px solid {THEME_COLORS['border']};"
            " border-radius: 8px;"
            "}"
            "QComboBox {"
            " background: transparent;"
            f" color: {THEME_COLORS['text']};"
            " border: none;"
            " padding: 3px 8px 3px 10px;"
            " min-height: 34px;"
            "}"
            "QToolButton {"
            f" background-color: {THEME_COLORS['panel_alt']};"
            " border: none;"
            f" border-left: 1px solid {THEME_COLORS['border']};"
            " border-top-right-radius: 7px;"
            " border-bottom-right-radius: 7px;"
            "}"
            "QToolButton:hover {"
            f" background-color: {THEME_COLORS['panel']};"
            "}"
            "QToolButton:pressed {"
            f" background-color: {THEME_COLORS['bg']};"
            "}"
        )
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.connect_button.clicked.connect(self._emit_connect_requested)
        self.combo_arrow_button.clicked.connect(self.port_combo.showPopup)

    def _emit_connect_requested(self) -> None:
        self.connect_requested.emit(self.selected_port())

    def selected_port(self) -> str:
        return self.port_combo.currentText().strip()

    def set_ports(self, ports: list[str], current_text: str = "") -> None:
        current = current_text.strip() or self.selected_port()
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        options = list(ports)
        if current and current not in options:
            options.insert(0, current)
        self.port_combo.addItems(options)
        self.port_combo.blockSignals(False)
        if current:
            self.port_combo.setCurrentText(current)
        elif options:
            self.port_combo.setCurrentIndex(0)

    def show_error(self, message: str) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(themed_label_style("danger"))

    def show_info(self, message: str) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(themed_label_style("muted"))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.board_service = BoardService(logger=self.logger)
        self.stream_worker = None
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self.channel_count = DEFAULT_CHANNEL_COUNT
        self.ring_buffer = RingBuffer(self.channel_count, self.sample_rate * self.window_seconds)
        self.display_buffer = RingBuffer(self.channel_count, self.sample_rate * self.window_seconds)
        self.display_pipeline = DisplayPipeline(self.sample_rate, self.channel_count)
        self.active_filter_plan: ActiveFilterPlan = self.display_pipeline.active_plan
        self.preset_store = PresetStore(PROJECT_ROOT / DISPLAY_USER_PRESETS_FILE)
        self.curves = []
        self.channel_plots = []
        self.x_axis = np.linspace(-self.window_seconds, 0.0, self.sample_rate * self.window_seconds, dtype=np.float64)
        self.latest_display_window = np.full((self.channel_count, self.ring_buffer.capacity), np.nan, dtype=np.float64)
        self.latest_scales = self.display_pipeline.get_fixed_scales()
        self.display_dirty = False
        self.last_error_message = None
        self.buffering_dialog = None
        self.buffering_label = None
        self.buffering_progress = None
        self.available_ports: list[str] = []
        self.selected_port_text = ""
        self.connection_status_detail = ""
        self._updating_filter_ui = False
        self._status_base_message = STATUS_READY
        self._ui_fps = 0.0
        self._data_sps = 0.0
        self._ui_frame_counter = 0
        self._data_point_counter = 0
        self._ui_rate_t0 = time.perf_counter()
        self._data_rate_t0 = time.perf_counter()
        self.plotting_suspended = False

        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 920)
        self.setMinimumSize(1280, 760)
        self._build_ui()
        self._apply_visual_theme()
        apply_dark_title_bar(self)
        self._center_main_window()
        self._configure_plot()
        self._setup_timer()
        self._set_button_state(connected=False, streaming=False)
        self._load_initial_filter_state()
        self._set_status(STATUS_READY)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(8)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.start_button = QtWidgets.QPushButton("Start")
        self.filters_toggle_button = QtWidgets.QPushButton("Filter")
        self.filters_toggle_button.setCheckable(True)
        self.filters_toggle_button.setChecked(False)
        self.auto_scale_checkbox = QtWidgets.QCheckBox("Auto Scale")
        self.auto_scale_checkbox.setChecked(True)
        self.ml_data_button = QtWidgets.QPushButton("Data Collection")
        self.ml_train_button = QtWidgets.QPushButton("Train Model")
        self.ml_load_button = QtWidgets.QPushButton("Load Model")
        self.ml_realtime_button = QtWidgets.QPushButton("Realtime Prediction")
        self.status_label = QtWidgets.QLabel()
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.metrics_label = QtWidgets.QLabel("UI FPS: 0 | Data SPS: 0")
        self.metrics_label.setMinimumWidth(220)
        self.metrics_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        control_layout.addWidget(self.connect_button)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.filters_toggle_button)
        control_layout.addWidget(self.auto_scale_checkbox)
        control_layout.addStretch(1)
        control_layout.addWidget(self.ml_data_button)
        control_layout.addWidget(self.ml_train_button)
        control_layout.addWidget(self.ml_load_button)
        control_layout.addWidget(self.ml_realtime_button)

        status_layout = QtWidgets.QHBoxLayout()
        status_layout.setSpacing(8)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.metrics_label, 0, QtCore.Qt.AlignRight)

        control_frame = QtWidgets.QFrame()
        control_frame.setProperty("card", True)
        control_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        control_frame_layout = QtWidgets.QHBoxLayout(control_frame)
        control_frame_layout.setContentsMargins(12, 10, 12, 10)
        control_frame_layout.setSpacing(0)
        control_frame_layout.addLayout(control_layout)

        status_frame = QtWidgets.QFrame()
        status_frame.setProperty("card", True)
        status_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        status_frame_layout = QtWidgets.QHBoxLayout(status_frame)
        status_frame_layout.setContentsMargins(12, 8, 12, 8)
        status_frame_layout.setSpacing(0)
        status_frame_layout.addLayout(status_layout)

        self.plot_widget = RoundedGraphicsLayoutWidget()
        self.plot_widget.setBackground(THEME_COLORS["graph_bg"])
        root_layout.addWidget(control_frame, 0)
        root_layout.addWidget(status_frame, 0)
        root_layout.addWidget(self.plot_widget, 1)

        self._build_filter_sidebar()
        self._build_connect_dialog()
        self._build_ml_window()

        self.connect_button.clicked.connect(self.handle_connect_toggle)
        self.start_button.clicked.connect(self.handle_start_stop_toggle)
        self.auto_scale_checkbox.toggled.connect(self.handle_auto_scale_toggled)
        self.ml_data_button.clicked.connect(self._open_data_collection_window)
        self.ml_train_button.clicked.connect(self._open_train_model_window)
        self.ml_load_button.clicked.connect(self._open_load_model_window)
        self.ml_realtime_button.clicked.connect(self._open_realtime_prediction_window)
        self.filters_toggle_button.toggled.connect(self._toggle_filter_window)
        self.filter_window.closed.connect(lambda: self.filters_toggle_button.setChecked(False))
        self.refresh_ports()

    def _apply_visual_theme(self) -> None:
        for button in [
            self.connect_button,
            self.start_button,
            self.filters_toggle_button,
            self.ml_data_button,
            self.ml_train_button,
            self.ml_load_button,
            self.ml_realtime_button,
        ]:
            button.setStyleSheet(themed_button_style("accent"))
        self.connect_button.setMinimumWidth(118)
        self.start_button.setMinimumWidth(92)
        self.filters_toggle_button.setMinimumWidth(88)
        self.ml_data_button.setMinimumWidth(128)
        self.ml_train_button.setMinimumWidth(112)
        self.ml_load_button.setMinimumWidth(108)
        self.ml_realtime_button.setMinimumWidth(152)
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 700; background: transparent;")
        self.metrics_label.setStyleSheet(themed_label_style("muted"))
        self.auto_scale_checkbox.setStyleSheet(
            f"QCheckBox {{ font-weight: 700; color: {THEME_COLORS['text']}; background: transparent; }}"
            f"QCheckBox::indicator {{ width: 16px; height: 16px; }}"
        )

    def _build_connect_dialog(self) -> None:
        self.connect_dialog = ConnectPortDialog(self)
        apply_dark_title_bar(self.connect_dialog)
        self.connect_dialog.refresh_requested.connect(self._refresh_connect_dialog_ports)
        self.connect_dialog.connect_requested.connect(self._handle_connect_dialog_submit)

    def _refresh_connect_dialog_ports(self) -> None:
        self.refresh_ports(update_status=False)
        info_text = f"Found {len(self.available_ports)} available endpoint(s)." if self.available_ports else "No ports found."
        self.connect_dialog.show_info(info_text)

    def _open_connect_dialog(self) -> None:
        self.refresh_ports(update_status=False)
        current_text = self.selected_port_text or (self.available_ports[0] if self.available_ports else "")
        self.connect_dialog.set_ports(self.available_ports, current_text=current_text)
        info_text = f"Found {len(self.available_ports)} available endpoint(s)." if self.available_ports else "No ports found."
        self.connect_dialog.show_info(info_text)
        self._position_window_centered_on_main(self.connect_dialog)
        self.connect_dialog.port_combo.setFocus()
        self.connect_dialog.exec_()

    def _handle_connect_dialog_submit(self, port: str) -> None:
        port = (port or "").strip()
        if not port:
            self.connect_dialog.show_error("Select a COM port or simulator endpoint first.")
            return
        try:
            self._connect_to_port(port)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Connect failed.")
            self.connect_dialog.show_error(str(exc))
            self._show_error(str(exc))
            self._set_button_state(connected=False, streaming=False)
            return
        self.selected_port_text = port
        self.connect_dialog.accept()

    def _build_filter_sidebar(self) -> None:
        self.filter_window = FilterWindow(self)
        self.filter_window.setWindowTitle("Display Filter- Only For Graph Plots")
        self.filter_window.setWindowFlag(QtCore.Qt.Window, True)
        self.filter_window.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)
        self.filter_window.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.filter_window.setModal(False)
        self.filter_window.resize(920, 430)
        self.filter_window.setObjectName("filterWindow")
        self.filter_window.setStyleSheet(
            "QDialog#filterWindow {"
            f" background-color: {THEME_COLORS['bg']};"
            " border: none;"
            "}"
            "QScrollArea {"
            " border: none;"
            " background: transparent;"
            "}"
            "QWidget#filterPanel {"
            f" background-color: {THEME_COLORS['bg']};"
            " border: none;"
            "}"
        )
        apply_dark_title_bar(self.filter_window)
        panel = QtWidgets.QWidget(self.filter_window)
        panel.setObjectName("filterPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.preset_combo = StyledArrowComboBox()
        self.scope_combo = StyledArrowComboBox()
        self.scope_combo.addItems(["All Channels", "Selected Channels"])
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_list.setMinimumHeight(120)
        self.channel_list.setMaximumHeight(180)

        self.general_group = QtWidgets.QGroupBox("General")
        self.general_group.setMinimumWidth(280)
        self.general_group.setMaximumWidth(360)
        general_form = QtWidgets.QFormLayout(self.general_group)
        general_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        general_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        general_form.setHorizontalSpacing(10)
        general_label_preset = QtWidgets.QLabel("Preset")
        general_label_scope = QtWidgets.QLabel("Apply Scope")
        general_label_channels = QtWidgets.QLabel("Channels")
        for label in [general_label_preset, general_label_scope, general_label_channels]:
            label.setMinimumWidth(92)
        general_form.addRow(general_label_preset, self.preset_combo)
        general_form.addRow(general_label_scope, self.scope_combo)
        general_form.addRow(general_label_channels, self.channel_list)

        self.notch_enable = QtWidgets.QCheckBox("Enable")
        self.notch_freq_combo = StyledArrowComboBox()
        self.notch_freq_combo.addItems(["50", "60"])
        self.notch_q_spin = QtWidgets.QDoubleSpinBox()
        self.notch_q_spin.setDecimals(1)
        self.notch_q_spin.setRange(0.1, 200.0)
        self.notch_q_spin.setValue(30.0)
        self.notch_q_spin.setSingleStep(0.5)
        self.notch_group = self._create_filter_group(
            "Notch",
            [
                ("Enable", self._wrap_checkbox(self.notch_enable)),
                ("Frequency", self.notch_freq_combo),
                ("Q", self.notch_q_spin),
            ],
        )

        self.hp_enable = QtWidgets.QCheckBox("Enable")
        self.hp_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.hp_cutoff_spin.setDecimals(2)
        self.hp_cutoff_spin.setRange(0.01, 200.0)
        self.hp_cutoff_spin.setValue(1.0)
        self.hp_cutoff_spin.setSuffix(" Hz")
        self.hp_order_combo = StyledArrowComboBox()
        self.hp_order_combo.addItems(["2", "4"])
        self.hp_group = self._create_filter_group(
            "High-pass",
            [
                ("Enable", self._wrap_checkbox(self.hp_enable)),
                ("Cutoff", self.hp_cutoff_spin),
                ("Order", self.hp_order_combo),
            ],
        )

        self.lp_enable = QtWidgets.QCheckBox("Enable")
        self.lp_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.lp_cutoff_spin.setDecimals(2)
        self.lp_cutoff_spin.setRange(0.01, 200.0)
        self.lp_cutoff_spin.setValue(40.0)
        self.lp_cutoff_spin.setSuffix(" Hz")
        self.lp_order_combo = StyledArrowComboBox()
        self.lp_order_combo.addItems(["2", "4"])
        self.lp_group = self._create_filter_group(
            "Low-pass",
            [
                ("Enable", self._wrap_checkbox(self.lp_enable)),
                ("Cutoff", self.lp_cutoff_spin),
                ("Order", self.lp_order_combo),
            ],
        )

        self.bp_enable = QtWidgets.QCheckBox("Enable")
        self.bp_low_spin = QtWidgets.QDoubleSpinBox()
        self.bp_low_spin.setDecimals(2)
        self.bp_low_spin.setRange(0.01, 200.0)
        self.bp_low_spin.setValue(8.0)
        self.bp_low_spin.setSuffix(" Hz")
        self.bp_high_spin = QtWidgets.QDoubleSpinBox()
        self.bp_high_spin.setDecimals(2)
        self.bp_high_spin.setRange(0.01, 200.0)
        self.bp_high_spin.setValue(13.0)
        self.bp_high_spin.setSuffix(" Hz")
        self.bp_order_combo = StyledArrowComboBox()
        self.bp_order_combo.addItems(["2", "4"])
        self.bp_group = self._create_filter_group(
            "Band-pass",
            [
                ("Enable", self._wrap_checkbox(self.bp_enable)),
                ("Low", self.bp_low_spin),
                ("High", self.bp_high_spin),
                ("Order", self.bp_order_combo),
            ],
        )

        filters_grid = QtWidgets.QGridLayout()
        filters_grid.setHorizontalSpacing(12)
        filters_grid.setVerticalSpacing(12)
        filters_grid.addWidget(self.notch_group, 0, 0)
        filters_grid.addWidget(self.hp_group, 0, 1)
        filters_grid.addWidget(self.lp_group, 1, 0)
        filters_grid.addWidget(self.bp_group, 1, 1)

        self.filters_group = QtWidgets.QGroupBox("Filter Chain")
        filters_group_layout = QtWidgets.QVBoxLayout(self.filters_group)
        filters_group_layout.setContentsMargins(10, 10, 10, 10)
        filters_group_layout.addLayout(filters_grid)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self.general_group, 0)
        top_row.addWidget(self.filters_group, 1)
        layout.addLayout(top_row)

        self.filter_error_label = QtWidgets.QLabel("")
        self.filter_error_label.setWordWrap(False)
        self.filter_error_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.filter_error_label.setStyleSheet(themed_label_style("danger"))
        self.filter_error_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.custom_hint_label = QtWidgets.QLabel(
            "[HP + Notch + LP] for broad EEG display  |  [BP + Notch] for band-focused analysis."
        )
        self.custom_hint_label.setWordWrap(False)
        self.custom_hint_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.custom_hint_label.setStyleSheet(themed_label_style("muted"))
        self.custom_hint_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        feedback_row = QtWidgets.QHBoxLayout()
        feedback_row.setSpacing(10)
        feedback_row.addWidget(self.filter_error_label, 1)
        feedback_row.addWidget(self.custom_hint_label, 0, QtCore.Qt.AlignRight)
        layout.addLayout(feedback_row)

        button_row = QtWidgets.QHBoxLayout()
        self.apply_filters_button = QtWidgets.QPushButton("Apply")
        self.cancel_filters_button = QtWidgets.QPushButton("Cancel")
        self.reset_filters_button = QtWidgets.QPushButton("Reset")
        self.save_preset_button = QtWidgets.QPushButton("Save Preset")
        self.delete_preset_button = QtWidgets.QPushButton("Delete Preset")
        self.apply_filters_button.setStyleSheet(themed_button_style("success"))
        self.cancel_filters_button.setStyleSheet(themed_button_style("muted"))
        self.reset_filters_button.setStyleSheet(themed_button_style("muted"))
        self.save_preset_button.setStyleSheet(themed_button_style("accent"))
        self.delete_preset_button.setStyleSheet(themed_button_style("danger"))
        button_row.addWidget(self.apply_filters_button)
        button_row.addWidget(self.cancel_filters_button)
        button_row.addWidget(self.reset_filters_button)
        button_row.addStretch(1)
        button_row.addWidget(self.save_preset_button)
        button_row.addWidget(self.delete_preset_button)
        layout.addLayout(button_row)
        layout.addStretch(1)

        scroll = QtWidgets.QScrollArea(self.filter_window)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        root = QtWidgets.QVBoxLayout(self.filter_window)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)
        panel.adjustSize()
        panel_hint = panel.minimumSizeHint()
        min_width = max(920, int(panel_hint.width() + 24))
        min_height = max(430, int(panel_hint.height() + 24))
        self.filter_window.setMinimumSize(min_width, min_height)

        combo_style = (
            "QComboBox {"
            f" background-color: {THEME_COLORS['input_bg']};"
            f" color: {THEME_COLORS['text']};"
            f" border: 1px solid {THEME_COLORS['border']};"
            " border-radius: 8px;"
            " padding: 3px 22px 3px 8px;"
            " min-height: 32px;"
            "}"
            "QComboBox::drop-down {"
            " subcontrol-origin: padding;"
            " subcontrol-position: top right;"
            " width: 18px;"
            " border: none;"
            " background: transparent;"
            "}"
        )
        for combo in [
            self.preset_combo,
            self.scope_combo,
            self.notch_freq_combo,
            self.hp_order_combo,
            self.lp_order_combo,
            self.bp_order_combo,
        ]:
            combo.setStyleSheet(combo_style)

        self.preset_combo.currentTextChanged.connect(self.handle_preset_changed)
        self.scope_combo.currentTextChanged.connect(self.handle_scope_changed)
        self.notch_enable.toggled.connect(self._mark_custom_from_edit)
        self.notch_freq_combo.currentTextChanged.connect(self._mark_custom_from_edit)
        self.notch_q_spin.valueChanged.connect(self._mark_custom_from_edit)
        self.hp_enable.toggled.connect(self._mark_custom_from_edit)
        self.hp_cutoff_spin.valueChanged.connect(self._mark_custom_from_edit)
        self.hp_order_combo.currentTextChanged.connect(self._mark_custom_from_edit)
        self.lp_enable.toggled.connect(self._mark_custom_from_edit)
        self.lp_cutoff_spin.valueChanged.connect(self._mark_custom_from_edit)
        self.lp_order_combo.currentTextChanged.connect(self._mark_custom_from_edit)
        self.bp_enable.toggled.connect(self._mark_custom_from_edit)
        self.bp_low_spin.valueChanged.connect(self._mark_custom_from_edit)
        self.bp_high_spin.valueChanged.connect(self._mark_custom_from_edit)
        self.bp_order_combo.currentTextChanged.connect(self._mark_custom_from_edit)
        self.channel_list.itemSelectionChanged.connect(self._mark_custom_from_edit)
        self.apply_filters_button.clicked.connect(self.handle_apply_filters)
        self.cancel_filters_button.clicked.connect(self.handle_cancel_filters)
        self.reset_filters_button.clicked.connect(self.handle_reset_filters)
        self.save_preset_button.clicked.connect(self.handle_save_preset)
        self.delete_preset_button.clicked.connect(self.handle_delete_preset)

    def _toggle_filter_window(self, visible: bool) -> None:
        if visible:
            self._clear_filter_error()
            active_cfg = self.active_filter_plan.config
            self._select_preset_name(active_cfg.preset_name)
            self._fill_filter_controls_from_config(active_cfg)
            self._position_filter_window()
            self.filter_window.show()
            self.filter_window.raise_()
            self.filter_window.activateWindow()
        else:
            self.filter_window.hide()

    def _position_filter_window(self) -> None:
        self._position_window_centered_on_main(self.filter_window)

    def _build_ml_window(self) -> None:
        self.eeg_ml_collect_window = EegMLWindow(logger=self.logger, parent=self, mode="data_collection")
        self.eeg_ml_train_window = EegMLWindow(logger=self.logger, parent=self, mode="train_model")
        self.eeg_ml_load_window = EegMLWindow(logger=self.logger, parent=self, mode="load_model")
        for window in self._ml_windows():
            window.hide()
            window.set_stream_context(self.sample_rate, self.channel_count, self.board_service.connected)
            window.set_streaming(self.board_service.streaming)
            window.recording_activity_changed.connect(self._set_plotting_suspended)

    def _ml_windows(self) -> list[EegMLWindow]:
        return [self.eeg_ml_collect_window, self.eeg_ml_train_window, self.eeg_ml_load_window]

    def _open_ml_window(self, window: EegMLWindow) -> None:
        self._position_window_centered_on_main(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _open_data_collection_window(self) -> None:
        self._open_ml_window(self.eeg_ml_collect_window)

    def _open_train_model_window(self) -> None:
        self._open_ml_window(self.eeg_ml_train_window)

    def _open_load_model_window(self) -> None:
        self._open_ml_window(self.eeg_ml_load_window)

    def _open_realtime_prediction_window(self) -> None:
        self._open_ml_window(self.eeg_ml_load_window)

    def _center_main_window(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        center = screen.availableGeometry().center()
        frame.moveCenter(center)
        self.move(frame.topLeft())

    def _position_window_centered_on_main(self, window: QtWidgets.QWidget) -> None:
        frame = self.frameGeometry()
        center = frame.center()
        target = QtCore.QPoint(
            int(center.x() - (window.width() / 2)),
            int(center.y() - (window.height() / 2)),
        )
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            x = min(max(target.x(), avail.left()), avail.right() - window.width())
            y = min(max(target.y(), avail.top()), avail.bottom() - window.height())
            target = QtCore.QPoint(x, y)
        window.move(target)

    @staticmethod
    def _create_filter_group(title: str, rows: list[tuple[str, QtWidgets.QWidget]]) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for i, (label, widget) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), i, 0)
            grid.addWidget(widget, i, 1)
        return group

    @staticmethod
    def _wrap_checkbox(checkbox: QtWidgets.QCheckBox) -> QtWidgets.QWidget:
        checkbox.setText("")
        holder = QtWidgets.QWidget()
        holder.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        holder.setStyleSheet("background: transparent; border: none;")
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(checkbox, 0, QtCore.Qt.AlignLeft)
        layout.addStretch(1)
        return holder

    def _configure_plot(self) -> None:
        pg.setConfigOptions(antialias=False, useOpenGL=False)
        self._rebuild_curves()

    def _rebuild_curves(self) -> None:
        self.plot_widget.clear()
        self.curves = []
        self.channel_plots = []
        colors = ["#58C7A4", "#FF6D76", "#7FD9FF", "#F7C85A", "#B2F0D4", "#D496FF", "#FF8E8E", "#68D8E8", "#F3D17A"]
        axis_pen = pg.mkPen(THEME_COLORS["graph_axis"])
        axis_font = QtGui.QFont()
        axis_font.setPointSize(9)
        axis_font.setBold(True)

        x_min = float(self.x_axis[0])
        x_max = float(self.x_axis[-1])
        x_span = x_max - x_min
        for channel_index in range(self.channel_count):
            channel_color = colors[channel_index % len(colors)]
            badge_proxy = self._create_channel_badge(channel_index + 1, channel_color)
            self.plot_widget.addItem(badge_proxy, row=channel_index, col=0)

            plot = self.plot_widget.addPlot(row=channel_index, col=1)
            plot.getViewBox().setDefaultPadding(0.0)
            plot.setMenuEnabled(False)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            plot.showGrid(x=False, y=True, alpha=0.24)
            left_axis = plot.getAxis("left")
            left_axis.setPen(axis_pen)
            left_axis.setStyle(showValues=False, autoExpandTextSpace=False, tickTextWidth=0, tickFont=axis_font)
            left_axis.setWidth(6)
            bottom_axis = plot.getAxis("bottom")
            bottom_axis.setPen(axis_pen)
            bottom_axis.setStyle(tickFont=axis_font)
            left_axis.setTextPen(pg.mkColor(THEME_COLORS["graph_axis"]))
            bottom_axis.setTextPen(pg.mkColor(THEME_COLORS["graph_axis"]))
            if hasattr(left_axis, "label") and left_axis.label is not None:
                left_axis.label.hide()
            plot.disableAutoRange(axis="x")
            plot.disableAutoRange(axis="y")
            plot.setLimits(xMin=x_min, xMax=x_max, minXRange=x_span, maxXRange=x_span)
            plot.setXRange(x_min, x_max, padding=0)
            plot.setYRange(-DISPLAY_FIXED_SCALE_UV, DISPLAY_FIXED_SCALE_UV, padding=0)
            plot.hideAxis("bottom")
            plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen(THEME_COLORS["graph_zero"], width=1)))

            curve = plot.plot(pen=pg.mkPen(channel_color, width=2))
            self.channel_plots.append(plot)
            self.curves.append(curve)

        axis_spacer = QtWidgets.QWidget()
        axis_spacer.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        axis_spacer.setFixedSize(34, 28)
        axis_spacer_proxy = QtWidgets.QGraphicsProxyWidget()
        axis_spacer_proxy.setWidget(axis_spacer)
        self.plot_widget.addItem(axis_spacer_proxy, row=self.channel_count, col=0)

        bottom_plot = self.plot_widget.addPlot(row=self.channel_count, col=1)
        bottom_plot.getViewBox().setDefaultPadding(0.0)
        bottom_plot.setMenuEnabled(False)
        bottom_plot.setMouseEnabled(x=False, y=False)
        bottom_plot.hideButtons()
        bottom_plot.hideAxis("left")
        bottom_plot.setYRange(0.0, 1.0, padding=0)
        bottom_plot.disableAutoRange(axis="x")
        bottom_plot.disableAutoRange(axis="y")
        bottom_plot.setLimits(xMin=x_min, xMax=x_max, minXRange=x_span, maxXRange=x_span)
        bottom_plot.setXRange(x_min, x_max, padding=0)
        bottom_axis = bottom_plot.getAxis("bottom")
        bottom_axis.setPen(axis_pen)
        bottom_axis.setTextPen(pg.mkColor(THEME_COLORS["graph_axis"]))
        bottom_axis.setStyle(showValues=False, tickFont=axis_font)
        bottom_plot.setLabel("bottom", "Time (s)", color=THEME_COLORS["graph_axis"], size="8pt")
        if hasattr(bottom_plot, "setMaximumHeight"):
            bottom_plot.setMaximumHeight(36)
        if hasattr(bottom_plot, "setMinimumHeight"):
            bottom_plot.setMinimumHeight(36)

    @staticmethod
    def _create_channel_badge(channel_number: int, channel_color: str) -> QtWidgets.QGraphicsProxyWidget:
        wrapper = QtWidgets.QWidget()
        wrapper.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(4, 0, 2, 0)
        wrapper_layout.setSpacing(0)

        badge = QtWidgets.QLabel(str(channel_number))
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setFixedSize(28, 28)
        badge.setStyleSheet(
            "QLabel {"
            f" color: {channel_color};"
            f" border: 2px solid {channel_color};"
            " border-radius: 14px;"
            " background: transparent;"
            " font-size: 8pt;"
            " font-weight: 700;"
            "}"
        )
        wrapper_layout.addWidget(badge, 0, QtCore.Qt.AlignCenter)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(wrapper)
        return proxy

    def _setup_timer(self) -> None:
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setInterval(DEFAULT_REDRAW_INTERVAL_MS)
        self.redraw_timer.timeout.connect(self.redraw_plot)
        self.redraw_timer.start()
        self.metrics_timer = QtCore.QTimer(self)
        self.metrics_timer.setInterval(500)
        self.metrics_timer.timeout.connect(self._refresh_runtime_metrics)
        self.metrics_timer.start()

    def _refresh_runtime_metrics(self) -> None:
        now = time.perf_counter()

        ui_elapsed = now - self._ui_rate_t0
        if ui_elapsed >= 0.5:
            self._ui_fps = float(self._ui_frame_counter) / ui_elapsed
            self._ui_frame_counter = 0
            self._ui_rate_t0 = now

        data_elapsed = now - self._data_rate_t0
        if data_elapsed >= 0.5:
            avg_points_per_channel = float(self._data_point_counter) / max(1, int(self.channel_count))
            self._data_sps = avg_points_per_channel / data_elapsed
            self._data_point_counter = 0
            self._data_rate_t0 = now

        if not self.board_service.streaming:
            self._data_sps = 0.0
            self._data_point_counter = 0
            self._data_rate_t0 = now

        self._render_status_label()

    def _reset_runtime_metrics(self) -> None:
        now = time.perf_counter()
        self._ui_fps = 0.0
        self._data_sps = 0.0
        self._ui_frame_counter = 0
        self._data_point_counter = 0
        self._ui_rate_t0 = now
        self._data_rate_t0 = now

    def _format_runtime_metrics(self) -> str:
        return f"UI FPS: {int(round(self._ui_fps))} | Data SPS: {int(round(self._data_sps))}"

    def _render_status_label(self) -> None:
        status_message = self._status_base_message
        if not status_message.startswith("Error:") and self.active_filter_plan and "Filters:" not in status_message:
            preset_name = (self.active_filter_plan.config.preset_name or "Custom").strip() or "Custom"
            status_message = f"{status_message} | Filter: {preset_name} | {self.active_filter_plan.summary}"
        self.status_label.setText(f"Status: {status_message}")
        self.metrics_label.setText(self._format_runtime_metrics())
        if status_message.startswith("Error:"):
            self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {THEME_COLORS['danger']}; background: transparent;")
        elif "Streaming" in status_message:
            self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {THEME_COLORS['success']}; background: transparent;")
        else:
            self.status_label.setStyleSheet("font-size: 16px; font-weight: 700; background: transparent;")

    def _create_worker(self) -> StreamWorker:
        chunk_size = max(1, int(self.sample_rate * DEFAULT_POLL_INTERVAL_MS / 1000.0 * 3))
        worker = StreamWorker(self.board_service, poll_interval_ms=DEFAULT_POLL_INTERVAL_MS, chunk_size=chunk_size, logger=self.logger)
        worker.data_ready.connect(self.handle_data_ready)
        worker.error.connect(self.handle_worker_error)
        worker.state_changed.connect(self.handle_worker_state_changed)
        return worker

    def _load_initial_filter_state(self) -> None:
        self._update_channel_selector_items()
        self._populate_preset_combo()
        self._select_preset_name(DISPLAY_FILTER_PRESET)
        preset_cfg = self.preset_store.get(self.preset_combo.currentText()) or FilterConfig(preset_name="Custom")
        self._fill_filter_controls_from_config(preset_cfg)
        self._sync_filter_control_enabled()
        self.handle_apply_filters()

    def _populate_preset_combo(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(self.preset_store.list_preset_names())
        self.preset_combo.blockSignals(False)
        self._update_delete_button_state()

    def _select_preset_name(self, name: str) -> None:
        requested = (name or "").strip()
        if requested == "EEG Default":
            requested = "EEG Lab"
        index = self.preset_combo.findText(requested)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        else:
            custom_index = self.preset_combo.findText("Custom")
            if custom_index >= 0:
                self.preset_combo.setCurrentIndex(custom_index)

    def _update_channel_selector_items(self) -> None:
        selected = set(self._selected_channel_indexes())
        self.channel_list.clear()
        for i in range(self.channel_count):
            item = QtWidgets.QListWidgetItem(f"CH {i + 1}")
            self.channel_list.addItem(item)
            if i in selected:
                item.setSelected(True)

    def _selected_channel_indexes(self) -> list[int]:
        return sorted(item.row() for item in self.channel_list.selectedIndexes())

    def _set_status(self, message: str) -> None:
        self._status_base_message = message
        self._render_status_label()
        if not message.startswith("Error:"):
            self.last_error_message = None

    def _set_button_state(self, connected: bool, streaming: bool) -> None:
        self.connect_button.setEnabled(True)
        self.start_button.setEnabled(connected)
        self._update_connect_button_appearance(connected)
        self._update_start_button_appearance(connected=connected, streaming=streaming)

    def _update_connect_button_appearance(self, connected: bool) -> None:
        if connected:
            self.connect_button.setText("Disconnect")
            self.connect_button.setStyleSheet(themed_button_style("danger"))
        else:
            self.connect_button.setText("Connect")
            self.connect_button.setStyleSheet(themed_button_style("accent"))
        self.connect_button.setMinimumWidth(118)

    def _update_start_button_appearance(self, connected: bool, streaming: bool) -> None:
        if not connected:
            self.start_button.setText("Start")
            self.start_button.setStyleSheet(themed_button_style("accent"))
            self.start_button.setMinimumWidth(92)
            return
        if streaming:
            self.start_button.setText("Stop")
            self.start_button.setStyleSheet(themed_button_style("danger"))
            self.start_button.setMinimumWidth(92)
            return
        self.start_button.setText("Start")
        self.start_button.setStyleSheet(themed_button_style("accent"))
        self.start_button.setMinimumWidth(92)

    def handle_connect_toggle(self) -> None:
        if self.board_service.connected:
            self.handle_disconnect()
            return
        self._open_connect_dialog()

    def handle_start_stop_toggle(self) -> None:
        if self.board_service.streaming:
            self.handle_stop()
            return
        self.handle_start()

    def _current_base_status(self) -> str:
        if self.board_service.streaming:
            return STATUS_STREAMING
        if self.board_service.connected:
            return self._connected_status_message()
        return STATUS_READY

    def _connected_status_message(self) -> str:
        if self.connection_status_detail:
            return f"{STATUS_CONNECTED} {self.connection_status_detail}"
        return STATUS_CONNECTED

    def _connect_to_port(self, port: str) -> None:
        self.board_service.connect(port)
        self.channel_count = len(self.board_service.get_eeg_channels())
        self.sample_rate = self.board_service.get_sample_rate()
        connection_name = self.board_service.get_connection_name()
        self.selected_port_text = connection_name
        self.connection_status_detail = f"[{connection_name}] ({self.channel_count} ch @ {self.sample_rate} Hz)"
        self._reset_buffer()
        self._set_button_state(connected=True, streaming=False)
        self._set_status(self._connected_status_message())
        for window in self._ml_windows():
            window.set_stream_context(self.sample_rate, self.channel_count, connected=True)
            window.set_streaming(False)

    def handle_connect(self) -> None:
        self._open_connect_dialog()

    def _reset_buffer(self) -> None:
        capacity = self.sample_rate * self.window_seconds
        self.ring_buffer = RingBuffer(self.channel_count, capacity)
        self.display_buffer = RingBuffer(self.channel_count, capacity)
        self.x_axis = np.linspace(-self.window_seconds, 0.0, capacity, dtype=np.float64)
        self.latest_display_window = np.full((self.channel_count, capacity), np.nan, dtype=np.float64)
        self.latest_scales = self.display_pipeline.get_fixed_scales()
        self.display_dirty = False
        self._update_channel_selector_items()
        self._rebuild_curves()
        self._rebuild_pipeline_for_current_ui_config()

    def _rebuild_pipeline_for_current_ui_config(self) -> None:
        self.display_pipeline.reset(self.sample_rate, self.channel_count)
        config = self._collect_filter_config_from_ui()
        try:
            plan = DisplayPipeline.compile_plan(config, self.sample_rate, self.channel_count)
        except ValueError as exc:
            self._show_filter_error(str(exc))
            return
        self._clear_filter_error()
        self.active_filter_plan = plan
        self.display_pipeline.set_active_plan(plan)
        self._reprocess_display_buffer()

    def refresh_ports(self, update_status: bool = True) -> None:
        current_text = self.selected_port_text.strip()
        try:
            ports = list_serial_ports()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Port refresh failed.")
            self.available_ports = []
            if hasattr(self, "connect_dialog"):
                self.connect_dialog.set_ports([], current_text=current_text)
            self._show_error(str(exc))
            return

        self.available_ports = list(ports)
        if not current_text and self.available_ports:
            current_text = self.available_ports[0]
        if hasattr(self, "connect_dialog"):
            self.connect_dialog.set_ports(self.available_ports, current_text=current_text)

        if update_status and not self.last_error_message:
            if ports:
                self._set_status(f"{STATUS_READY} | Found {len(ports)} COM port(s)")
            else:
                self._set_status(f"{STATUS_READY} | No COM ports found")

    def handle_start(self) -> None:
        try:
            self._reset_runtime_metrics()
            self.board_service.start_stream()
            self.stream_worker = self._create_worker()
            self.stream_worker.start()
            self._set_button_state(connected=True, streaming=True)
            self._set_status(STATUS_STREAMING)
            for window in self._ml_windows():
                window.set_stream_context(self.sample_rate, self.channel_count, connected=True)
                window.set_streaming(True)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Start failed.")
            try:
                self._stop_worker()
                self.board_service.stop_stream()
                for window in self._ml_windows():
                    window.set_streaming(False)
            except Exception:  # pylint: disable=broad-except
                self.logger.exception("Cleanup after failed start did not complete cleanly.")
            self._show_error(str(exc))
            self._set_button_state(connected=self.board_service.connected, streaming=False)

    def handle_stop(self) -> None:
        had_error = False
        try:
            self._stop_worker()
            self.board_service.stop_stream()
            self._reset_runtime_metrics()
            self._hide_buffering_dialog()
            for window in self._ml_windows():
                window.set_streaming(False)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Stop failed.")
            self._show_error(str(exc))
            had_error = True
        finally:
            if had_error:
                self._set_button_state(connected=self.board_service.connected, streaming=False)
            elif self.board_service.connected:
                self._set_button_state(connected=True, streaming=False)
                self._set_status(STATUS_CONNECTED)
            else:
                self._set_button_state(connected=False, streaming=False)
                self._set_status(STATUS_READY)

    def handle_disconnect(self) -> None:
        had_error = False
        try:
            self._stop_worker()
            self.board_service.disconnect()
            self._reset_runtime_metrics()
            self._hide_buffering_dialog()
            self.connection_status_detail = ""
            for window in self._ml_windows():
                window.set_stream_context(self.sample_rate, self.channel_count, connected=False)
                window.set_streaming(False)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Disconnect failed.")
            self._show_error(str(exc))
            had_error = True
        finally:
            self._set_button_state(connected=False, streaming=False)
            if not had_error:
                self._set_status(STATUS_READY)

    def handle_data_ready(self, payload) -> None:
        try:
            if isinstance(payload, dict):
                chunk = payload.get("eeg")
                timestamps = payload.get("timestamps")
                sample_index = payload.get("sample_index")
            else:
                chunk = payload
                timestamps = None
                sample_index = None

            if chunk is None:
                return
            self._data_point_counter += int(chunk.size)
            for window in self._ml_windows():
                window.handle_incoming_chunk(chunk, timestamps=timestamps, sample_index=sample_index)
            if not self.plotting_suspended:
                self.ring_buffer.append(chunk)
                processed_chunk = self.display_pipeline.process_chunk(chunk)
                self.display_buffer.append(processed_chunk)
                if self.auto_scale_checkbox.isChecked():
                    self.latest_scales = self.display_pipeline.update_auto_scales_from_chunk(processed_chunk)
                else:
                    self.latest_scales = self.display_pipeline.get_fixed_scales()
                self.display_dirty = True
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to append incoming EEG chunk.")
            self._show_error(str(exc))

    def handle_worker_error(self, message: str) -> None:
        self._show_error(message)
        self._stop_worker()
        try:
            self.board_service.stop_stream()
            self._hide_buffering_dialog()
            for window in self._ml_windows():
                window.set_streaming(False)
        except Exception:  # pylint: disable=broad-except
            self.logger.exception("Failed to stop stream after worker error.")
        self._set_button_state(connected=self.board_service.connected, streaming=False)

    def handle_worker_state_changed(self, state: str) -> None:
        if state == "stopped" and self.board_service.connected and not self.board_service.streaming and not self.last_error_message:
            self._set_status(self._connected_status_message())

    def redraw_plot(self) -> None:
        self._ui_frame_counter += 1
        if self.plotting_suspended:
            return
        filled_count = self.ring_buffer.get_filled_count()
        if filled_count <= 0:
            return
        if filled_count < self.ring_buffer.capacity:
            self._show_buffering_dialog(filled_count, self.ring_buffer.capacity, self.sample_rate)
            blank = np.full(self.ring_buffer.capacity, np.nan, dtype=np.float64)
            for curve in self.curves:
                curve.setData(self.x_axis, blank)
            if self.board_service.streaming:
                buffered_seconds = filled_count / float(self.sample_rate)
                self._set_status(f"{STATUS_STREAMING} | Buffering {buffered_seconds:.1f}/{self.window_seconds:.1f}s")
            return

        self._hide_buffering_dialog()
        if self.board_service.streaming and "Buffering" in self.status_label.text():
            self._set_status(STATUS_STREAMING)

        if self.display_dirty:
            self.latest_display_window = self.display_buffer.get_window()
            self.display_dirty = False

        for index, curve in enumerate(self.curves):
            curve.setData(self.x_axis, self.latest_display_window[index])
            if self.auto_scale_checkbox.isChecked():
                scale = max(DISPLAY_AUTOSCALE_MIN_UV, float(self.latest_scales[index]))
                self.channel_plots[index].setYRange(-scale, scale, padding=0)

    def _set_plotting_suspended(self, suspended: bool) -> None:
        suspended = bool(suspended)
        if self.plotting_suspended == suspended:
            return
        self.plotting_suspended = suspended
        if suspended:
            self.ring_buffer.clear()
            self.display_buffer.clear()
            self.display_dirty = False
            blank = np.full(self.ring_buffer.capacity, np.nan, dtype=np.float64)
            for curve in self.curves:
                curve.setData(self.x_axis, blank)
            if self.board_service.streaming:
                self._set_status(f"{STATUS_STREAMING} | Plot paused for recording")
        else:
            self.display_pipeline.reset(self.sample_rate, self.channel_count)
            self.latest_scales = self.display_pipeline.get_fixed_scales()
            if self.board_service.streaming:
                self._set_status(STATUS_STREAMING)

    def handle_auto_scale_toggled(self, enabled: bool) -> None:
        if not enabled:
            for plot in self.channel_plots:
                plot.disableAutoRange(axis="y")
                plot.setYRange(-DISPLAY_FIXED_SCALE_UV, DISPLAY_FIXED_SCALE_UV, padding=0)
            self.latest_scales = self.display_pipeline.get_fixed_scales()
            self._set_status(f"{self._current_base_status()} | Auto Scale OFF")
            return

        for plot in self.channel_plots:
            plot.disableAutoRange(axis="y")
        if np.isfinite(self.latest_display_window).any():
            peaks = np.nanmax(np.abs(self.latest_display_window), axis=1)
            self.latest_scales = np.maximum(DISPLAY_AUTOSCALE_MIN_UV, peaks * DISPLAY_AUTOSCALE_MARGIN)
        self.display_dirty = True
        self._set_status(self._current_base_status())

    def handle_preset_changed(self, preset_name: str) -> None:
        if self._updating_filter_ui:
            return
        cfg = self.preset_store.get(preset_name)
        if cfg is None:
            return
        if preset_name in {"OpenBCI Default", "EEG Lab", "EEG Default"}:
            cfg.notch_freq = int(self.notch_freq_combo.currentText() or "50")
        self._fill_filter_controls_from_config(cfg)
        self._update_delete_button_state()

    def handle_scope_changed(self, _scope_text: str) -> None:
        is_selected_scope = self.scope_combo.currentText() == "Selected Channels"
        self.channel_list.setEnabled(is_selected_scope)
        self._set_filter_section_faded(self.channel_list, not is_selected_scope)
        self._mark_custom_from_edit()

    def _mark_custom_from_edit(self, *_args) -> None:
        if self._updating_filter_ui:
            return
        self._sync_filter_control_enabled()
        current = self.preset_combo.currentText()
        if current != "Custom":
            index = self.preset_combo.findText("Custom")
            if index >= 0:
                self._updating_filter_ui = True
                self.preset_combo.setCurrentIndex(index)
                self._updating_filter_ui = False
        self._update_delete_button_state()

    def _sync_filter_control_enabled(self) -> None:
        preset_name = (self.preset_combo.currentText() or "").strip()
        is_custom_preset = preset_name == "Custom"
        is_selected_scope = self.scope_combo.currentText() == "Selected Channels"

        self.custom_hint_label.setVisible(is_custom_preset)
        self.scope_combo.setEnabled(is_custom_preset)
        self.channel_list.setEnabled(is_custom_preset and is_selected_scope)
        self.notch_group.setEnabled(is_custom_preset)
        self.hp_group.setEnabled(is_custom_preset)
        self.lp_group.setEnabled(is_custom_preset)
        self.bp_group.setEnabled(is_custom_preset)
        self._set_filter_section_faded(self.scope_combo, not is_custom_preset)
        self._set_filter_section_faded(self.channel_list, (not is_custom_preset) or (not is_selected_scope))
        self._set_filter_section_faded(self.notch_group, not is_custom_preset)
        self._set_filter_section_faded(self.hp_group, not is_custom_preset)
        self._set_filter_section_faded(self.lp_group, not is_custom_preset)
        self._set_filter_section_faded(self.bp_group, not is_custom_preset)

        if not is_custom_preset:
            return

        self.notch_freq_combo.setEnabled(self.notch_enable.isChecked())
        self.notch_q_spin.setEnabled(self.notch_enable.isChecked())
        self.hp_cutoff_spin.setEnabled(self.hp_enable.isChecked())
        self.hp_order_combo.setEnabled(self.hp_enable.isChecked())
        self.lp_cutoff_spin.setEnabled(self.lp_enable.isChecked())
        self.lp_order_combo.setEnabled(self.lp_enable.isChecked())
        self.bp_low_spin.setEnabled(self.bp_enable.isChecked())
        self.bp_high_spin.setEnabled(self.bp_enable.isChecked())
        self.bp_order_combo.setEnabled(self.bp_enable.isChecked())

    @staticmethod
    def _set_filter_section_faded(widget: QtWidgets.QWidget, faded: bool) -> None:
        if widget is None:
            return
        opacity = 0.45 if faded else 1.0
        effect = widget.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        effect.setOpacity(opacity)

    def _fill_filter_controls_from_config(self, cfg: FilterConfig) -> None:
        self._updating_filter_ui = True
        self.scope_combo.setCurrentText("Selected Channels" if cfg.scope == "selected" else "All Channels")
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setSelected(i in set(cfg.selected_channels))
        self.notch_enable.setChecked(bool(cfg.notch_enabled))
        self.notch_freq_combo.setCurrentText(str(int(cfg.notch_freq)))
        self.notch_q_spin.setValue(float(cfg.notch_q))
        self.hp_enable.setChecked(bool(cfg.highpass_enabled))
        self.hp_cutoff_spin.setValue(float(cfg.highpass_hz))
        self.hp_order_combo.setCurrentText(str(int(cfg.highpass_order)))
        self.lp_enable.setChecked(bool(cfg.lowpass_enabled))
        self.lp_cutoff_spin.setValue(float(cfg.lowpass_hz))
        self.lp_order_combo.setCurrentText(str(int(cfg.lowpass_order)))
        self.bp_enable.setChecked(bool(cfg.bandpass_enabled))
        self.bp_low_spin.setValue(float(cfg.bandpass_low_hz))
        self.bp_high_spin.setValue(float(cfg.bandpass_high_hz))
        self.bp_order_combo.setCurrentText(str(int(cfg.bandpass_order)))
        self._updating_filter_ui = False
        self._sync_filter_control_enabled()

    def _collect_filter_config_from_ui(self) -> FilterConfig:
        return FilterConfig(
            enabled=True,
            preset_name=self.preset_combo.currentText().strip() or "Custom",
            scope="selected" if self.scope_combo.currentText() == "Selected Channels" else "all",
            selected_channels=self._selected_channel_indexes(),
            notch_enabled=self.notch_enable.isChecked(),
            notch_freq=int(self.notch_freq_combo.currentText() or "50"),
            notch_q=float(self.notch_q_spin.value()),
            highpass_enabled=self.hp_enable.isChecked(),
            highpass_hz=float(self.hp_cutoff_spin.value()),
            highpass_order=int(self.hp_order_combo.currentText() or "2"),
            lowpass_enabled=self.lp_enable.isChecked(),
            lowpass_hz=float(self.lp_cutoff_spin.value()),
            lowpass_order=int(self.lp_order_combo.currentText() or "2"),
            bandpass_enabled=self.bp_enable.isChecked(),
            bandpass_low_hz=float(self.bp_low_spin.value()),
            bandpass_high_hz=float(self.bp_high_spin.value()),
            bandpass_order=int(self.bp_order_combo.currentText() or "2"),
        )

    def handle_apply_filters(self) -> None:
        cfg = self._collect_filter_config_from_ui()
        try:
            plan = DisplayPipeline.compile_plan(cfg, self.sample_rate, self.channel_count)
        except ValueError as exc:
            self._show_filter_error(str(exc))
            return

        self._clear_filter_error()
        self.active_filter_plan = plan
        self.display_pipeline.set_active_plan(plan)
        self._reprocess_display_buffer()
        self._set_status(f"{self._current_base_status()} | Filter Applied")

    def handle_cancel_filters(self) -> None:
        active_cfg = self.active_filter_plan.config
        self._select_preset_name(active_cfg.preset_name)
        self._fill_filter_controls_from_config(active_cfg)
        self._clear_filter_error()
        self.filters_toggle_button.setChecked(False)

    def handle_reset_filters(self) -> None:
        cfg = self.preset_store.get("Raw") or FilterConfig(preset_name="Raw")
        cfg.notch_freq = int(self.notch_freq_combo.currentText() or "50")
        self._select_preset_name(cfg.preset_name)
        self._fill_filter_controls_from_config(cfg)
        self.handle_apply_filters()

    def handle_save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok:
            return
        try:
            cfg = self._collect_filter_config_from_ui()
            self.preset_store.save_user_preset(name, cfg)
            self._populate_preset_combo()
            self._select_preset_name(name.strip())
            self._set_status(f"{self._current_base_status()} | Preset saved")
        except Exception as exc:  # pylint: disable=broad-except
            self._show_filter_error(str(exc))

    def handle_delete_preset(self) -> None:
        name = self.preset_combo.currentText().strip()
        if self.preset_store.is_builtin(name):
            self._show_filter_error("Built-in presets cannot be deleted.")
            return
        try:
            self.preset_store.delete_user_preset(name)
            self._populate_preset_combo()
            self._select_preset_name("Custom")
            self._set_status(f"{self._current_base_status()} | Preset deleted")
        except Exception as exc:  # pylint: disable=broad-except
            self._show_filter_error(str(exc))

    def _update_delete_button_state(self) -> None:
        name = self.preset_combo.currentText().strip()
        self.delete_preset_button.setEnabled(bool(name) and not self.preset_store.is_builtin(name))

    def _show_filter_error(self, message: str) -> None:
        clean_message = (message or "").strip()
        lower = clean_message.lower()
        if "select at least one channel" in lower:
            clean_message = "Select at least one Channel"
        self.filter_error_label.setText(clean_message)
        self._set_status(f"Error: {clean_message.splitlines()[0]}")

    def _clear_filter_error(self) -> None:
        self.filter_error_label.setText("")

    def _reprocess_display_buffer(self) -> None:
        raw_window = self.ring_buffer.get_window()
        filled_count = self.ring_buffer.get_filled_count()
        capacity = self.ring_buffer.capacity
        rebuilt_display = RingBuffer(self.channel_count, capacity)
        self.display_pipeline.set_active_plan(self.active_filter_plan)

        if self.auto_scale_checkbox.isChecked():
            self.latest_scales = self.display_pipeline.get_fixed_scales()

        if filled_count > 0:
            valid_window = raw_window[:, -filled_count:]
            chunk_step = max(1, int(self.sample_rate // 2))
            for start in range(0, valid_window.shape[1], chunk_step):
                raw_chunk = valid_window[:, start : start + chunk_step]
                if raw_chunk.shape[1] == 0:
                    continue
                processed_chunk = self.display_pipeline.process_chunk(raw_chunk)
                rebuilt_display.append(processed_chunk)
                if self.auto_scale_checkbox.isChecked():
                    self.latest_scales = self.display_pipeline.update_auto_scales_from_chunk(processed_chunk)

        self.display_buffer = rebuilt_display
        if not self.auto_scale_checkbox.isChecked():
            self.latest_scales = self.display_pipeline.get_fixed_scales()
        self.latest_display_window = self.display_buffer.get_window()
        self.display_dirty = False

    def _stop_worker(self) -> None:
        if self.stream_worker is None:
            return
        if self.stream_worker.isRunning():
            self.stream_worker.stop()
            self.stream_worker.wait(1500)
        self.stream_worker.deleteLater()
        self.stream_worker = None

    def _show_error(self, message: str) -> None:
        self.last_error_message = message
        self._status_base_message = f"Error: {message}"
        self._render_status_label()
        self._hide_buffering_dialog()

    def _show_buffering_dialog(self, filled_count: int, capacity: int, sample_rate: int) -> None:
        if self.buffering_dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Buffering")
            dialog.setModal(False)
            dialog.setWindowFlag(QtCore.Qt.CustomizeWindowHint, True)
            dialog.setWindowFlag(QtCore.Qt.WindowTitleHint, True)
            dialog.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
            dialog.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
            dialog.setFixedSize(320, 120)
            apply_dark_title_bar(dialog)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)
            self.buffering_label = QtWidgets.QLabel("Preparing live window...")
            self.buffering_label.setStyleSheet("font-size: 16px; font-weight: 700; background: transparent;")
            self.buffering_progress = QtWidgets.QProgressBar()
            self.buffering_progress.setRange(0, 100)
            layout.addWidget(self.buffering_label)
            layout.addWidget(self.buffering_progress)
            self.buffering_dialog = dialog

        if capacity <= 0 or sample_rate <= 0:
            return
        elapsed_s = filled_count / float(sample_rate)
        total_s = capacity / float(sample_rate)
        percent = int(max(0, min(100, round((filled_count / float(capacity)) * 100.0))))
        self.buffering_label.setText(f"Buffering {elapsed_s:.1f}s / {total_s:.1f}s")
        self.buffering_progress.setValue(percent)
        if not self.buffering_dialog.isVisible():
            self._position_window_centered_on_main(self.buffering_dialog)
            self.buffering_dialog.show()

    def _hide_buffering_dialog(self) -> None:
        if self.buffering_dialog is not None:
            self.buffering_dialog.hide()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:  # noqa: N802
        super().moveEvent(event)
        if hasattr(self, "filter_window") and self.filter_window.isVisible():
            self._position_filter_window()
        if hasattr(self, "connect_dialog") and self.connect_dialog.isVisible():
            self._position_window_centered_on_main(self.connect_dialog)
        if hasattr(self, "eeg_ml_collect_window"):
            for window in self._ml_windows():
                if window.isVisible():
                    self._position_window_centered_on_main(window)
        if self.buffering_dialog is not None and self.buffering_dialog.isVisible():
            self._position_window_centered_on_main(self.buffering_dialog)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "filter_window") and self.filter_window.isVisible():
            self._position_filter_window()
        if hasattr(self, "connect_dialog") and self.connect_dialog.isVisible():
            self._position_window_centered_on_main(self.connect_dialog)
        if hasattr(self, "eeg_ml_collect_window"):
            for window in self._ml_windows():
                if window.isVisible():
                    self._position_window_centered_on_main(window)
        if self.buffering_dialog is not None and self.buffering_dialog.isVisible():
            self._position_window_centered_on_main(self.buffering_dialog)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            self._stop_worker()
            self.board_service.disconnect()
            self._hide_buffering_dialog()
            if hasattr(self, "filter_window"):
                self.filter_window.close()
            if hasattr(self, "eeg_ml_collect_window"):
                for window in self._ml_windows():
                    window.shutdown()
                    window.hide()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Shutdown cleanup failed.")
            self.last_error_message = str(exc)
        event.accept()
