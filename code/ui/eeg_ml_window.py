from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app_theme import THEME_COLORS, apply_dark_title_bar, themed_button_style, themed_label_style
from config import PROJECT_ROOT
from core.eeg_inference_worker import EEGInferenceWorker
from core.eeg_ml import (
    DEFAULT_ML_DATA_DIR,
    DEFAULT_ML_MODEL_ARTIFACT,
    DEFAULT_ML_MODEL_DIR,
    DEFAULT_ML_RUN_NAME,
    LabeledEegRecorder,
    load_model_bundle,
    train_eeg_model,
)
from core.ring_buffer import RingBuffer


class TrainModelWorker(QtCore.QThread):
    success = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, kwargs: dict) -> None:
        super().__init__()
        self.kwargs = dict(kwargs)

    def run(self) -> None:
        try:
            result = train_eeg_model(**self.kwargs)
            self.success.emit(result)
        except Exception as exc:  # pylint: disable=broad-except
            self.failed.emit(str(exc))


class ClassLabelEditorDialog(QtWidgets.QDialog):
    def __init__(
        self,
        labels: list[str] | None = None,
        label_image_paths: dict[str, str] | None = None,
        image_dir: Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Class Label Editor")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.resize(760, 460)
        self.setMinimumSize(680, 380)
        apply_dark_title_bar(self)

        self.saved_labels: list[str] = []
        self.saved_label_image_paths: dict[str, str] = {}
        self.image_dir = Path(image_dir) if image_dir is not None else (PROJECT_ROOT / "code" / "images" / "class_label_image")
        self.label_image_paths = {str(k): str(v) for k, v in dict(label_image_paths or {}).items() if str(v).strip()}
        self.rows: list[dict[str, object]] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Class Labels")
        title.setStyleSheet("font-size: 18px; font-weight: 700; background: transparent;")
        subtitle = QtWidgets.QLabel("Add the labels you want to record. Optional images are shown as cues during recording.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(themed_label_style("muted"))
        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        rows_host = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(rows_host)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch(1)
        scroll.setWidget(rows_host)
        layout.addWidget(scroll, 1)

        seed_labels = [str(x).strip() for x in (labels or []) if str(x).strip()]
        if not seed_labels:
            seed_labels = ["Left", "Right", "Forward"]
        for text in seed_labels:
            self._add_row(text, self.label_image_paths.get(text, ""))

        row_actions = QtWidgets.QHBoxLayout()
        row_actions.setSpacing(8)
        self.btn_add = QtWidgets.QPushButton("Add Class Label")
        self.btn_add.setStyleSheet(themed_button_style("accent"))
        self.btn_add.clicked.connect(lambda: self._add_row(""))
        self.btn_save = QtWidgets.QPushButton("Save Class Label")
        self.btn_save.setStyleSheet(themed_button_style("success"))
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(themed_button_style("muted"))
        self.btn_cancel.clicked.connect(self.reject)
        row_actions.addWidget(self.btn_add)
        row_actions.addStretch(1)
        row_actions.addWidget(self.btn_cancel)
        row_actions.addWidget(self.btn_save)
        layout.addLayout(row_actions)

    def _refresh_titles(self) -> None:
        for idx, row in enumerate(self.rows):
            row_widget = row["widget"]
            if not isinstance(row_widget, QtWidgets.QWidget):
                continue
            row_layout = row_widget.layout()
            lbl = row_layout.itemAt(0).widget()
            if isinstance(lbl, QtWidgets.QLabel):
                lbl.setText(f"Class {idx + 1}:")

    def _add_row(self, text: str, image_path: str = "") -> None:
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        lbl = QtWidgets.QLabel("Class:")
        row_layout.addWidget(lbl)
        edit = QtWidgets.QLineEdit(str(text or ""))
        row_layout.addWidget(edit, 1)
        image_status = QtWidgets.QLabel("")
        image_status.setMinimumWidth(120)
        image_status.setStyleSheet(themed_label_style("muted"))
        btn_image = QtWidgets.QPushButton("Add Image")
        btn_image.setStyleSheet(themed_button_style("accent"))
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.setStyleSheet(themed_button_style("muted"))
        row_state: dict[str, object] = {
            "widget": row_widget,
            "edit": edit,
            "image_status": image_status,
            "image_path": str(image_path or ""),
        }
        btn_image.clicked.connect(lambda _checked=False, state=row_state: self._add_image_for_row(state))
        btn_delete.clicked.connect(lambda _checked=False, w=row_widget: self._delete_row(w))
        row_layout.addWidget(image_status)
        row_layout.addWidget(btn_image)
        row_layout.addWidget(btn_delete)
        self.rows_layout.insertWidget(max(0, self.rows_layout.count() - 1), row_widget)
        self.rows.append(row_state)
        self._refresh_image_status(row_state)
        self._refresh_titles()

    def _delete_row(self, row_widget: QtWidgets.QWidget) -> None:
        if len(self.rows) <= 1:
            return
        idx = -1
        for i, row in enumerate(self.rows):
            widget = row.get("widget")
            if widget is row_widget:
                idx = i
                break
        if idx < 0:
            return
        row = self.rows.pop(idx)
        widget = row.get("widget")
        if not isinstance(widget, QtWidgets.QWidget):
            return
        self.rows_layout.removeWidget(widget)
        widget.deleteLater()
        self._refresh_titles()

    def _refresh_image_status(self, row: dict[str, object]) -> None:
        status = row.get("image_status")
        if not isinstance(status, QtWidgets.QLabel):
            return
        image_path = str(row.get("image_path", "") or "").strip()
        if image_path:
            status.setText(Path(image_path).name)
            status.setToolTip(image_path)
            status.setStyleSheet(themed_label_style("success"))
        else:
            status.setText("No image")
            status.setToolTip("")
            status.setStyleSheet(themed_label_style("muted"))

    def _copy_label_image(self, source_path: str, label: str) -> str:
        src = Path(source_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label or "class").strip()).strip("_") or "class"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = src.suffix.lower() or ".png"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        dst = self.image_dir / f"{safe_label}_{stamp}{suffix}"
        counter = 2
        while dst.exists():
            dst = self.image_dir / f"{safe_label}_{stamp}_{counter:02d}{suffix}"
            counter += 1
        shutil.copy2(src, dst)
        return str(dst)

    def _add_image_for_row(self, row: dict[str, object]) -> None:
        edit = row.get("edit")
        label = edit.text().strip() if isinstance(edit, QtWidgets.QLineEdit) else ""
        if not label:
            QtWidgets.QMessageBox.warning(self, "Class Image", "Enter the class label before adding an image.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Add Class Image",
            str(PROJECT_ROOT),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
        )
        if not path:
            return
        try:
            copied = self._copy_label_image(path, label)
            row["image_path"] = copied
            self._refresh_image_status(row)
        except Exception as exc:  # pylint: disable=broad-except
            QtWidgets.QMessageBox.warning(self, "Class Image", f"Failed to add image:\n{exc}")

    def _save(self) -> None:
        labels = []
        image_paths: dict[str, str] = {}
        for row in self.rows:
            edit = row.get("edit")
            if not isinstance(edit, QtWidgets.QLineEdit):
                continue
            text = edit.text().strip()
            if text:
                labels.append(text)
                image_path = str(row.get("image_path", "") or "").strip()
                if image_path:
                    image_paths[text] = image_path
        if not labels:
            QtWidgets.QMessageBox.warning(self, "Class Labels", "Provide at least one class label.")
            return
        self.saved_labels = labels
        self.saved_label_image_paths = image_paths
        self.accept()


class InfoTextDialog(QtWidgets.QDialog):
    def __init__(self, title: str, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setModal(True)
        self.resize(620, 420)
        self.setMinimumSize(520, 340)
        apply_dark_title_bar(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: 700; background: transparent;")
        body = QtWidgets.QPlainTextEdit()
        body.setReadOnly(True)
        body.setPlainText(text)
        body.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {THEME_COLORS['input_bg']}; "
            f"border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; padding: 8px; }}"
        )
        close_button = QtWidgets.QPushButton("Close")
        close_button.setStyleSheet(themed_button_style("accent"))
        close_button.clicked.connect(self.accept)

        layout.addWidget(title_label)
        layout.addWidget(body, 1)
        layout.addWidget(close_button, 0, QtCore.Qt.AlignRight)


class RecordingSessionWindow(QtWidgets.QDialog):
    start_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    toggle_pause_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Recording Session")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)
        self.setModal(False)
        self.resize(1240, 760)
        self.setMinimumSize(980, 620)
        apply_dark_title_bar(self)
        self._activity_pixmap: QtGui.QPixmap | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(18)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(12)
        self.start_button = QtWidgets.QPushButton("Start Recording")
        self.start_button.setStyleSheet(themed_button_style("accent"))
        self.start_button.clicked.connect(self.start_requested.emit)
        self.start_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.fullscreen_button = QtWidgets.QPushButton("Full Screen")
        self.fullscreen_button.setStyleSheet(themed_button_style("accent"))
        self.fullscreen_button.clicked.connect(self.toggle_full_screen)
        self.fullscreen_button.setFocusPolicy(QtCore.Qt.NoFocus)
        button_row.addWidget(self.start_button, 0)
        button_row.addWidget(self.fullscreen_button, 0)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.activity_area = QtWidgets.QWidget()
        self.activity_layout = QtWidgets.QVBoxLayout(self.activity_area)
        self.activity_layout.setContentsMargins(0, 0, 0, 0)
        self.activity_layout.setSpacing(16)

        self.phase_label = QtWidgets.QLabel("Get Ready")
        self.phase_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phase_label.setStyleSheet("font-size: 104px; font-weight: 800; background: transparent;")
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setVisible(False)
        self.image_label.setStyleSheet("background: transparent;")

        self.activity_layout.addStretch(1)
        self.activity_layout.addWidget(self.phase_label, 0)
        self.activity_layout.addWidget(self.image_label, 1)
        self.activity_layout.addStretch(2)
        layout.addWidget(self.activity_area, 1)

        self.instruction_label = QtWidgets.QLabel("Start this session by clicking Start Recording")
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 24px; background: transparent;")
        layout.addWidget(self.instruction_label, 0)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(28)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress, 0)

        footer_row = QtWidgets.QHBoxLayout()
        self.footer_left = QtWidgets.QLabel("Follow The Instructions")
        self.footer_left.setStyleSheet("font-size: 18px; font-weight: 700; background: transparent;")
        self.footer_right = QtWidgets.QLabel("Double Click for Full screen | ESC to Cancel Session | Spacebar to Pause/Resume")
        self.footer_right.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.footer_right.setStyleSheet(themed_label_style("muted"))
        footer_row.addWidget(self.footer_left, 1)
        footer_row.addWidget(self.footer_right, 1)
        layout.addLayout(footer_row)

        self.space_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self.space_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.space_shortcut.activated.connect(self.toggle_pause_requested.emit)

    def set_can_start(self, enabled: bool) -> None:
        self.start_button.setEnabled(bool(enabled))

    def set_start_text(self, text: str) -> None:
        self.start_button.setText(str(text))

    def set_phase(self, text: str) -> None:
        clean = str(text).strip()
        if clean.startswith("{") and clean.endswith("}") and len(clean) >= 2:
            clean = clean[1:-1].strip()
        self.phase_label.setText(clean or "Get Ready")

    def set_activity_image(self, image_path: str | None) -> None:
        path_text = str(image_path or "").strip()
        if not path_text:
            self._activity_pixmap = None
            self.image_label.clear()
            self.image_label.setVisible(False)
            self.activity_layout.setStretch(0, 1)
            self.activity_layout.setStretch(1, 0)
            self.activity_layout.setStretch(2, 0)
            self.activity_layout.setStretch(3, 2)
            return
        pixmap = QtGui.QPixmap(path_text)
        if pixmap.isNull():
            self._activity_pixmap = None
            self.image_label.clear()
            self.image_label.setVisible(False)
            self.activity_layout.setStretch(0, 1)
            self.activity_layout.setStretch(3, 2)
            return
        self._activity_pixmap = pixmap
        self.image_label.setVisible(True)
        self.activity_layout.setStretch(0, 1)
        self.activity_layout.setStretch(1, 0)
        self.activity_layout.setStretch(2, 5)
        self.activity_layout.setStretch(3, 1)
        self._render_activity_image()

    def _render_activity_image(self) -> None:
        if self._activity_pixmap is None or self._activity_pixmap.isNull() or not self.image_label.isVisible():
            return
        target = self.image_label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._activity_pixmap.scaled(
            target,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def set_instruction(self, text: str) -> None:
        self.instruction_label.setText(str(text))

    def set_progress_value(self, value: int) -> None:
        self.progress.setValue(max(0, min(100, int(value))))

    def set_progress_format(self, text: str) -> None:
        self.progress.setFormat(str(text))

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._sync_fullscreen_button_text()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self.toggle_full_screen()
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key_Escape:
            self.cancel_requested.emit()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Space:
            self.toggle_pause_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange and self.windowState() & QtCore.Qt.WindowMaximized:
            QtCore.QTimer.singleShot(0, self.showFullScreen)
        if event.type() == QtCore.QEvent.WindowStateChange:
            QtCore.QTimer.singleShot(0, self._sync_fullscreen_button_text)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_activity_image()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.cancel_requested.emit()
        super().closeEvent(event)

    def _sync_fullscreen_button_text(self) -> None:
        self.fullscreen_button.setText("Exit Full Screen" if self.isFullScreen() else "Full Screen")


class EegMLWindow(QtWidgets.QDialog):
    closed = QtCore.pyqtSignal()
    recording_activity_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        logger: logging.Logger,
        parent: QtWidgets.QWidget | None = None,
        mode: str = "all",
    ) -> None:
        super().__init__(parent)
        self.logger = logger
        self.mode = str(mode or "all").strip().lower()
        self.sample_rate = 125
        self.channel_count = 16
        self.connected = False
        self.streaming = False
        self.recorder: LabeledEegRecorder | None = None
        self.model_bundle = None
        self.prediction_enabled = False
        self.train_worker: TrainModelWorker | None = None
        self.inference_worker = EEGInferenceWorker()
        self.inference_worker.prediction_ready.connect(self._on_prediction_ready)
        self.inference_worker.error.connect(self._on_prediction_error)
        self.inference_worker.start()
        self.infer_ring = RingBuffer(self.channel_count, self.sample_rate * 10)
        self._samples_since_submit = 0
        self.protocol_running = False
        self.protocol_labels: list[str] = []
        self.protocol_repeat_total = 0
        self.protocol_repeat_index = 0
        self.protocol_label_index = 0
        self.protocol_phase = "idle"
        self.protocol_phase_end_ts = 0.0
        self.protocol_base_trial = ""
        self.protocol_timer = QtCore.QTimer(self)
        self.protocol_timer.setInterval(200)
        self.protocol_timer.timeout.connect(self._on_protocol_tick)
        self.record_rest = False
        self.session_prepared = False
        self.session_finished_waiting_for_close = False
        self.labels_saved = False
        self.labels_confirmed = False
        self.protocol_confirmed = False
        self.terms_viewed = False
        self.prestart_running = False
        self.prestart_end_ts = 0.0
        self.prestart_duration_s = 5.0
        self.session_paused = False
        self.paused_mode = ""
        self.paused_remaining_s = 0.0
        self.folder_name_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_session_window: RecordingSessionWindow | None = None
        self.prestart_timer = QtCore.QTimer(self)
        self.prestart_timer.setInterval(100)
        self.prestart_timer.timeout.connect(self._on_prestart_tick)

        self.collection_session_dir: Path | None = None
        self.collection_data_path: Path | None = None
        self.collection_metadata_path: Path | None = None
        self.collection_bundle_name = ""
        self.collection_rows_total = 0
        self.collection_labels: set[str] = set()
        self.collection_started_at = ""
        self.label_image_paths: dict[str, str] = {}

        self.setWindowTitle("EEG ML Pipeline")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setWindowFlag(QtCore.Qt.CustomizeWindowHint, True)
        self.setWindowFlag(QtCore.Qt.WindowTitleHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, self.mode != "data_collection")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.setModal(False)
        self.resize(1160, 760)
        self.setMinimumSize(980, 680)
        self._build_ui()
        self._apply_visual_theme()
        self._apply_mode_layout()
        apply_dark_title_bar(self)
        self._update_ui_state()

    def _build_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        self.root_layout = root_layout
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        self.title_label = QtWidgets.QLabel("EEG ML Pipeline")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: 700; background: transparent;")
        root_layout.addWidget(self.title_label)

        self.subtitle_label = QtWidgets.QLabel("Collect raw EEG, train a model, and run live inference from dedicated workflow windows.")
        self.subtitle_label.setWordWrap(True)
        root_layout.addWidget(self.subtitle_label)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        root_layout.addWidget(self.status_label)

        self.top_panel = QtWidgets.QWidget()
        self.top_grid = QtWidgets.QGridLayout(self.top_panel)
        self.top_grid.setContentsMargins(0, 0, 0, 0)
        self.top_grid.setHorizontalSpacing(14)
        self.top_grid.setVerticalSpacing(14)
        self.top_grid.setColumnStretch(0, 1)
        self.top_grid.setColumnStretch(1, 1)

        self.collect_group = QtWidgets.QGroupBox("1) Data Collection (RAW EEG)")
        self.collect_form = QtWidgets.QFormLayout(self.collect_group)
        collect_form = self.collect_form
        self.collect_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.collect_form.setHorizontalSpacing(10)
        self.collect_form.setVerticalSpacing(8)

        default_dataset_root = PROJECT_ROOT / DEFAULT_ML_DATA_DIR
        self.collect_root_edit = QtWidgets.QLineEdit(str(default_dataset_root))
        browse_collect = QtWidgets.QPushButton("Browse")
        browse_collect.clicked.connect(self._browse_collect_root)
        collect_path_row = QtWidgets.QHBoxLayout()
        collect_path_row.addWidget(self.collect_root_edit, 1)
        collect_path_row.addWidget(browse_collect)
        collect_path_box = QtWidgets.QWidget()
        collect_path_box.setLayout(collect_path_row)
        collect_form.addRow("Dataset Root", collect_path_box)

        self.contributor_edit = QtWidgets.QLineEdit("")
        self.contributor_edit.setPlaceholderText("Enter contributor name")
        collect_form.addRow("Contributor", self.contributor_edit)

        consent_row = QtWidgets.QWidget()
        consent_layout = QtWidgets.QHBoxLayout(consent_row)
        consent_layout.setContentsMargins(0, 0, 0, 0)
        consent_layout.setSpacing(8)
        self.agree_yes = QtWidgets.QCheckBox("Yes")
        self.agree_no = QtWidgets.QCheckBox("No")
        self.agree_no.setChecked(True)
        self.terms_button = QtWidgets.QPushButton("Read T&C")
        self.terms_button.clicked.connect(self._show_terms)
        consent_layout.addWidget(QtWidgets.QLabel("Agree to contribute?"))
        consent_layout.addWidget(self.agree_yes)
        consent_layout.addWidget(self.agree_no)
        consent_layout.addStretch(1)
        consent_layout.addWidget(self.terms_button)
        collect_form.addRow("Consent", consent_row)

        self.session_info_label = QtWidgets.QLabel("Session: not started")
        self.session_info_label.setWordWrap(True)
        collect_form.addRow("Session", self.session_info_label)

        labels_row = QtWidgets.QWidget()
        labels_layout = QtWidgets.QHBoxLayout(labels_row)
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_layout.setSpacing(8)
        self.protocol_labels_edit = QtWidgets.QLineEdit("Left,Right,Forward")
        self.protocol_labels_edit.setPlaceholderText("comma-separated labels")
        self.protocol_labels_edit.setReadOnly(True)
        self.edit_labels_button = QtWidgets.QPushButton("Edit")
        self.edit_labels_button.clicked.connect(self._open_labels_editor)
        labels_layout.addWidget(self.protocol_labels_edit, 1)
        labels_layout.addWidget(self.edit_labels_button)
        collect_form.addRow("Class Labels", labels_row)

        self.labels_state_label = QtWidgets.QLabel("No class labels saved yet. Click Edit.")
        collect_form.addRow("Label State", self.labels_state_label)

        self.protocol_base_trial = datetime.now().strftime("trial_%Y%m%d_%H%M%S")

        self.protocol_prep_spin = QtWidgets.QSpinBox()
        self.protocol_prep_spin.setRange(0, 30)
        self.protocol_prep_spin.setValue(3)

        self.protocol_hold_spin = QtWidgets.QSpinBox()
        self.protocol_hold_spin.setRange(1, 60)
        self.protocol_hold_spin.setValue(4)

        self.protocol_rest_spin = QtWidgets.QSpinBox()
        self.protocol_rest_spin.setRange(0, 60)
        self.protocol_rest_spin.setValue(6)

        self.protocol_repeats_spin = QtWidgets.QSpinBox()
        self.protocol_repeats_spin.setRange(1, 100)
        self.protocol_repeats_spin.setValue(5)

        timing_row = QtWidgets.QWidget()
        timing_layout = QtWidgets.QHBoxLayout(timing_row)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        timing_layout.setSpacing(6)
        timing_layout.addWidget(QtWidgets.QLabel("Prep"))
        timing_layout.addWidget(self.protocol_prep_spin)
        timing_layout.addWidget(QtWidgets.QLabel("Hold"))
        timing_layout.addWidget(self.protocol_hold_spin)
        timing_layout.addWidget(QtWidgets.QLabel("Rest"))
        timing_layout.addWidget(self.protocol_rest_spin)
        timing_layout.addWidget(QtWidgets.QLabel("Repeats"))
        timing_layout.addWidget(self.protocol_repeats_spin)
        collect_form.addRow("Timing (s)", timing_row)

        self.record_rest_check = QtWidgets.QCheckBox("")
        self.record_rest_check.setChecked(True)
        collect_form.addRow("Options", self.record_rest_check)

        self.proceed_button = QtWidgets.QPushButton("Proceed")
        self.proceed_button.clicked.connect(self._on_proceed_clicked)
        collect_form.addRow("Prepare", self.proceed_button)

        self.protocol_toggle = QtWidgets.QPushButton("Start Recording")
        self.protocol_toggle.setCheckable(True)
        self.protocol_toggle.toggled.connect(self._on_protocol_toggled)
        collect_form.addRow("Record", self.protocol_toggle)

        self.protocol_status_label = QtWidgets.QLabel("Protocol: Idle")
        collect_form.addRow("Protocol State", self.protocol_status_label)

        self.collect_info_label = QtWidgets.QLabel("Session Samples: 0")
        collect_form.addRow("Stats", self.collect_info_label)

        self.agree_yes.stateChanged.connect(self._on_agree_yes_changed)
        self.agree_no.stateChanged.connect(self._on_agree_no_changed)
        self.top_grid.addWidget(self.collect_group, 0, 0)

        self.train_group = QtWidgets.QGroupBox("2) Train Model")
        train_form = QtWidgets.QFormLayout(self.train_group)
        train_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        train_form.setHorizontalSpacing(10)
        train_form.setVerticalSpacing(8)

        self.train_input_edit = QtWidgets.QLineEdit(str(default_dataset_root))
        browse_train_in = QtWidgets.QPushButton("CSV")
        browse_train_in.clicked.connect(self._browse_train_input_file)
        browse_train_folder = QtWidgets.QPushButton("Folder")
        browse_train_folder.clicked.connect(self._browse_train_input_folder)
        train_in_row = QtWidgets.QHBoxLayout()
        train_in_row.addWidget(self.train_input_edit, 1)
        train_in_row.addWidget(browse_train_in)
        train_in_row.addWidget(browse_train_folder)
        train_in_box = QtWidgets.QWidget()
        train_in_box.setLayout(train_in_row)
        train_form.addRow("Dataset Input", train_in_box)

        default_model_root = PROJECT_ROOT / DEFAULT_ML_MODEL_DIR
        self.model_output_root_edit = QtWidgets.QLineEdit(str(default_model_root))
        browse_model_out = QtWidgets.QPushButton("Browse")
        browse_model_out.clicked.connect(self._browse_model_output_root)
        model_out_row = QtWidgets.QHBoxLayout()
        model_out_row.addWidget(self.model_output_root_edit, 1)
        model_out_row.addWidget(browse_model_out)
        model_out_box = QtWidgets.QWidget()
        model_out_box.setLayout(model_out_row)
        train_form.addRow("Model Root", model_out_box)

        self.run_name_edit = QtWidgets.QLineEdit(DEFAULT_ML_RUN_NAME)
        train_form.addRow("Run Name", self.run_name_edit)

        self.model_file_edit = QtWidgets.QLineEdit(DEFAULT_ML_MODEL_ARTIFACT)
        train_form.addRow("Model File", self.model_file_edit)

        self.window_ms_spin = QtWidgets.QSpinBox()
        self.window_ms_spin.setRange(500, 5000)
        self.window_ms_spin.setSingleStep(100)
        self.window_ms_spin.setValue(2000)
        train_form.addRow("Window (ms)", self.window_ms_spin)

        self.stride_ms_spin = QtWidgets.QSpinBox()
        self.stride_ms_spin.setRange(100, 2000)
        self.stride_ms_spin.setSingleStep(50)
        self.stride_ms_spin.setValue(500)

        window_stride_row = QtWidgets.QWidget()
        window_stride_layout = QtWidgets.QHBoxLayout(window_stride_row)
        window_stride_layout.setContentsMargins(0, 0, 0, 0)
        window_stride_layout.setSpacing(6)
        window_stride_layout.addWidget(QtWidgets.QLabel("Window"))
        window_stride_layout.addWidget(self.window_ms_spin)
        window_stride_layout.addWidget(QtWidgets.QLabel("Stride"))
        window_stride_layout.addWidget(self.stride_ms_spin)
        train_form.addRow("Window/Stride (ms)", window_stride_row)

        self.trees_spin = QtWidgets.QSpinBox()
        self.trees_spin.setRange(100, 3000)
        self.trees_spin.setValue(800)

        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(0, 100)
        self.depth_spin.setValue(0)
        self.depth_spin.setToolTip("0 means no max depth.")

        rf_row = QtWidgets.QWidget()
        rf_layout = QtWidgets.QHBoxLayout(rf_row)
        rf_layout.setContentsMargins(0, 0, 0, 0)
        rf_layout.setSpacing(6)
        rf_layout.addWidget(QtWidgets.QLabel("Trees"))
        rf_layout.addWidget(self.trees_spin)
        rf_layout.addWidget(QtWidgets.QLabel("Max Depth"))
        rf_layout.addWidget(self.depth_spin)
        train_form.addRow("RF Params", rf_row)

        self.test_split_spin = QtWidgets.QDoubleSpinBox()
        self.test_split_spin.setRange(0.1, 0.4)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.2)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)

        eval_row = QtWidgets.QWidget()
        eval_layout = QtWidgets.QHBoxLayout(eval_row)
        eval_layout.setContentsMargins(0, 0, 0, 0)
        eval_layout.setSpacing(6)
        eval_layout.addWidget(QtWidgets.QLabel("Test Split"))
        eval_layout.addWidget(self.test_split_spin)
        eval_layout.addWidget(QtWidgets.QLabel("Seed"))
        eval_layout.addWidget(self.seed_spin)
        train_form.addRow("Eval Params", eval_row)

        self.train_button = QtWidgets.QPushButton("Train EEG Model")
        self.train_button.clicked.connect(self._on_train_clicked)
        train_form.addRow("Train", self.train_button)
        self.top_grid.addWidget(self.train_group, 0, 1)

        root_layout.addWidget(self.top_panel, 2)

        self.load_group = QtWidgets.QGroupBox("3) Load Model")
        predict_form = QtWidgets.QFormLayout(self.load_group)
        predict_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        predict_form.setHorizontalSpacing(10)
        predict_form.setVerticalSpacing(8)

        self.run_folder_edit = QtWidgets.QLineEdit(str(default_model_root))
        browse_run_folder = QtWidgets.QPushButton("Browse")
        browse_run_folder.clicked.connect(self._browse_run_folder)
        load_run_button = QtWidgets.QPushButton("Load Run")
        load_run_button.clicked.connect(self._on_load_run_folder_clicked)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_folder_edit, 1)
        run_row.addWidget(browse_run_folder)
        run_row.addWidget(load_run_button)
        run_box = QtWidgets.QWidget()
        run_box.setLayout(run_row)
        predict_form.addRow("Run Folder", run_box)

        self.model_path_edit = QtWidgets.QLineEdit(str(default_model_root / DEFAULT_ML_MODEL_ARTIFACT))
        browse_model_in = QtWidgets.QPushButton("Browse")
        browse_model_in.clicked.connect(self._browse_model_input)
        load_model_button = QtWidgets.QPushButton("Load File")
        load_model_button.clicked.connect(self._on_load_model_clicked)
        model_in_row = QtWidgets.QHBoxLayout()
        model_in_row.addWidget(self.model_path_edit, 1)
        model_in_row.addWidget(browse_model_in)
        model_in_row.addWidget(load_model_button)
        model_in_box = QtWidgets.QWidget()
        model_in_box.setLayout(model_in_row)
        predict_form.addRow("Model File", model_in_box)

        self.loaded_model_label = QtWidgets.QLabel("No model loaded.")
        predict_form.addRow("Loaded Model", self.loaded_model_label)

        self.predict_toggle = QtWidgets.QPushButton("Start Live Prediction")
        self.predict_toggle.setCheckable(True)
        self.predict_toggle.toggled.connect(self._on_predict_toggled)
        predict_form.addRow("Prediction", self.predict_toggle)

        self.pred_label = QtWidgets.QLabel("Prediction: N/A")
        self.conf_label = QtWidgets.QLabel("Confidence: N/A")
        self.latency_label = QtWidgets.QLabel("Latency: N/A")

        output_row = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(10)
        output_layout.addWidget(self.pred_label, 1)
        output_layout.addWidget(self.conf_label, 1)
        output_layout.addWidget(self.latency_label, 1)
        predict_form.addRow("Live Output", output_row)
        root_layout.addWidget(self.load_group, 1)

        self.metrics_box = QtWidgets.QPlainTextEdit()
        self.metrics_box.setReadOnly(True)
        self.metrics_box.setMinimumHeight(190)
        self.log_group = QtWidgets.QGroupBox("Training / Runtime Log")
        log_layout = QtWidgets.QVBoxLayout(self.log_group)
        log_layout.setContentsMargins(10, 10, 10, 10)
        log_layout.addWidget(self.metrics_box)
        root_layout.addWidget(self.log_group, 1)

        self._build_data_collection_page(root_layout)
        self._build_recording_session_window()

    def _build_data_collection_page(self, root_layout: QtWidgets.QVBoxLayout) -> None:
        self.collection_page = QtWidgets.QWidget()
        self.collection_page.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        page_layout = QtWidgets.QVBoxLayout(self.collection_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(14)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(18)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnMinimumWidth(0, 410)
        grid.setColumnMinimumWidth(1, 410)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 0)

        self.contributor_info_group = QtWidgets.QGroupBox("Contributor Information")
        contributor_layout = QtWidgets.QGridLayout(self.contributor_info_group)
        contributor_layout.setContentsMargins(14, 14, 14, 14)
        contributor_layout.setHorizontalSpacing(10)
        contributor_layout.setVerticalSpacing(10)

        self.contributor_age_edit = QtWidgets.QLineEdit("")
        self.contributor_age_edit.setPlaceholderText("Age")
        self.contributor_age_edit.setValidator(QtGui.QIntValidator(1, 120, self))
        self.contributor_age_edit.setFixedWidth(108)
        self.contributor_sex_edit = QtWidgets.QLineEdit("")
        self.contributor_sex_edit.setPlaceholderText("Sex")
        self.contributor_sex_edit.setFixedWidth(108)

        contributor_layout.addWidget(QtWidgets.QLabel("Name:"), 0, 0)
        contributor_layout.addWidget(self.contributor_edit, 0, 1, 1, 6)
        contributor_layout.addWidget(QtWidgets.QLabel("Age:"), 1, 0)
        contributor_layout.addWidget(self.contributor_age_edit, 1, 1)
        contributor_layout.addItem(
            QtWidgets.QSpacerItem(16, 10, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum),
            1,
            2,
        )
        contributor_layout.addWidget(QtWidgets.QLabel("Sex:"), 1, 3)
        contributor_layout.addWidget(self.contributor_sex_edit, 1, 4, 1, 2)
        consent_row = QtWidgets.QWidget()
        consent_row.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        consent_row.setStyleSheet("background: transparent; border: none;")
        consent_row_layout = QtWidgets.QHBoxLayout(consent_row)
        consent_row_layout.setContentsMargins(0, 0, 0, 0)
        consent_row_layout.setSpacing(10)
        consent_row_layout.addWidget(QtWidgets.QLabel("Consent:"), 0)
        consent_row_layout.addWidget(self.agree_yes, 0)
        consent_row_layout.addWidget(self.agree_no, 0)
        consent_row_layout.addStretch(1)
        consent_row_layout.addWidget(self.terms_button, 0)
        contributor_layout.addWidget(consent_row, 2, 0, 1, 7)
        contributor_layout.setColumnStretch(4, 1)
        contributor_layout.setColumnStretch(5, 1)
        contributor_layout.setColumnStretch(6, 1)
        contributor_layout.setRowStretch(3, 0)

        self.class_label_group = QtWidgets.QGroupBox("Class Label")
        class_layout = QtWidgets.QGridLayout(self.class_label_group)
        class_layout.setContentsMargins(14, 16, 14, 16)
        class_layout.setHorizontalSpacing(10)
        class_layout.setVerticalSpacing(12)

        self.total_class_count_edit = QtWidgets.QLineEdit()
        self.total_class_count_edit.setReadOnly(True)
        self.total_class_count_edit.setMaximumWidth(64)
        self.labels_proceed_button = QtWidgets.QPushButton("Proceed")
        self.labels_proceed_button.clicked.connect(self._on_labels_proceed_clicked)
        self.labels_proceed_button.setStyleSheet(themed_button_style("success"))

        class_layout.addWidget(QtWidgets.QLabel("Total Class Count"), 0, 0)
        class_layout.addWidget(self.total_class_count_edit, 0, 1)
        class_layout.addWidget(QtWidgets.QLabel("Class Labels:"), 1, 0)
        class_layout.addWidget(self.protocol_labels_edit, 1, 1, 1, 3)
        class_layout.addWidget(self.edit_labels_button, 2, 0)
        class_layout.addWidget(self.labels_state_label, 2, 1, 1, 2)
        class_layout.addWidget(self.labels_proceed_button, 2, 3)
        class_layout.setColumnStretch(1, 1)
        class_layout.setColumnStretch(2, 1)
        class_layout.setColumnMinimumWidth(3, 106)
        class_layout.setRowStretch(3, 0)

        self.protocol_group_box = QtWidgets.QGroupBox("Data Collection Protocol (s)")
        protocol_layout = QtWidgets.QGridLayout(self.protocol_group_box)
        protocol_layout.setContentsMargins(14, 12, 14, 10)
        protocol_layout.setHorizontalSpacing(10)
        protocol_layout.setVerticalSpacing(8)

        self.protocol_task_label = QtWidgets.QLabel("Task Duration")
        protocol_layout.addWidget(QtWidgets.QLabel("Preparation Duration:"), 0, 0)
        protocol_layout.addWidget(self.protocol_prep_spin, 0, 1)
        protocol_layout.addWidget(self.protocol_task_label, 0, 2)
        protocol_layout.addWidget(self.protocol_hold_spin, 0, 3)
        protocol_layout.addWidget(QtWidgets.QLabel("Rest Duration:"), 1, 0)
        protocol_layout.addWidget(self.protocol_rest_spin, 1, 1)
        protocol_layout.addWidget(QtWidgets.QLabel("Repeat:"), 1, 2)
        protocol_layout.addWidget(self.protocol_repeats_spin, 1, 3)
        protocol_bottom_row = QtWidgets.QWidget()
        protocol_bottom_row.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        protocol_bottom_row.setStyleSheet("background: transparent; border: none;")
        protocol_bottom_layout = QtWidgets.QHBoxLayout(protocol_bottom_row)
        protocol_bottom_layout.setContentsMargins(0, 0, 0, 0)
        protocol_bottom_layout.setSpacing(8)
        protocol_bottom_layout.addWidget(self.record_rest_check, 0)
        protocol_bottom_layout.addWidget(QtWidgets.QLabel("Record Rest"), 0)
        protocol_bottom_layout.addStretch(1)
        protocol_bottom_layout.addWidget(self.proceed_button, 0)
        protocol_layout.addWidget(protocol_bottom_row, 2, 0, 1, 4)
        protocol_layout.setColumnStretch(1, 1)
        protocol_layout.setColumnStretch(3, 1)
        protocol_layout.setRowStretch(3, 0)
        self.proceed_button.setText("Proceed")
        self.proceed_button.setStyleSheet(themed_button_style("success"))

        self.record_group = QtWidgets.QGroupBox("Record Session")
        record_layout = QtWidgets.QGridLayout(self.record_group)
        record_layout.setContentsMargins(14, 16, 14, 16)
        record_layout.setHorizontalSpacing(10)
        record_layout.setVerticalSpacing(12)

        self.record_summary_primary = QtWidgets.QLabel(
            "Follow the on-screen instructions. For motor imagery, stay relaxed and keep body movement minimal."
        )
        self.record_summary_primary.setWordWrap(True)
        self.record_summary_secondary = QtWidgets.QLabel("")
        self.record_summary_secondary.setWordWrap(True)
        self.record_summary_secondary.hide()
        self.record_summary_block = QtWidgets.QWidget()
        self.record_summary_block.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.record_summary_block.setStyleSheet("background: transparent; border: none;")
        record_summary_layout = QtWidgets.QVBoxLayout(self.record_summary_block)
        record_summary_layout.setContentsMargins(0, 0, 0, 0)
        record_summary_layout.setSpacing(0)
        record_summary_layout.addWidget(self.record_summary_primary, 0)
        record_summary_layout.addWidget(self.record_summary_secondary, 0)
        record_summary_layout.addStretch(0)
        self.folder_name_edit = QtWidgets.QLineEdit("")
        self.folder_name_edit.setMaximumWidth(180)
        self.open_record_session_button = QtWidgets.QPushButton("Start Recording")
        self.open_record_session_button.clicked.connect(self._open_recording_session_window)
        self.open_record_session_button.setStyleSheet(themed_button_style("accent"))
        self.open_record_session_button.setMinimumWidth(150)
        self.open_record_session_button.setMaximumWidth(170)
        self.record_status_separator = QtWidgets.QLabel("|")
        self.record_status_row = QtWidgets.QWidget()
        self.record_status_row.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.record_status_row.setStyleSheet("background: transparent; border: none;")
        record_status_layout = QtWidgets.QHBoxLayout(self.record_status_row)
        record_status_layout.setContentsMargins(0, 0, 0, 0)
        record_status_layout.setSpacing(8)
        record_status_layout.addWidget(self.protocol_status_label, 0)
        record_status_layout.addWidget(self.record_status_separator, 0)
        record_status_layout.addWidget(self.collect_info_label, 0)
        record_status_layout.addStretch(1)

        record_layout.setVerticalSpacing(10)
        record_layout.addWidget(self.record_summary_block, 0, 0, 1, 4)
        record_layout.addWidget(self.record_status_row, 1, 0, 1, 4)
        record_layout.addWidget(QtWidgets.QLabel("Folder Name:"), 2, 0)
        record_layout.addWidget(self.folder_name_edit, 2, 1, 1, 2)
        record_layout.addWidget(self.open_record_session_button, 2, 3)
        record_layout.setColumnStretch(1, 1)
        record_layout.setColumnStretch(2, 1)
        record_layout.setRowStretch(3, 0)

        grid.addWidget(self.contributor_info_group, 0, 0)
        grid.addWidget(self.class_label_group, 0, 1)
        grid.addWidget(self.protocol_group_box, 1, 0)
        grid.addWidget(self.record_group, 1, 1)
        page_layout.addLayout(grid, 0)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(12)
        self.collection_instruction_edit = QtWidgets.QLineEdit()
        self.collection_instruction_edit.setReadOnly(True)
        self.collection_instruction_edit.setMinimumHeight(42)
        self.collection_instruction_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.collection_instruction_edit.setPlaceholderText("Instruction:")
        self.load_collection_config_button = QtWidgets.QPushButton("Load Config")
        self.load_collection_config_button.setStyleSheet(themed_button_style("muted"))
        self.load_collection_config_button.clicked.connect(self._load_collection_config)
        self.save_collection_config_button = QtWidgets.QPushButton("Save Config")
        self.save_collection_config_button.setStyleSheet(themed_button_style("accent"))
        self.save_collection_config_button.clicked.connect(self._save_collection_config)
        self.load_collection_config_button.setMaximumWidth(116)
        self.save_collection_config_button.setMaximumWidth(116)
        config_buttons_row = QtWidgets.QWidget()
        config_buttons_layout = QtWidgets.QHBoxLayout(config_buttons_row)
        config_buttons_layout.setContentsMargins(0, 0, 0, 0)
        config_buttons_layout.setSpacing(12)
        config_buttons_layout.addWidget(self.load_collection_config_button, 0)
        config_buttons_layout.addWidget(self.save_collection_config_button, 0)
        bottom_row.addWidget(self.collection_instruction_edit, 1)
        bottom_row.addWidget(config_buttons_row, 0, QtCore.Qt.AlignRight)
        page_layout.addLayout(bottom_row)

        root_layout.insertWidget(3, self.collection_page, 0, QtCore.Qt.AlignHCenter)

        self.class_label_opacity = QtWidgets.QGraphicsOpacityEffect(self.class_label_group)
        self.class_label_group.setGraphicsEffect(self.class_label_opacity)
        self.protocol_opacity = QtWidgets.QGraphicsOpacityEffect(self.protocol_group_box)
        self.protocol_group_box.setGraphicsEffect(self.protocol_opacity)
        self.record_opacity = QtWidgets.QGraphicsOpacityEffect(self.record_group)
        self.record_group.setGraphicsEffect(self.record_opacity)

        self.contributor_edit.textChanged.connect(self._on_contributor_info_changed)
        self.contributor_age_edit.textChanged.connect(self._on_contributor_info_changed)
        self.contributor_sex_edit.textChanged.connect(self._on_contributor_info_changed)
        self.protocol_prep_spin.valueChanged.connect(self._on_protocol_settings_changed)
        self.protocol_hold_spin.valueChanged.connect(self._on_protocol_settings_changed)
        self.protocol_rest_spin.valueChanged.connect(self._on_protocol_settings_changed)
        self.protocol_repeats_spin.valueChanged.connect(self._on_protocol_settings_changed)
        self.record_rest_check.stateChanged.connect(self._on_protocol_settings_changed)

        protocol_spin_style = (
            f"QSpinBox {{ background-color: {THEME_COLORS['input_bg']}; padding: 0px 8px 0px 8px; min-height: 34px; }}"
            "QSpinBox::up-button, QSpinBox::down-button { width: 20px; }"
        )
        for spin in [self.protocol_prep_spin, self.protocol_hold_spin, self.protocol_rest_spin, self.protocol_repeats_spin]:
            spin.setAlignment(QtCore.Qt.AlignCenter)
            spin.setStyleSheet(protocol_spin_style)
            spin.setFixedWidth(92)

        for group in [
            self.contributor_info_group,
            self.class_label_group,
            self.protocol_group_box,
            self.record_group,
        ]:
            group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            group.setFixedHeight(196)

        self.labels_saved = bool(self._parse_protocol_labels())
        self._refresh_class_labels_display()
        self._refresh_folder_name_preview(force_new_stamp=True)

    def _build_recording_session_window(self) -> None:
        self.recording_session_window = RecordingSessionWindow(self)
        self.recording_session_window.hide()
        self.recording_session_window.start_requested.connect(self._start_recording_session_window)
        self.recording_session_window.cancel_requested.connect(self._cancel_recording_session)
        self.recording_session_window.toggle_pause_requested.connect(self._toggle_recording_session_pause)

    def _fit_data_collection_window_height(self) -> None:
        cards = [
            self.contributor_info_group,
            self.class_label_group,
            self.protocol_group_box,
            self.record_group,
        ]
        for card in cards:
            card.setMinimumWidth(0)
            card.setMaximumWidth(16777215)
        self.root_layout.invalidate()
        self.root_layout.activate()
        common_col_width = max(360, max(card.sizeHint().width() for card in cards))
        for card in cards:
            card.setFixedWidth(common_col_width)
        self.collection_page.adjustSize()
        self.root_layout.invalidate()
        self.root_layout.activate()
        margins = self.root_layout.contentsMargins()
        page_hint = self.collection_page.sizeHint()
        hint_width = max(680, page_hint.width()) + margins.left() + margins.right()
        hint_height = max(420, page_hint.height()) + margins.top() + margins.bottom()
        self.resize(hint_width, hint_height)
        self.setMinimumSize(hint_width, hint_height)

    def _apply_visual_theme(self) -> None:
        self.subtitle_label.setStyleSheet(themed_label_style("muted"))
        self.status_label.setStyleSheet("font-size: 16px; font-weight: 700; background: transparent;")
        self.labels_state_label.setStyleSheet(themed_label_style("muted"))
        self.session_info_label.setStyleSheet(themed_label_style("muted"))
        self.protocol_status_label.setStyleSheet(themed_label_style("warning"))
        self.collect_info_label.setStyleSheet(themed_label_style("muted"))
        self.loaded_model_label.setStyleSheet(themed_label_style("muted"))
        self.record_summary_primary.setStyleSheet(f"color: {THEME_COLORS['muted']}; font-weight: 400;")
        self.record_summary_secondary.setStyleSheet(themed_label_style("muted"))
        self.record_status_separator.setStyleSheet(themed_label_style("muted"))
        self.pred_label.setStyleSheet("font-weight: 700;")
        self.conf_label.setStyleSheet("font-weight: 700;")
        self.latency_label.setStyleSheet(themed_label_style("muted"))
        self.collection_instruction_edit.setStyleSheet(
            f"QLineEdit {{ background-color: {THEME_COLORS['panel_alt']}; font-weight: 700; }}"
        )
        section_title_style = "QGroupBox { font-size: 18px; font-weight: 700; }"
        for group in [
            self.contributor_info_group,
            self.class_label_group,
            self.protocol_group_box,
            self.record_group,
        ]:
            group.setStyleSheet(section_title_style)

        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        fixed_font.setPointSize(max(fixed_font.pointSize(), 10))
        self.metrics_box.setFont(fixed_font)
        self.metrics_box.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {THEME_COLORS['input_bg']}; "
            f"border: 1px solid {THEME_COLORS['border']}; border-radius: 10px; }}"
        )

        button_styles = {
            self.terms_button: "muted",
            self.edit_labels_button: "accent",
            self.labels_proceed_button: "success",
            self.proceed_button: "success",
            self.protocol_toggle: "accent",
            self.open_record_session_button: "accent",
            self.load_collection_config_button: "muted",
            self.save_collection_config_button: "accent",
            self.train_button: "success",
            self.predict_toggle: "accent",
        }
        for button, kind in button_styles.items():
            button.setStyleSheet(themed_button_style(kind))

        for button in self.findChildren(QtWidgets.QPushButton):
            if button not in button_styles:
                button.setStyleSheet(themed_button_style("muted"))

        for edit in self.findChildren(QtWidgets.QLineEdit):
            edit.setMinimumHeight(34)
        for spin in self.findChildren(QtWidgets.QAbstractSpinBox):
            spin.setMinimumHeight(34)
        for combo in self.findChildren(QtWidgets.QComboBox):
            combo.setMinimumHeight(34)

        self.collect_group.setMinimumWidth(700)
        self.train_group.setMinimumWidth(700)
        self.load_group.setMinimumWidth(700)
        self.log_group.setMinimumWidth(700)
        self.contributor_edit.setMaximumWidth(420)
        self.protocol_labels_edit.setMaximumWidth(300)
        self.folder_name_edit.setMaximumWidth(180)
        self.collection_page.setMinimumWidth(780)

    def set_stream_context(self, sample_rate: int, channel_count: int, connected: bool) -> None:
        changed = (int(sample_rate) != self.sample_rate) or (int(channel_count) != self.channel_count)
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self.connected = bool(connected)
        if changed:
            self._reset_inference_ring()
        self._update_ui_state()

    def _center_dialog_on_self(self, dialog: QtWidgets.QWidget) -> None:
        frame = self.frameGeometry()
        center = frame.center()
        dialog_frame = dialog.frameGeometry()
        dialog_frame.moveCenter(center)
        dialog.move(dialog_frame.topLeft())

    def _refresh_class_labels_display(self) -> None:
        labels = self._parse_protocol_labels()
        self.protocol_labels_edit.setText(",".join(labels))
        self.label_image_paths = {label: self.label_image_paths[label] for label in labels if label in self.label_image_paths}
        self.total_class_count_edit.setText(str(len(labels)))
        if self.labels_saved and labels:
            image_count = sum(1 for label in labels if self.label_image_paths.get(label))
            suffix = f" | {image_count} image cue(s)" if image_count else ""
            self.labels_state_label.setText(f"Saved {len(labels)} class labels.{suffix}")
            self.labels_state_label.setStyleSheet(themed_label_style("success"))
        else:
            self.labels_state_label.setText("No class labels saved yet. Click Edit Label to configure and save.")
            self.labels_state_label.setStyleSheet(themed_label_style("muted"))

    def _refresh_folder_name_preview(self, force_new_stamp: bool = False) -> None:
        if force_new_stamp:
            self.folder_name_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self._sanitize_for_trial(self.contributor_edit.text().strip() or "anonymous")
        age = self._sanitize_for_trial(self.contributor_age_edit.text().strip() or "age")
        sex = self._sanitize_for_trial(self.contributor_sex_edit.text().strip() or "sex")
        preview = f"{name}_{age}_{sex}"
        if not self.folder_name_edit.text().strip() or force_new_stamp or (not self.protocol_running and not self.session_prepared):
            self.folder_name_edit.setText(preview)

    def _contributor_fields_complete(self) -> bool:
        return (
            bool(self.contributor_edit.text().strip())
            and bool(self.contributor_age_edit.text().strip())
            and bool(self.contributor_sex_edit.text().strip())
        )

    def _contributor_unlocked(self) -> bool:
        return self._contributor_fields_complete() and self.agree_yes.isChecked() and (not self.agree_no.isChecked())

    def _set_dimmed_enabled(self, widget: QtWidgets.QWidget, enabled: bool, opacity_effect: QtWidgets.QGraphicsOpacityEffect | None) -> None:
        widget.setEnabled(bool(enabled))
        if opacity_effect is not None:
            opacity_effect.setOpacity(1.0 if enabled else 0.42)

    def _collection_instruction_text(self) -> str:
        if self.prestart_running:
            remaining = max(0.0, self.prestart_end_ts - time.monotonic())
            return f"Instruction: Session starts in {remaining:.1f} sec. Keep still and focus on the upcoming cue."
        if self.session_finished_waiting_for_close:
            return "Instruction: Session Finished | Close this window"
        if self.protocol_running:
            return "Instruction: Recording is in progress. Follow the recording-session window prompts."
        if not self.contributor_edit.text().strip():
            return "Instruction: Enter contributor name."
        if not self.contributor_age_edit.text().strip():
            return "Instruction: Enter contributor age."
        if not self.contributor_sex_edit.text().strip():
            return "Instruction: Enter contributor sex."
        if not self._contributor_unlocked():
            return "Instruction: Set consent to Yes to unlock class labels."
        if not self.labels_saved:
            return "Instruction: Edit and save the class labels."
        if not self.labels_confirmed:
            return "Instruction: Review the class labels, then click Proceed in Class Label."
        if not self.protocol_confirmed:
            return "Instruction: Review the protocol values, then click Proceed in Data Collection Protocol."
        if not self.streaming:
            return "Instruction: Start the EEG stream in the main app, then open Start Recording."
        return "Instruction: Open Start Recording, then begin the session from the recording window."

    def _on_contributor_info_changed(self, *_args) -> None:
        self.labels_confirmed = False if not self._contributor_unlocked() else self.labels_confirmed
        self.protocol_confirmed = False if not self._contributor_unlocked() else self.protocol_confirmed
        self.session_prepared = False if not self._contributor_unlocked() else self.session_prepared
        self._refresh_folder_name_preview()
        self._update_ui_state()

    def _on_protocol_settings_changed(self, *_args) -> None:
        if self.protocol_running or self.prestart_running:
            return
        self.protocol_confirmed = False
        self.session_prepared = False
        self._refresh_folder_name_preview(force_new_stamp=True)
        self._update_ui_state()

    def _on_labels_proceed_clicked(self) -> None:
        labels = self._parse_protocol_labels()
        if not labels or not self.labels_saved:
            QtWidgets.QMessageBox.warning(self, "Class Labels", "Edit and save the class labels before proceeding.")
            return
        self.labels_confirmed = True
        self.protocol_confirmed = False
        self.session_prepared = False
        self._append_log(f"Class labels confirmed: {', '.join(labels)}")
        self._update_ui_state()

    def _collection_config_default_dir(self) -> Path:
        default_dir = PROJECT_ROOT / DEFAULT_ML_DATA_DIR / "configs"
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir

    def _collection_image_dir(self) -> Path:
        image_dir = PROJECT_ROOT / "code" / "images" / "class_label_image"
        image_dir.mkdir(parents=True, exist_ok=True)
        return image_dir

    def _serialized_label_image_paths(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        for label, image_path in self.label_image_paths.items():
            text = str(image_path or "").strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if path.is_absolute():
                payload[str(label)] = self._to_project_relative(path)
            else:
                payload[str(label)] = text
        return payload

    def _resolve_label_image_paths(self, image_paths: dict, labels: list[str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for label in labels:
            raw = str(image_paths.get(label, "")).strip()
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            resolved[label] = str(candidate)
        return resolved

    def _collection_config_payload(self) -> dict:
        return {
            "task_labels": self._parse_protocol_labels(),
            "label_image_paths": self._serialized_label_image_paths(),
            "labels_saved": bool(self.labels_saved),
            "labels_confirmed": bool(self.labels_confirmed),
            "prep_s": float(self.protocol_prep_spin.value()),
            "hold_s": float(self.protocol_hold_spin.value()),
            "rest_s": float(self.protocol_rest_spin.value()),
            "repeats": int(self.protocol_repeats_spin.value()),
            "record_rest": bool(self.record_rest_check.isChecked()),
        }

    def _save_collection_config(self) -> None:
        default_name = self._sanitize_for_trial(self.contributor_edit.text().strip() or "eeg_collection_config")
        base = self._collection_config_default_dir() / f"{default_name}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Collection Config",
            str(base),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            payload = self._collection_config_payload()
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._append_log(f"Collection config saved: {path}")
        except Exception as exc:  # pylint: disable=broad-except
            QtWidgets.QMessageBox.warning(self, "Save Config", f"Failed to save config:\n{exc}")

    def _load_collection_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Collection Config",
            str(self._collection_config_default_dir()),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            labels = payload.get("task_labels", [])
            if not isinstance(labels, list):
                labels = [x.strip() for x in str(labels).split(",") if x.strip()]
            labels = [str(x).strip() for x in labels if str(x).strip()]
            if labels:
                self.protocol_labels_edit.setText(",".join(labels))
                self.labels_saved = bool(payload.get("labels_saved", True))
                self.labels_confirmed = bool(payload.get("labels_confirmed", True))
                image_paths = payload.get("label_image_paths", {})
                if isinstance(image_paths, dict):
                    self.label_image_paths = self._resolve_label_image_paths(image_paths, labels)
                else:
                    self.label_image_paths = {}
            self.protocol_prep_spin.setValue(int(round(float(payload.get("prep_s", self.protocol_prep_spin.value())))))
            self.protocol_hold_spin.setValue(int(round(float(payload.get("hold_s", self.protocol_hold_spin.value())))))
            self.protocol_rest_spin.setValue(int(round(float(payload.get("rest_s", self.protocol_rest_spin.value())))))
            self.protocol_repeats_spin.setValue(int(payload.get("repeats", self.protocol_repeats_spin.value())))
            self.record_rest_check.setChecked(bool(payload.get("record_rest", self.record_rest_check.isChecked())))
            self.protocol_confirmed = bool(labels)
            self.session_prepared = bool(labels)
            self._refresh_class_labels_display()
            self._refresh_folder_name_preview(force_new_stamp=True)
            self._append_log(f"Collection config loaded: {path}")
            self._update_ui_state()
        except Exception as exc:  # pylint: disable=broad-except
            QtWidgets.QMessageBox.warning(self, "Load Config", f"Failed to load config:\n{exc}")

    def _open_recording_session_window(self) -> None:
        if self.recording_session_window is None:
            return
        self._center_dialog_on_self(self.recording_session_window)
        self.recording_session_window.show()
        self.recording_session_window.raise_()
        self.recording_session_window.activateWindow()
        self._update_recording_session_window()

    def _start_recording_session_window(self) -> None:
        if self.prestart_running or self.protocol_running:
            return
        if not self.streaming:
            QtWidgets.QMessageBox.warning(self, "Stream Required", "Start the EEG stream in the main app before recording.")
            return
        if not self.session_prepared:
            QtWidgets.QMessageBox.information(self, "Proceed First", "Complete the protocol section first, then start recording.")
            return
        self.session_paused = False
        self.paused_mode = ""
        self.paused_remaining_s = 0.0
        self.prestart_running = True
        self.prestart_end_ts = time.monotonic() + float(self.prestart_duration_s)
        self.recording_activity_changed.emit(True)
        self._update_recording_session_window()
        if not self.prestart_timer.isActive():
            self.prestart_timer.start()

    def _on_prestart_tick(self) -> None:
        if not self.prestart_running:
            self.prestart_timer.stop()
            return
        remaining = max(0.0, self.prestart_end_ts - time.monotonic())
        pct = int(np.clip(((self.prestart_duration_s - remaining) / max(0.1, self.prestart_duration_s)) * 100.0, 0.0, 100.0))
        if self.recording_session_window is not None:
            self.recording_session_window.set_activity_image(None)
            self.recording_session_window.set_phase("{Get Ready}")
            self.recording_session_window.set_instruction(
                f"Session starts in {remaining:.1f} sec. Relax, minimize movement, and focus on the first class cue."
            )
            self.recording_session_window.set_progress_value(pct)
            self.recording_session_window.set_progress_format(f"Session starts in {remaining:.1f} sec")
            self.recording_session_window.set_can_start(False)
            self.recording_session_window.set_start_text("Starting...")
        self.collection_instruction_edit.setText(self._collection_instruction_text())
        if remaining > 0.0:
            return
        self.prestart_timer.stop()
        self.prestart_running = False
        self._begin_protocol_run()

    def _cancel_recording_session(self) -> None:
        if self.session_finished_waiting_for_close:
            self.session_finished_waiting_for_close = False
            self.recording_activity_changed.emit(False)
            if self.recording_session_window is not None:
                self.recording_session_window.set_activity_image(None)
            self._update_ui_state()
            return
        self.session_paused = False
        self.paused_mode = ""
        self.paused_remaining_s = 0.0
        if self.prestart_running:
            self.prestart_running = False
            self.prestart_timer.stop()
            self._append_log("Recording countdown canceled.")
        elif self.protocol_running:
            self._stop_protocol("Stopped by user.")
        self.recording_activity_changed.emit(False)
        self._update_recording_session_window()
        self._update_ui_state()

    def _toggle_recording_session_pause(self) -> None:
        if self.prestart_running:
            if not self.session_paused:
                self.paused_remaining_s = max(0.0, self.prestart_end_ts - time.monotonic())
                self.prestart_timer.stop()
                self.session_paused = True
                self.paused_mode = "countdown"
                if self.recording_session_window is not None:
                    self.recording_session_window.set_instruction("Countdown paused. Press Spacebar to resume.")
                    self.recording_session_window.set_progress_format("Paused")
                self.protocol_status_label.setText("Protocol: Countdown paused")
            else:
                self.prestart_end_ts = time.monotonic() + max(0.0, self.paused_remaining_s)
                self.prestart_timer.start()
                self.session_paused = False
                self.paused_mode = ""
                self._append_log("Recording countdown resumed.")
            self.collection_instruction_edit.setText(self._collection_instruction_text())
            self.recording_activity_changed.emit(bool(self.prestart_running or self.protocol_running))
            return

        if self.protocol_running:
            if not self.session_paused:
                self.paused_remaining_s = max(0.0, self.protocol_phase_end_ts - time.monotonic())
                self.protocol_timer.stop()
                self.session_paused = True
                self.paused_mode = "protocol"
                if self.recording_session_window is not None:
                    self.recording_session_window.set_instruction("Session paused. Press Spacebar to resume.")
                    self.recording_session_window.set_progress_format("Paused")
                self.protocol_status_label.setText("Protocol: Paused")
            else:
                self.protocol_phase_end_ts = time.monotonic() + max(0.0, self.paused_remaining_s)
                self.protocol_timer.start()
                self.session_paused = False
                self.paused_mode = ""
                self._append_log("Recording session resumed.")
            self.collection_instruction_edit.setText(self._collection_instruction_text())
            self.recording_activity_changed.emit(bool(self.prestart_running or self.protocol_running))

    def _begin_protocol_run(self) -> None:
        labels = self._parse_protocol_labels()
        if not labels:
            self._append_log("Protocol labels cannot be empty.")
            return
        try:
            self._ensure_collection_session()
        except Exception as exc:  # pylint: disable=broad-except
            self._append_log(f"Cannot create collection session: {exc}")
            return

        self.protocol_labels = labels
        self.record_rest = bool(self.record_rest_check.isChecked())
        self.protocol_repeat_total = int(self.protocol_repeats_spin.value())
        self.protocol_repeat_index = 0
        self.protocol_label_index = 0
        self.protocol_phase = "idle"
        self.protocol_base_trial = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
        self.protocol_running = True
        self.recording_activity_changed.emit(True)
        self._append_log(
            f"Auto protocol started: labels={labels}, repeats={self.protocol_repeat_total}, "
            f"prep={self.protocol_prep_spin.value()}s, hold={self.protocol_hold_spin.value()}s, "
            f"rest={self.protocol_rest_spin.value()}s"
        )
        self._update_ui_state()
        self._enter_protocol_phase("prep")

    def _update_recording_session_window(self) -> None:
        if self.recording_session_window is None:
            return
        if self.session_finished_waiting_for_close:
            self.recording_session_window.set_can_start(False)
            self.recording_session_window.set_start_text("Start Recording")
            self.recording_session_window.set_activity_image(None)
            self.recording_session_window.set_phase("Data Collection\nSession Finished")
            self.recording_session_window.set_instruction("Session Finished | Close this window")
            self.recording_session_window.set_progress_value(100)
            self.recording_session_window.set_progress_format("Finished")
            return
        can_start = self.streaming and self.session_prepared and (not self.prestart_running) and (not self.protocol_running)
        self.recording_session_window.set_can_start(can_start)
        if self.protocol_running:
            self.recording_session_window.set_start_text("Recording...")
        elif self.session_paused:
            self.recording_session_window.set_start_text("Paused")
        else:
            self.recording_session_window.set_start_text("Start Recording")
        if not self.prestart_running and not self.protocol_running:
            self.recording_session_window.set_phase("Get Ready")
            if self.session_paused:
                self.recording_session_window.set_instruction("Session paused. Press Spacebar to resume.")
            else:
                self.recording_session_window.set_instruction("Start this session by clicking Start Recording")
            self.recording_session_window.set_progress_value(0)
            self.recording_session_window.set_progress_format("Ready")

    def _apply_mode_layout(self) -> None:
        mode = self.mode
        if mode == "data_collection":
            self.setWindowTitle("NeuroWave-EEG Data Collection")
            self.title_label.hide()
            self.subtitle_label.setVisible(False)
            self.status_label.setVisible(False)
            self.top_panel.setVisible(False)
            self.collection_page.setVisible(True)
            self.train_group.setVisible(False)
            self.load_group.setVisible(False)
            self.log_group.setVisible(False)
            self.root_layout.setContentsMargins(14, 6, 14, 10)
            self.root_layout.setSpacing(0)
            self.resize(1120, 450)
            self.setMinimumSize(980, 450)
            self._fit_data_collection_window_height()
        elif mode == "train_model":
            self.setWindowTitle("EEG ML - Train Model")
            self.title_label.show()
            self.title_label.setText("Train Model")
            self.subtitle_label.setVisible(True)
            self.status_label.setVisible(True)
            self.subtitle_label.setText("Choose dataset input, configure the training run, and save artifacts into a clean run folder structure.")
            self.top_panel.setVisible(True)
            self.collection_page.setVisible(False)
            self.collect_group.setVisible(False)
            self.train_group.setVisible(True)
            self.load_group.setVisible(False)
            self.log_group.setVisible(True)
            self.root_layout.setContentsMargins(18, 18, 18, 18)
            self.root_layout.setSpacing(14)
            self.top_grid.setColumnStretch(0, 0)
            self.top_grid.setColumnStretch(1, 1)
            self.resize(900, 760)
            self.setMinimumSize(840, 700)
        elif mode == "load_model":
            self.setWindowTitle("EEG ML - Load Model")
            self.title_label.show()
            self.title_label.setText("Load Model")
            self.subtitle_label.setVisible(True)
            self.status_label.setVisible(True)
            self.subtitle_label.setText("Load a saved training run or direct model artifact, then run live EEG predictions when streaming is active.")
            self.top_panel.setVisible(False)
            self.collection_page.setVisible(False)
            self.collect_group.setVisible(False)
            self.train_group.setVisible(False)
            self.load_group.setVisible(True)
            self.log_group.setVisible(True)
            self.root_layout.setContentsMargins(18, 18, 18, 18)
            self.root_layout.setSpacing(14)
            self.resize(900, 740)
            self.setMinimumSize(820, 680)
        else:
            self.setWindowTitle("EEG ML Pipeline")
            self.title_label.show()
            self.title_label.setText("EEG ML Pipeline")
            self.subtitle_label.setVisible(True)
            self.status_label.setVisible(True)
            self.subtitle_label.setText("Collect raw EEG, train a model, and run live inference from dedicated workflow windows.")
            self.top_panel.setVisible(True)
            self.collection_page.setVisible(False)
            self.collect_group.setVisible(True)
            self.train_group.setVisible(True)
            self.load_group.setVisible(True)
            self.log_group.setVisible(True)
            self.root_layout.setContentsMargins(18, 18, 18, 18)
            self.root_layout.setSpacing(14)

    @staticmethod
    def _set_form_row_visible(form: QtWidgets.QFormLayout, field: QtWidgets.QWidget, visible: bool) -> None:
        if form is None or field is None:
            return
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def set_streaming(self, streaming: bool) -> None:
        self.streaming = bool(streaming)
        if not self.streaming and self.prestart_running:
            self.prestart_running = False
            self.prestart_timer.stop()
        if not self.streaming and self.protocol_running:
            self._stop_protocol("Protocol stopped because stream ended.")
        if not self.streaming and self.predict_toggle.isChecked():
            self.predict_toggle.setChecked(False)
        self._update_ui_state()

    def handle_incoming_chunk(
        self,
        eeg_chunk: np.ndarray,
        timestamps: np.ndarray | None = None,
        sample_index: np.ndarray | None = None,
    ) -> None:
        if eeg_chunk is None or eeg_chunk.size == 0:
            return
        if eeg_chunk.shape[0] != self.channel_count:
            return
        if self.session_paused:
            return

        if self.recorder is not None:
            self.recorder.write_chunk(eeg_chunk, timestamps=timestamps, sample_index=sample_index)
            self._update_collection_stats_label()

        if self.prediction_enabled and self.model_bundle is not None:
            self.infer_ring.append(eeg_chunk)
            self._samples_since_submit += int(eeg_chunk.shape[1])
            if self.infer_ring.get_filled_count() < self.model_bundle.window_samples:
                return
            step = max(1, int(self.model_bundle.stride_samples))
            while self._samples_since_submit >= step:
                window = self.infer_ring.get_window()[:, -self.model_bundle.window_samples :]
                self.inference_worker.submit_window(window)
                self._samples_since_submit -= step

    def _start_labeled_recorder(self, label: str, trial_id: str, phase: str) -> None:
        self._stop_labeled_recorder()
        self._ensure_collection_session()
        if self.collection_data_path is None:
            raise RuntimeError("Collection session path is not available.")
        self.collection_labels.add(label)
        self.recorder = LabeledEegRecorder(
            path=str(self.collection_data_path),
            channel_count=self.channel_count,
            label=label,
            trial_id=trial_id,
            phase=phase,
        )

    def _stop_labeled_recorder(self) -> tuple[int, Path | None]:
        if self.recorder is None:
            return 0, None
        rows = int(self.recorder.rows_written)
        path = self.recorder.path
        self.recorder.close()
        self.recorder = None
        self.collection_rows_total += rows
        self._update_collection_stats_label()
        return rows, path

    def _parse_protocol_labels(self) -> list[str]:
        raw = self.protocol_labels_edit.text().strip()
        labels = [part.strip() for part in raw.split(",")]
        return [x for x in labels if x]

    def _show_terms(self) -> None:
        dialog = InfoTextDialog(
            "Terms & Conditions",
            (
                "1. Participation is voluntary.\n\n"
                "2. The recorded EEG data is intended for model development, testing, and research-oriented workflow validation.\n\n"
                "3. Avoid entering personally identifying details beyond the requested contributor information.\n\n"
                "4. If you do not agree, leave consent as No and the collection workflow will remain locked."
            ),
            self,
        )
        self._center_dialog_on_self(dialog)
        dialog.exec_()
        self.terms_viewed = True
        self._update_ui_state()

    def _on_agree_yes_changed(self, _state: int) -> None:
        if self.agree_yes.isChecked():
            self.agree_no.blockSignals(True)
            self.agree_no.setChecked(False)
            self.agree_no.blockSignals(False)
        elif not self.agree_no.isChecked():
            self.agree_no.blockSignals(True)
            self.agree_no.setChecked(True)
            self.agree_no.blockSignals(False)
        self._update_ui_state()

    def _on_agree_no_changed(self, _state: int) -> None:
        if self.agree_no.isChecked():
            self.agree_yes.blockSignals(True)
            self.agree_yes.setChecked(False)
            self.agree_yes.blockSignals(False)
        elif not self.agree_yes.isChecked():
            self.agree_yes.blockSignals(True)
            self.agree_yes.setChecked(True)
            self.agree_yes.blockSignals(False)
        self._update_ui_state()

    def _open_labels_editor(self) -> None:
        dialog = ClassLabelEditorDialog(
            self._parse_protocol_labels(),
            label_image_paths=self.label_image_paths,
            image_dir=self._collection_image_dir(),
            parent=self,
        )
        self._center_dialog_on_self(dialog)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        labels = list(dialog.saved_labels)
        self.protocol_labels_edit.setText(",".join(labels))
        self.label_image_paths = dict(dialog.saved_label_image_paths)
        self.labels_saved = True
        self.labels_confirmed = False
        self.protocol_confirmed = False
        self.session_prepared = False
        self._refresh_class_labels_display()
        self._refresh_folder_name_preview(force_new_stamp=True)
        self._update_ui_state()

    def _on_proceed_clicked(self) -> None:
        if not self._contributor_unlocked():
            QtWidgets.QMessageBox.warning(self, "Contributor Information", "Complete contributor information and set consent to Yes first.")
            return
        if not self.labels_confirmed:
            QtWidgets.QMessageBox.warning(self, "Class Labels", "Confirm the class labels first.")
            return
        if not self._parse_protocol_labels():
            QtWidgets.QMessageBox.warning(self, "Class Labels", "At least one class label is required.")
            return
        self.record_rest = bool(self.record_rest_check.isChecked())
        self.protocol_confirmed = True
        self.session_prepared = True
        self._refresh_folder_name_preview(force_new_stamp=True)
        self.protocol_status_label.setText("Protocol: Ready | Open Start Recording")
        self._append_log("Protocol confirmed. Recording session is unlocked.")
        self._update_ui_state()

    @staticmethod
    def _sanitize_for_trial(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
        return safe.strip("_") or "task"

    @staticmethod
    def _to_project_relative(path: Path) -> str:
        abs_path = path.resolve()
        root = PROJECT_ROOT.resolve()
        try:
            return abs_path.relative_to(root).as_posix()
        except ValueError:
            return str(abs_path)

    def _build_collection_session_paths(self) -> tuple[Path, Path, Path, str]:
        root_text = self.collect_root_edit.text().strip()
        root_dir = Path(root_text) if root_text else (PROJECT_ROOT / DEFAULT_ML_DATA_DIR)
        root_dir = root_dir.expanduser()
        if not root_dir.is_absolute():
            root_dir = (PROJECT_ROOT / root_dir).resolve()
        else:
            root_dir = root_dir.resolve()
        base_bundle_name = self._sanitize_for_trial(
            self.folder_name_edit.text().strip() or self._sanitize_for_trial(self.contributor_edit.text().strip() or "anonymous")
        )
        self.folder_name_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle_name = f"{base_bundle_name}_{self.folder_name_stamp}"
        session_dir = root_dir / bundle_name
        suffix = 2
        while session_dir.exists():
            bundle_name = f"{base_bundle_name}_{self.folder_name_stamp}_{suffix:02d}"
            session_dir = root_dir / bundle_name
            suffix += 1
        data_path = session_dir / f"eeg_data_{bundle_name}.csv"
        metadata_path = session_dir / f"metadata_{bundle_name}.txt"
        return session_dir, data_path, metadata_path, bundle_name

    def _ensure_collection_session(self) -> None:
        if self.collection_data_path is not None and self.collection_session_dir is not None:
            return
        session_dir, data_path, metadata_path, bundle_name = self._build_collection_session_paths()
        session_dir.mkdir(parents=True, exist_ok=True)
        self.collection_session_dir = session_dir
        self.collection_data_path = data_path
        self.collection_metadata_path = metadata_path
        self.collection_bundle_name = bundle_name
        self.collection_rows_total = 0
        self.collection_labels = set()
        self.collection_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_info_label.setText(f"Session: {self._to_project_relative(data_path)}")
        self.train_input_edit.setText(str(data_path))
        self._append_log(f"Collection session started: {data_path}")
        self._update_collection_stats_label()

    def _write_collection_metadata(self) -> None:
        if self.collection_metadata_path is None or self.collection_data_path is None or self.collection_session_dir is None:
            return
        configured_labels = self._parse_protocol_labels()
        labels = configured_labels if configured_labels else sorted(self.collection_labels)
        root_dir = Path(self.collect_root_edit.text().strip()) if self.collect_root_edit.text().strip() else (PROJECT_ROOT / DEFAULT_ML_DATA_DIR)
        root_dir = root_dir.expanduser()
        if not root_dir.is_absolute():
            root_dir = (PROJECT_ROOT / root_dir).resolve()
        else:
            root_dir = root_dir.resolve()
        try:
            data_rel = self.collection_data_path.resolve().relative_to(root_dir).as_posix()
        except ValueError:
            data_rel = self._to_project_relative(self.collection_data_path)
        lines = [
            f"created_at={self.collection_started_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"contributor={self.contributor_edit.text().strip()}",
            f"age={self.contributor_age_edit.text().strip()}",
            f"sex={self.contributor_sex_edit.text().strip()}",
            f"agreement={bool(self.agree_yes.isChecked() and not self.agree_no.isChecked())}",
            f"labels={','.join(labels)}",
            f"label_images={json.dumps(self._serialized_label_image_paths(), sort_keys=True)}",
            f"labels_saved={bool(self.labels_saved)}",
            f"repeats={int(self.protocol_repeats_spin.value())}",
            f"prep_s={float(self.protocol_prep_spin.value())}",
            f"hold_s={float(self.protocol_hold_spin.value())}",
            f"rest_s={float(self.protocol_rest_spin.value())}",
            f"record_rest={bool(self.record_rest)}",
            f"channel_count={int(self.channel_count)}",
            f"sample_rate_hz={int(self.sample_rate)}",
            f"num_samples={int(self.collection_rows_total)}",
            f"data_file={self.collection_data_path.name}",
            f"data_path={data_rel}",
        ]
        self.collection_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _finalize_collection_session(self, reason: str) -> None:
        if self.recorder is not None:
            self._stop_labeled_recorder()
        if self.collection_data_path is None or self.collection_metadata_path is None:
            return
        try:
            self._write_collection_metadata()
            self._append_log(
                f"{reason} Saved {self.collection_rows_total} samples to {self.collection_data_path} "
                f"and metadata to {self.collection_metadata_path}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._append_log(f"Failed to write metadata: {exc}")
        self.train_input_edit.setText(str(self.collection_data_path))
        self.session_info_label.setText(f"Last session: {self._to_project_relative(self.collection_data_path)}")
        self.collection_session_dir = None
        self.collection_data_path = None
        self.collection_metadata_path = None
        self.collection_bundle_name = ""
        self.collection_rows_total = 0
        self.collection_labels = set()
        self.collection_started_at = ""
        self._refresh_folder_name_preview(force_new_stamp=True)
        self._update_collection_stats_label()

    def _update_collection_stats_label(self) -> None:
        active = self.recorder.rows_written if self.recorder is not None else 0
        total = int(self.collection_rows_total + active)
        self.collect_info_label.setText(f"Session Samples: {total}")

    def _on_protocol_toggled(self, checked: bool) -> None:
        if checked:
            self._begin_protocol_run()
        else:
            self._stop_protocol("Stopped by user.")

        self._update_ui_state()

    def _enter_protocol_phase(self, phase: str) -> None:
        self.protocol_phase = phase
        current_label = self.protocol_labels[self.protocol_label_index]
        rep = self.protocol_repeat_index + 1
        total = self.protocol_repeat_total

        if phase == "prep":
            duration_s = int(self.protocol_prep_spin.value())
            self.protocol_status_label.setText(
                f"Protocol: PREP | Next={current_label} | Repeat {rep}/{total}"
            )
            if self.recording_session_window is not None:
                self.recording_session_window.set_activity_image(None)
                self.recording_session_window.set_phase(f"Prepare for {current_label}")
                self.recording_session_window.set_instruction(f"Prepare for {current_label}")
            if duration_s <= 0:
                self._enter_protocol_phase("hold")
                return
        elif phase == "hold":
            duration_s = int(self.protocol_hold_spin.value())
            safe_label = self._sanitize_for_trial(current_label)
            trial_id = f"{self.protocol_base_trial}_r{rep:02d}_{safe_label}"
            try:
                self._start_labeled_recorder(label=current_label, trial_id=trial_id, phase="hold")
            except Exception as exc:  # pylint: disable=broad-except
                self._stop_protocol(f"Protocol failed to start recorder: {exc}")
                return
            self.protocol_status_label.setText(
                f"Protocol: HOLD/RECORD | Label={current_label} | Repeat {rep}/{total}"
            )
            self._append_log(f"Recording started for label={current_label}, trial={trial_id}")
            if self.recording_session_window is not None:
                self.recording_session_window.set_activity_image(self.label_image_paths.get(current_label))
                self.recording_session_window.set_phase(f"{{{current_label}}}")
                self.recording_session_window.set_instruction(
                    f"Imagine the '{current_label}' action now. Stay still, avoid jaw/neck movement, and keep your eyes steady."
                )
        elif phase == "rest":
            rows, path = self._stop_labeled_recorder()
            if path is not None:
                self._append_log(f"Phase saved: {rows} samples -> {path}")
            duration_s = int(self.protocol_rest_spin.value())
            if self.record_rest and duration_s > 0:
                rest_trial_id = f"{self.protocol_base_trial}_r{rep:02d}_rest"
                try:
                    self._start_labeled_recorder(label="Rest", trial_id=rest_trial_id, phase="rest")
                    self.collection_labels.add("Rest")
                except Exception as exc:  # pylint: disable=broad-except
                    self._stop_protocol(f"Protocol failed to start rest recorder: {exc}")
                    return
            self.protocol_status_label.setText(
                f"Protocol: REST | Next={current_label} | Repeat {rep}/{total}"
            )
            if self.recording_session_window is not None:
                self.recording_session_window.set_activity_image(None)
                self.recording_session_window.set_phase("{Rest}")
                self.recording_session_window.set_instruction(
                    "Relax, minimize movement, and prepare for the next class label."
                )
            if duration_s <= 0:
                self._advance_protocol_step()
                return
        else:
            return

        self.protocol_phase_end_ts = time.monotonic() + float(duration_s)
        if not self.protocol_timer.isActive():
            self.protocol_timer.start()
        self._on_protocol_tick()

    def _advance_protocol_step(self) -> None:
        if self.protocol_phase == "rest":
            rows, path = self._stop_labeled_recorder()
            if path is not None and rows > 0:
                self._append_log(f"Rest phase saved: {rows} samples -> {path}")
        self.protocol_label_index += 1
        if self.protocol_label_index >= len(self.protocol_labels):
            self.protocol_label_index = 0
            self.protocol_repeat_index += 1

        if self.protocol_repeat_index >= self.protocol_repeat_total:
            self._stop_protocol("Protocol completed.")
            return
        self._enter_protocol_phase("prep")

    def _on_protocol_tick(self) -> None:
        if not self.protocol_running:
            self.protocol_timer.stop()
            return
        remaining = max(0.0, self.protocol_phase_end_ts - time.monotonic())
        current_label = self.protocol_labels[self.protocol_label_index] if self.protocol_labels else "N/A"
        rep = self.protocol_repeat_index + 1
        total = self.protocol_repeat_total

        if self.protocol_phase == "prep":
            self.protocol_status_label.setText(
                f"Protocol: PREP {remaining:.1f}s | Next={current_label} | Repeat {rep}/{total}"
            )
            if self.recording_session_window is not None:
                self.recording_session_window.set_progress_format(f"Prepare | {remaining:.1f} s")
        elif self.protocol_phase == "hold":
            self.protocol_status_label.setText(
                f"Protocol: HOLD {remaining:.1f}s | Label={current_label} | Repeat {rep}/{total}"
            )
            if self.recording_session_window is not None:
                self.recording_session_window.set_progress_format(f"Record | {remaining:.1f} s")
        elif self.protocol_phase == "rest":
            self.protocol_status_label.setText(
                f"Protocol: REST {remaining:.1f}s | Label={current_label} | Repeat {rep}/{total}"
            )
            if self.recording_session_window is not None:
                self.recording_session_window.set_progress_format(f"Rest | {remaining:.1f} s")

        if self.recording_session_window is not None:
            phase_duration = max(0.1, float(int(self.protocol_prep_spin.value()) if self.protocol_phase == "prep" else int(self.protocol_hold_spin.value()) if self.protocol_phase == "hold" else int(self.protocol_rest_spin.value())))
            pct = int(np.clip(((phase_duration - remaining) / phase_duration) * 100.0, 0.0, 100.0))
            self.recording_session_window.set_progress_value(pct)

        if remaining > 0.0:
            return

        if self.protocol_phase == "prep":
            self._enter_protocol_phase("hold")
        elif self.protocol_phase == "hold":
            self._enter_protocol_phase("rest")
        elif self.protocol_phase == "rest":
            self._advance_protocol_step()

    def _stop_protocol(self, reason: str) -> None:
        if self.protocol_timer.isActive():
            self.protocol_timer.stop()
        rows, path = self._stop_labeled_recorder()
        if path is not None and rows > 0:
            self._append_log(f"Protocol phase saved: {rows} samples -> {path}")
        was_running = self.protocol_running
        self.protocol_running = False
        self.session_paused = False
        self.paused_mode = ""
        self.paused_remaining_s = 0.0
        self.protocol_phase = "idle"
        completed = "complete" in str(reason or "").lower()
        if completed:
            self.session_finished_waiting_for_close = True
        else:
            self.recording_activity_changed.emit(False)
        self.protocol_status_label.setText("Protocol: Idle")
        self.protocol_toggle.blockSignals(True)
        self.protocol_toggle.setChecked(False)
        self.protocol_toggle.blockSignals(False)
        self.protocol_toggle.setText("Start Recording")
        self.session_prepared = False
        if was_running:
            final_reason = "Auto protocol complete." if completed else (reason or "Protocol stopped.")
            self._finalize_collection_session(final_reason)
            if reason:
                self._append_log(reason)
        if self.recording_session_window is not None:
            self.recording_session_window.set_can_start(False)
            self.recording_session_window.set_start_text("Start Recording")
            self.recording_session_window.set_activity_image(None)
            phase_text = "Data Collection\nSession Finished" if "complete" in reason.lower() else "{Session Stopped}"
            self.recording_session_window.set_phase(phase_text)
            self.recording_session_window.set_instruction(
                "Session Finished | Close this window" if "complete" in reason.lower() else (reason or "Session stopped.")
            )
            self.recording_session_window.set_progress_value(100 if "complete" in reason.lower() else 0)
            self.recording_session_window.set_progress_format("Finished" if "complete" in reason.lower() else "Stopped")
        self._update_ui_state()

    def _on_train_clicked(self) -> None:
        if self.train_worker is not None and self.train_worker.isRunning():
            return
        kwargs = {
            "csv_path": self.train_input_edit.text().strip(),
            "output_dir": self.model_output_root_edit.text().strip(),
            "run_name": self.run_name_edit.text().strip() or DEFAULT_ML_RUN_NAME,
            "model_filename": self.model_file_edit.text().strip() or DEFAULT_ML_MODEL_ARTIFACT,
            "sample_rate": self.sample_rate,
            "window_ms": int(self.window_ms_spin.value()),
            "stride_ms": int(self.stride_ms_spin.value()),
            "n_estimators": int(self.trees_spin.value()),
            "max_depth": int(self.depth_spin.value()),
            "test_size": float(self.test_split_spin.value()),
            "random_seed": int(self.seed_spin.value()),
        }
        self.train_worker = TrainModelWorker(kwargs)
        self.train_worker.success.connect(self._on_train_success)
        self.train_worker.failed.connect(self._on_train_failed)
        self.train_button.setEnabled(False)
        self._append_log("Training started...")
        self.train_worker.start()

    def _on_train_success(self, result: dict) -> None:
        self.train_button.setEnabled(True)
        self.train_worker = None
        summary = (
            f"Training done.\n"
            f"Run Folder: {result.get('run_dir', 'N/A')}\n"
            f"Model: {result['model_path']}\n"
            f"Acc: {result['accuracy']:.4f} | Weighted F1: {result['weighted_f1']:.4f}\n"
            f"CV Acc: {result['cv_mean_accuracy']:.4f} +/- {result['cv_std_accuracy']:.4f}\n"
            f"Train/Test windows: {result['train_windows']}/{result['test_windows']}\n"
            f"Classes: {', '.join(result['classes'])}\n"
            f"Feature dim: {result['feature_dim']} | Train time: {result['train_seconds']:.2f}s\n\n"
            f"{result['report']}"
        )
        self.metrics_box.setPlainText(summary)
        self.model_path_edit.setText(result["model_path"])
        if result.get("run_dir"):
            self.run_folder_edit.setText(str(result["run_dir"]))
        self._append_log("Training complete. Loading model artifact.")
        self._on_load_model_clicked()
        self._update_ui_state()

    def _on_train_failed(self, message: str) -> None:
        self.train_button.setEnabled(True)
        self.train_worker = None
        self._append_log(f"Training failed: {message}")
        self._update_ui_state()

    def _on_load_run_folder_clicked(self) -> None:
        run_dir = Path(self.run_folder_edit.text().strip()).expanduser()
        if not run_dir.is_absolute():
            run_dir = (PROJECT_ROOT / run_dir).resolve()
        else:
            run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            self._append_log(f"Run folder not found: {run_dir}")
            return

        setup_path = run_dir / "training_setup.json"
        results_path = run_dir / "training_results.json"
        model_path = run_dir / DEFAULT_ML_MODEL_ARTIFACT

        if results_path.is_file():
            try:
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                model_file = str(payload.get("model_file", DEFAULT_ML_MODEL_ARTIFACT)).strip() or DEFAULT_ML_MODEL_ARTIFACT
                model_path = run_dir / model_file
            except Exception as exc:  # pylint: disable=broad-except
                self._append_log(f"Failed to parse training_results.json: {exc}")

        if setup_path.is_file():
            try:
                setup = json.loads(setup_path.read_text(encoding="utf-8"))
                datasets = list(setup.get("dataset_paths", []))
                if datasets:
                    first = Path(str(datasets[0]))
                    if not first.is_absolute():
                        first = (PROJECT_ROOT / first).resolve()
                    self.train_input_edit.setText(str(first))
                output_dir = str(setup.get("output_dir", "")).strip()
                if output_dir:
                    out_path = Path(output_dir)
                    if not out_path.is_absolute():
                        out_path = (PROJECT_ROOT / out_path).resolve()
                    self.model_output_root_edit.setText(str(out_path))
                run_name = str(setup.get("run_name", "")).strip()
                if run_name:
                    self.run_name_edit.setText(run_name)
            except Exception as exc:  # pylint: disable=broad-except
                self._append_log(f"Failed to parse training_setup.json: {exc}")

        if not model_path.is_file():
            self._append_log(f"Model file not found in run folder: {model_path}")
            return

        self.model_path_edit.setText(str(model_path))
        self._append_log(f"Run folder loaded: {run_dir}")
        self._on_load_model_clicked()

    def _on_load_model_clicked(self) -> None:
        path = self.model_path_edit.text().strip()
        try:
            bundle = load_model_bundle(path)
            if self.connected and bundle.channel_count != self.channel_count:
                raise ValueError(
                    f"Model expects {bundle.channel_count} channels, but board currently has {self.channel_count}."
                )
            self.model_bundle = bundle
            self.inference_worker.set_model_bundle(bundle)
            self.loaded_model_label.setText(
                f"{Path(path).name} | {len(bundle.class_names)} classes | "
                f"{bundle.window_samples} win / {bundle.stride_samples} stride"
            )
            self._reset_inference_ring()
            self._append_log(f"Model loaded: {path}")
        except Exception as exc:  # pylint: disable=broad-except
            self._append_log(f"Failed to load model: {exc}")

        self._update_ui_state()

    def _on_predict_toggled(self, checked: bool) -> None:
        if checked:
            if not self.streaming:
                self._append_log("Cannot start prediction: stream is not running.")
                self.predict_toggle.setChecked(False)
                return
            if self.model_bundle is None:
                self._append_log("Cannot start prediction: no model loaded.")
                self.predict_toggle.setChecked(False)
                return
            self.prediction_enabled = True
            self._samples_since_submit = 0
            self.predict_toggle.setText("Stop Live Prediction")
            self._append_log("Live EEG prediction started.")
        else:
            self.prediction_enabled = False
            self.predict_toggle.setText("Start Live Prediction")
            self._append_log("Live EEG prediction stopped.")
        self._update_ui_state()

    def _on_prediction_ready(self, result: dict) -> None:
        self.pred_label.setText(str(result.get("label", "N/A")))
        self.conf_label.setText(f"{100.0 * float(result.get('confidence', 0.0)):.1f}%")
        self.latency_label.setText(f"{float(result.get('latency_ms', 0.0)):.1f} ms")

    def _on_prediction_error(self, message: str) -> None:
        self._append_log(f"Inference error: {message}")

    def _reset_inference_ring(self) -> None:
        capacity = max(int(self.sample_rate * 10), 512)
        self.infer_ring = RingBuffer(self.channel_count, capacity)
        self._samples_since_submit = 0

    def _update_ui_state(self) -> None:
        if self.mode == "data_collection":
            active_lock = self.protocol_running or self.prestart_running
            contributor_ready = self._contributor_unlocked()
            labels_ready = contributor_ready and self.labels_saved
            protocol_ready = labels_ready and self.labels_confirmed
            record_ready = protocol_ready and self.protocol_confirmed

            self._refresh_class_labels_display()
            self._refresh_folder_name_preview()

            self.contributor_edit.setEnabled(not active_lock)
            self.contributor_age_edit.setEnabled(not active_lock)
            self.contributor_sex_edit.setEnabled(not active_lock)
            self.agree_yes.setEnabled(not active_lock)
            self.agree_no.setEnabled(not active_lock)
            self.terms_button.setEnabled(not active_lock)

            self.protocol_labels_edit.setEnabled(False)
            self.edit_labels_button.setEnabled(labels_ready and not active_lock)
            self.labels_proceed_button.setEnabled(labels_ready and not active_lock)

            self.protocol_prep_spin.setEnabled(protocol_ready and not active_lock)
            self.protocol_hold_spin.setEnabled(protocol_ready and not active_lock)
            self.protocol_rest_spin.setEnabled(protocol_ready and not active_lock)
            self.protocol_repeats_spin.setEnabled(protocol_ready and not active_lock)
            self.record_rest_check.setEnabled(protocol_ready and not active_lock)
            self.proceed_button.setEnabled(protocol_ready and not active_lock)

            self.folder_name_edit.setEnabled(record_ready and not active_lock)
            self.open_record_session_button.setEnabled(record_ready and not active_lock)
            self.load_collection_config_button.setEnabled(not active_lock)
            self.save_collection_config_button.setEnabled(not active_lock)

            self._set_dimmed_enabled(self.class_label_group, contributor_ready and not active_lock, self.class_label_opacity)
            self._set_dimmed_enabled(self.protocol_group_box, protocol_ready and not active_lock, self.protocol_opacity)
            self._set_dimmed_enabled(self.record_group, record_ready or active_lock, self.record_opacity)

            if self.prestart_running:
                self.protocol_status_label.setText("Protocol: Countdown running...")
            elif self.protocol_running:
                pass
            elif self.protocol_confirmed:
                if self.streaming:
                    self.protocol_status_label.setText("Protocol: Ready | Open Start Recording")
                else:
                    self.protocol_status_label.setText("Protocol: Ready | Start EEG stream")
            elif self.labels_confirmed:
                self.protocol_status_label.setText("Protocol: Review & Proceed")
            else:
                self.protocol_status_label.setText("Protocol: Idle")

            self.collection_instruction_edit.setText(self._collection_instruction_text())
        else:
            self.protocol_toggle.setEnabled(self.streaming)
            self.protocol_labels_edit.setEnabled(not self.protocol_running)
            self.protocol_prep_spin.setEnabled(not self.protocol_running)
            self.protocol_hold_spin.setEnabled(not self.protocol_running)
            self.protocol_rest_spin.setEnabled(not self.protocol_running)
            self.protocol_repeats_spin.setEnabled(not self.protocol_running)
        self.train_button.setEnabled(self.train_worker is None or not self.train_worker.isRunning())
        self.predict_toggle.setEnabled(self.streaming and self.model_bundle is not None)
        status = f"Connected={self.connected} | Streaming={self.streaming} | SR={self.sample_rate} | CH={self.channel_count}"
        self.status_label.setText(f"Status: {status}")
        if self.protocol_running:
            self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {THEME_COLORS['success']}; background: transparent;")
        elif self.connected:
            self.status_label.setStyleSheet("font-size: 16px; font-weight: 700; background: transparent;")
        else:
            self.status_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {THEME_COLORS['muted']}; background: transparent;")
        self._update_recording_session_window()

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.metrics_box.appendPlainText(f"[{timestamp}] {message}")
        self.logger.info("EEG-ML: %s", message)

    def _browse_collect_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Dataset Root",
            self.collect_root_edit.text().strip(),
        )
        if path:
            self.collect_root_edit.setText(path)

    def _browse_train_input_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Input Dataset CSV",
            self.train_input_edit.text().strip(),
            "CSV Files (*.csv)",
        )
        if path:
            self.train_input_edit.setText(path)

    def _browse_train_input_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Input Dataset Folder",
            self.train_input_edit.text().strip(),
        )
        if path:
            self.train_input_edit.setText(path)

    def _browse_model_output_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Model Root Folder",
            self.model_output_root_edit.text().strip(),
        )
        if path:
            self.model_output_root_edit.setText(path)
            self.run_folder_edit.setText(path)

    def _browse_run_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Training Run Folder",
            self.run_folder_edit.text().strip(),
        )
        if path:
            self.run_folder_edit.setText(path)

    def _browse_model_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Model",
            self.model_path_edit.text().strip(),
            "Joblib Files (*.joblib)",
        )
        if path:
            self.model_path_edit.setText(path)

    def shutdown(self) -> None:
        try:
            if self.prestart_running:
                self.prestart_running = False
                self.prestart_timer.stop()
            if self.protocol_running:
                self._stop_protocol("Protocol stopped.")
            if self.collection_data_path is not None:
                self._finalize_collection_session("Collection finalized on shutdown.")
            if self.predict_toggle.isChecked():
                self.predict_toggle.setChecked(False)
            if self.inference_worker is not None:
                self.inference_worker.stop()
            if self.recording_session_window is not None:
                self.recording_session_window.hide()
        except Exception:  # pylint: disable=broad-except
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.hide()
        self.closed.emit()
        event.ignore()
