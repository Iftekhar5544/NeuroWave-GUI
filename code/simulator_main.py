from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

from PyQt5 import QtGui, QtWidgets

from app_theme import THEME_COLORS, apply_dark_theme, apply_dark_title_bar, themed_button_style, themed_label_style
from config import PROJECT_ROOT


os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, LogLevels
except ImportError as exc:  # pragma: no cover
    BoardIds = None
    BoardShim = None
    BrainFlowInputParams = None
    LogLevels = None
    BRAINFLOW_IMPORT_ERROR = exc
else:
    BRAINFLOW_IMPORT_ERROR = None


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("neurowave-sim")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


class SimulatorWindow(QtWidgets.QMainWindow):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger
        self.board = None
        self.streaming = False
        self.setWindowTitle("NeuroWave-EEG CSV Replay Streamer")
        self._build_ui()
        apply_dark_title_bar(self)
        hint = self.minimumSizeHint()
        self.setMinimumSize(max(720, hint.width() + 28), max(300, hint.height() + 24))
        self.resize(max(760, hint.width() + 40), max(340, hint.height() + 30))
        self._set_status("Idle")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        card = QtWidgets.QFrame()
        card.setProperty("card", True)
        card.setFrameShape(QtWidgets.QFrame.NoFrame)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(14)

        header = QtWidgets.QLabel("CSV EEG Replay Stream")
        header.setStyleSheet("font-size: 22px; font-weight: 700;")
        subheader = QtWidgets.QLabel("Select a recorded EEG CSV and replay it through the main app as a continuous live-like stream.")
        subheader.setWordWrap(True)
        subheader.setStyleSheet(themed_label_style("muted"))
        card_layout.addWidget(header)
        card_layout.addWidget(subheader)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        default_csv = self._default_csv_path()
        self.csv_path_edit = QtWidgets.QLineEdit(str(default_csv) if default_csv is not None else "")
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_csv)
        csv_row = QtWidgets.QHBoxLayout()
        csv_row.setContentsMargins(0, 0, 0, 0)
        csv_row.setSpacing(8)
        csv_row.addWidget(self.csv_path_edit, 1)
        csv_row.addWidget(self.browse_button)
        csv_row_widget = QtWidgets.QWidget()
        csv_row_widget.setObjectName("csvPathRow")
        csv_row_widget.setStyleSheet("QWidget#csvPathRow { background: transparent; border: none; }")
        csv_row_widget.setLayout(csv_row)
        self.sr_label = QtWidgets.QLabel("Replay sample rate: 125 Hz | Loop: continuous")
        form.addRow("Data CSV", csv_row_widget)
        form.addRow("Source", self.sr_label)
        card_layout.addLayout(form)

        self.endpoint_label = QtWidgets.QLabel("")
        card_layout.addWidget(self.endpoint_label)

        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Use CSV Replay")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.setStyleSheet(themed_button_style("accent"))
        self.stop_button.setStyleSheet(themed_button_style("danger"))
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        card_layout.addLayout(row)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-weight: 700;")
        card_layout.addWidget(self.status_label)

        self.tip_label = QtWidgets.QLabel(
            "In the main app, click Refresh and select the csv:// endpoint from the COM dropdown, then Connect and Start."
        )
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet(themed_label_style("muted"))
        card_layout.addWidget(self.tip_label)
        layout.addWidget(card)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self._refresh_endpoint_label()
        self.csv_path_edit.textChanged.connect(self._refresh_endpoint_label)

    def _refresh_endpoint_label(self) -> None:
        endpoint = self._csv_endpoint()
        self.endpoint_label.setText(f"Endpoint: {endpoint}" if endpoint else "Endpoint: Select a CSV file")
        self.endpoint_label.setStyleSheet(f"font-weight: 700; color: {THEME_COLORS['success']};")

    def _default_csv_path(self) -> Path | None:
        data_dir = PROJECT_ROOT / "code" / "streamer_data"
        if not data_dir.is_dir():
            return None
        files = sorted(data_dir.glob("*.csv"), key=lambda p: p.name.lower())
        return files[0].resolve() if files else None

    def _csv_endpoint(self) -> str:
        raw = self.csv_path_edit.text().strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        try:
            rel_path = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            rel_path = str(path)
        return f"csv://{rel_path}"

    def browse_csv(self) -> None:
        start_dir = str((PROJECT_ROOT / "code" / "streamer_data").resolve())
        current = self.csv_path_edit.text().strip()
        if current:
            current_path = Path(current).expanduser()
            if current_path.exists():
                start_dir = str(current_path.parent if current_path.is_file() else current_path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select EEG CSV Replay File",
            start_dir,
            "CSV Files (*.csv)",
        )
        if path:
            self.csv_path_edit.setText(path)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")
        if message.lower().startswith("error"):
            self.status_label.setStyleSheet(themed_label_style("danger"))
        elif "stream" in message.lower():
            self.status_label.setStyleSheet(themed_label_style("success"))
        else:
            self.status_label.setStyleSheet("font-weight: 700;")

    def start_stream(self) -> None:
        if self.streaming:
            return
        raw = self.csv_path_edit.text().strip()
        if not raw:
            self._set_status("Error: Select a CSV file first.")
            return
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            self._set_status(f"Error: CSV file not found: {path}")
            return

        self.logger.info("CSV replay endpoint selected: %s", self._csv_endpoint())
        self.streaming = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.csv_path_edit.setEnabled(False)
        self.browse_button.setEnabled(False)
        self._set_status("CSV replay ready. Select the csv:// endpoint in the main app.")

    def stop_stream(self) -> None:
        if not self.streaming:
            return
        self.logger.info("Stopping simulator stream.")
        try:
            if self.board is not None:
                self.board.stop_stream()
        except Exception:
            pass
        try:
            if self.board is not None:
                self.board.release_session()
        except Exception:
            pass
        self.board = None
        self.streaming = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.csv_path_edit.setEnabled(True)
        self.browse_button.setEnabled(True)
        self._set_status("Stopped")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_stream()
        event.accept()


def main() -> int:
    logger = _build_logger()
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app, font_size=16)
    icon_path = Path(__file__).resolve().parent / "images" / "app_icon.png"
    if icon_path.exists():
        icon = QtGui.QIcon(str(icon_path))
        if not icon.isNull():
            app.setWindowIcon(icon)
    window = SimulatorWindow(logger=logger)
    if icon_path.exists():
        icon = QtGui.QIcon(str(icon_path))
        if not icon.isNull():
            window.setWindowIcon(icon)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
