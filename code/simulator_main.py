from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

from PyQt5 import QtGui, QtWidgets

from app_theme import THEME_COLORS, apply_dark_theme, apply_dark_title_bar, themed_button_style, themed_label_style
from config import DEFAULT_SIM_MULTICAST_IP, DEFAULT_SIM_MULTICAST_PORT


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
        self.setWindowTitle("NeuroWave-EEG Simulator")
        self._build_ui()
        apply_dark_title_bar(self)
        hint = self.minimumSizeHint()
        self.setMinimumSize(max(620, hint.width() + 28), max(260, hint.height() + 24))
        self.resize(max(640, hint.width() + 40), max(300, hint.height() + 30))
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

        header = QtWidgets.QLabel("Synthetic EEG Stream")
        header.setStyleSheet("font-size: 22px; font-weight: 700;")
        subheader = QtWidgets.QLabel("Use this when the Cyton board is not in hand and you want the main EEG app to keep running.")
        subheader.setWordWrap(True)
        subheader.setStyleSheet(themed_label_style("muted"))
        card_layout.addWidget(header)
        card_layout.addWidget(subheader)

        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        self.host_edit = QtWidgets.QLineEdit(DEFAULT_SIM_MULTICAST_IP)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(int(DEFAULT_SIM_MULTICAST_PORT))
        self.sr_label = QtWidgets.QLabel("Synthetic Board (default BrainFlow sample rate)")
        form.addRow("Host", self.host_edit)
        form.addRow("Port", self.port_spin)
        form.addRow("Source", self.sr_label)
        card_layout.addLayout(form)

        self.endpoint_label = QtWidgets.QLabel("")
        card_layout.addWidget(self.endpoint_label)

        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Simulator Stream")
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
            "In main app COM dropdown, select endpoint:\n"
            "sim://<host>:<port>  (example shown above)."
        )
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet(themed_label_style("muted"))
        card_layout.addWidget(self.tip_label)
        layout.addWidget(card)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self._refresh_endpoint_label()
        self.host_edit.textChanged.connect(self._refresh_endpoint_label)
        self.port_spin.valueChanged.connect(self._refresh_endpoint_label)

    def _refresh_endpoint_label(self) -> None:
        host = self.host_edit.text().strip() or DEFAULT_SIM_MULTICAST_IP
        port = int(self.port_spin.value())
        self.endpoint_label.setText(f"Endpoint: sim://{host}:{port}")
        self.endpoint_label.setStyleSheet(f"font-weight: 700; color: {THEME_COLORS['success']};")

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
        if BRAINFLOW_IMPORT_ERROR is not None:
            self._set_status("BrainFlow not installed.")
            return

        host = self.host_edit.text().strip() or DEFAULT_SIM_MULTICAST_IP
        port = int(self.port_spin.value())
        streamer = f"streaming_board://{host}:{port}"
        self.logger.info("Starting synthetic stream on %s", streamer)

        try:
            BoardShim.set_log_level(LogLevels.LEVEL_ERROR.value)
        except Exception:
            pass

        try:
            params = BrainFlowInputParams()
            self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
            self.board.prepare_session()
            self.board.start_stream(45000, streamer)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("Failed to start simulator stream.")
            self._set_status(f"Error: {exc}")
            try:
                if self.board is not None:
                    self.board.release_session()
            except Exception:
                pass
            self.board = None
            return

        self.streaming = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.host_edit.setEnabled(False)
        self.port_spin.setEnabled(False)
        self._set_status(f"Streaming synthetic EEG on sim://{host}:{port}")

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
        self.host_edit.setEnabled(True)
        self.port_spin.setEnabled(True)
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
