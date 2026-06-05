from __future__ import annotations

import logging
import os
import sys
import warnings
import ctypes
from logging.handlers import RotatingFileHandler
from pathlib import Path


os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("neurowave")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler("neurowave.log", maxBytes=512_000, backupCount=2)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main() -> int:
    logger = configure_logging()

    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NeuroWave.EEGMonitor")
        except Exception:
            pass

    try:
        from PyQt5 import QtGui, QtWidgets
    except ImportError as exc:
        print("PyQt5 is not installed. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        logger.exception("PyQt5 import failed.")
        return 1

    try:
        import pyqtgraph  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print("Missing runtime dependency. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        logger.exception("Runtime dependency import failed.")
        return 1

    from app_theme import apply_dark_theme
    from ui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app, font_size=16)
    icon_path = Path(__file__).resolve().parent / "images" / "app_icon.png"
    if icon_path.exists():
        app_icon = QtGui.QIcon(str(icon_path))
        app.setWindowIcon(app_icon)
    else:
        app_icon = QtGui.QIcon()
    window = MainWindow(logger=logger)
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
