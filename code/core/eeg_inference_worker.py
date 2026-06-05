from __future__ import annotations

import threading

import numpy as np
from PyQt5 import QtCore

from core.eeg_ml import ModelBundle, predict_from_window


class EEGInferenceWorker(QtCore.QThread):
    prediction_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._lock = threading.Lock()
        self._pending = threading.Event()
        self._bundle: ModelBundle | None = None
        self._window: np.ndarray | None = None

    def set_model_bundle(self, bundle: ModelBundle | None) -> None:
        with self._lock:
            self._bundle = bundle
            self._window = None
        self._pending.clear()

    def submit_window(self, window: np.ndarray) -> None:
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            return
        with self._lock:
            self._window = arr
        self._pending.set()

    def stop(self) -> None:
        self._running = False
        self._pending.set()
        self.wait(1500)

    def run(self) -> None:
        self._running = True
        while self._running:
            self._pending.wait(0.2)
            if not self._running:
                break
            if not self._pending.is_set():
                continue
            self._pending.clear()

            with self._lock:
                bundle = self._bundle
                window = self._window
                self._window = None

            if bundle is None or window is None:
                continue

            try:
                result = predict_from_window(bundle, window)
                self.prediction_ready.emit(result)
            except Exception as exc:  # pylint: disable=broad-except
                self.error.emit(str(exc))

