from __future__ import annotations

import logging
import time

import numpy as np
from PyQt5 import QtCore


class StreamWorker(QtCore.QThread):
    """Poll BrainFlow data in the background and emit EEG-only chunks."""

    data_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    state_changed = QtCore.pyqtSignal(str)

    def __init__(self, board_service, poll_interval_ms: int, chunk_size: int, logger: logging.Logger | None = None):
        super().__init__()
        self.board_service = board_service
        self.poll_interval_ms = poll_interval_ms
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger(__name__)
        self._running = False

    def run(self) -> None:
        self._running = True
        self.state_changed.emit("streaming")
        self.logger.info("Stream worker started with poll interval %sms", self.poll_interval_ms)

        while self._running:
            try:
                available_samples = self.board_service.get_available_sample_count()
                if available_samples <= 0:
                    time.sleep(self.poll_interval_ms / 1000.0)
                    continue

                board_data = self.board_service.get_pending_data(min(self.chunk_size, available_samples))
                eeg_channels = self.board_service.get_eeg_channels()
                eeg_data = np.asarray(board_data[eeg_channels, :], dtype=np.float64)
                if eeg_data.size > 0 and eeg_data.shape[1] > 0:
                    timestamp_channel = self.board_service.get_timestamp_channel()
                    package_channel = self.board_service.get_package_channel()

                    timestamps = None
                    sample_index = None
                    if timestamp_channel is not None and timestamp_channel < board_data.shape[0]:
                        timestamps = np.asarray(board_data[timestamp_channel, :], dtype=np.float64).copy()
                    if package_channel is not None and package_channel < board_data.shape[0]:
                        sample_index = np.asarray(board_data[package_channel, :], dtype=np.float64).copy()

                    payload = {
                        "eeg": eeg_data.copy(),
                        "timestamps": timestamps,
                        "sample_index": sample_index,
                    }
                    self.data_ready.emit(payload)
            except RuntimeError as exc:
                self.logger.exception("Stream worker failed while reading board data.")
                self.error.emit(str(exc))
                break

            time.sleep(self.poll_interval_ms / 1000.0)

        self._running = False
        self.state_changed.emit("stopped")
        self.logger.info("Stream worker stopped.")

    def stop(self) -> None:
        self._running = False
