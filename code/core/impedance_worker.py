from __future__ import annotations

import math
import time

import numpy as np
from PyQt5 import QtCore


CYTON_DAISY_IMPEDANCE_TOKENS = ["1", "2", "3", "4", "5", "6", "7", "8", "Q", "W", "E", "R", "T", "Y", "U", "I"]
IMPEDANCE_TEST_FREQ_HZ = 31.5
IMPEDANCE_TEST_CURRENT_A = 6e-9
ADS_INPUT_RESISTANCE_KOHM = 2.2


class ImpedanceCheckWorker(QtCore.QThread):
    channel_started = QtCore.pyqtSignal(int)
    channel_result = QtCore.pyqtSignal(int, float, str, float)
    channel_error = QtCore.pyqtSignal(int, str)
    status = QtCore.pyqtSignal(str)
    finished_cleanly = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        board_service,
        channels: list[int] | None = None,
        sample_seconds: float = 5.0,
        settle_seconds: float = 0.45,
        good_threshold_kohm: float = 25.0,
        ok_threshold_kohm: float = 100.0,
        input_mode: str = "n",
    ) -> None:
        super().__init__()
        self.board_service = board_service
        self.channels = list(channels or [])
        self.sample_seconds = float(sample_seconds)
        self.settle_seconds = float(settle_seconds)
        self.good_threshold_kohm = float(good_threshold_kohm)
        self.ok_threshold_kohm = float(ok_threshold_kohm)
        self.input_mode = str(input_mode or "n").strip().lower()
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        stream_started = False
        active_token = ""
        try:
            if not self.board_service.is_impedance_supported():
                raise RuntimeError("Impedance checking is only supported for Cyton/Cyton+Daisy hardware.")
            eeg_channels = self.board_service.get_eeg_channels()
            sample_rate = self.board_service.get_sample_rate()
            if not self.channels:
                self.channels = list(range(min(len(eeg_channels), len(CYTON_DAISY_IMPEDANCE_TOKENS))))

            for channel_index in self.channels:
                if not self._running:
                    break
                if channel_index < 0 or channel_index >= len(eeg_channels):
                    self.channel_error.emit(channel_index, "Channel is not available on this board.")
                    continue
                if channel_index >= len(CYTON_DAISY_IMPEDANCE_TOKENS):
                    self.channel_error.emit(channel_index, "No impedance command token for this channel.")
                    continue

                token = CYTON_DAISY_IMPEDANCE_TOKENS[channel_index]
                p_flag, n_flag = self._input_mode_flags()
                active_token = token
                self.channel_started.emit(channel_index)
                self.status.emit(f"Testing CH{channel_index + 1}...")

                try:
                    self.board_service.config_board(f"z{token}{p_flag}{n_flag}Z")
                    time.sleep(max(0.0, self.settle_seconds))
                    self.board_service.start_stream()
                    stream_started = True
                    time.sleep(max(0.5, self.sample_seconds))
                    sample_count = max(1, int(sample_rate * self.sample_seconds))
                    board_data = self.board_service.get_current_data(sample_count)
                    self.board_service.stop_stream()
                    stream_started = False
                    self.board_service.config_board(f"z{token}00Z")
                    active_token = ""

                    eeg_row = int(eeg_channels[channel_index])
                    values = np.asarray(board_data[eeg_row, :], dtype=np.float64)
                    impedance_kohm, signal_uv_rms = estimate_impedance_kohm(values, sample_rate)
                    quality = impedance_quality_label(
                        impedance_kohm,
                        good_threshold_kohm=self.good_threshold_kohm,
                        ok_threshold_kohm=self.ok_threshold_kohm,
                    )
                    self.channel_result.emit(channel_index, impedance_kohm, quality, signal_uv_rms)
                except Exception as exc:  # pylint: disable=broad-except
                    if stream_started:
                        try:
                            self.board_service.stop_stream()
                        except Exception:
                            pass
                        stream_started = False
                    if active_token:
                        try:
                            self.board_service.config_board(f"z{active_token}00Z")
                        except Exception:
                            pass
                        active_token = ""
                    self.channel_error.emit(channel_index, str(exc))

            self._disable_all_impedance_channels()
            self.finished_cleanly.emit()
        except Exception as exc:  # pylint: disable=broad-except
            if stream_started:
                try:
                    self.board_service.stop_stream()
                except Exception:
                    pass
            if active_token:
                try:
                    self.board_service.config_board(f"z{active_token}00Z")
                except Exception:
                    pass
            self._disable_all_impedance_channels()
            self.failed.emit(str(exc))
        finally:
            self._running = False

    def _disable_all_impedance_channels(self) -> None:
        for token in CYTON_DAISY_IMPEDANCE_TOKENS:
            try:
                self.board_service.config_board(f"z{token}00Z")
            except Exception:
                pass
        try:
            self.board_service.restore_all_channels_on()
        except Exception:
            pass

    def _input_mode_flags(self) -> tuple[str, str]:
        if self.input_mode == "p":
            return "1", "0"
        if self.input_mode == "both":
            return "1", "1"
        return "0", "1"


def estimate_impedance_kohm(samples_uv: np.ndarray, sample_rate: int) -> tuple[float, float]:
    values = np.asarray(samples_uv, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < max(32, int(sample_rate * 0.5)):
        raise ValueError("Not enough samples for impedance estimate.")

    values = values - float(np.mean(values))
    t = np.arange(values.size, dtype=np.float64) / float(sample_rate)
    sin_ref = np.sin(2.0 * math.pi * IMPEDANCE_TEST_FREQ_HZ * t)
    cos_ref = np.cos(2.0 * math.pi * IMPEDANCE_TEST_FREQ_HZ * t)
    sin_amp = 2.0 * float(np.dot(values, sin_ref)) / float(values.size)
    cos_amp = 2.0 * float(np.dot(values, cos_ref)) / float(values.size)
    peak_uv = math.sqrt((sin_amp * sin_amp) + (cos_amp * cos_amp))
    signal_uv_rms = peak_uv / math.sqrt(2.0)

    impedance_ohm = (math.sqrt(2.0) * signal_uv_rms * 1e-6) / IMPEDANCE_TEST_CURRENT_A
    impedance_kohm = max(0.0, (impedance_ohm / 1000.0) - ADS_INPUT_RESISTANCE_KOHM)
    return impedance_kohm, signal_uv_rms


def impedance_quality_label(
    impedance_kohm: float,
    good_threshold_kohm: float = 25.0,
    ok_threshold_kohm: float = 100.0,
) -> str:
    value = float(impedance_kohm)
    good_limit = float(good_threshold_kohm)
    ok_limit = max(good_limit, float(ok_threshold_kohm))
    if value <= good_limit:
        return "Good"
    if value <= ok_limit:
        return "OK"
    return "Poor"
