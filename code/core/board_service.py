from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import List

import numpy as np

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams, LogLevels
except ImportError as exc:  # pragma: no cover - import guard for environments without dependencies
    BoardIds = None
    BoardShim = None
    BrainFlowError = Exception
    BrainFlowInputParams = None
    LogLevels = None
    BRAINFLOW_IMPORT_ERROR = exc
else:
    BRAINFLOW_IMPORT_ERROR = None

from config import DEFAULT_CHANNEL_COUNT, DEFAULT_SAMPLE_RATE, DEFAULT_STREAM_BUFFER_SIZE, PROJECT_ROOT


class BoardService:
    """Thin wrapper around BrainFlow board lifecycle management."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.board = None
        self.board_id = None
        self.connected = False
        self.streaming = False
        self._eeg_channels: List[int] = []
        self._sample_rate = 0
        self._timestamp_channel: int | None = None
        self._package_channel: int | None = None
        self._connection_name = "Unknown"
        self._csv_replay_path: Path | None = None
        self._csv_replay_data: np.ndarray | None = None
        self._csv_replay_timestamps: np.ndarray | None = None
        self._csv_replay_sample_index: np.ndarray | None = None
        self._csv_replay_cursor = 0
        self._csv_replay_emitted = 0
        self._csv_replay_started_at = 0.0

    def connect(self, port: str) -> None:
        if self.connected:
            raise RuntimeError("Board is already connected.")
        if not port.strip():
            raise ValueError("A serial COM port is required.")

        csv_path = self._parse_csv_stream_url(port.strip())
        if csv_path is not None:
            self._connect_csv_replay(csv_path)
            return

        self._ensure_brainflow()
        params = BrainFlowInputParams()
        connection_name = "Cyton + Daisy"
        board_id = BoardIds.CYTON_DAISY_BOARD.value
        sim_endpoint = self._parse_sim_stream_url(port.strip())
        if sim_endpoint is None:
            params.serial_port = port.strip()
        else:
            host, port_num = sim_endpoint
            params.ip_address = host
            params.ip_port = int(port_num)
            params.master_board = BoardIds.SYNTHETIC_BOARD.value
            board_id = BoardIds.STREAMING_BOARD.value
            connection_name = "Simulator Stream (Synthetic)"

        try:
            BoardShim.set_log_level(LogLevels.LEVEL_ERROR.value)
        except Exception:
            self.logger.debug("Could not lower BrainFlow board logger verbosity.")

        if sim_endpoint is None:
            self.logger.info("Preparing BrainFlow session on port %s", params.serial_port)
        else:
            self.logger.info("Preparing BrainFlow streaming session on %s:%s", params.ip_address, params.ip_port)
        try:
            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            resolved_board_id = int(self.board.get_board_id())
            self.board_id = resolved_board_id
            self._eeg_channels = BoardShim.get_eeg_channels(resolved_board_id)
            self._sample_rate = BoardShim.get_sampling_rate(resolved_board_id)
            self._timestamp_channel = self._safe_get_channel(BoardShim.get_timestamp_channel, resolved_board_id)
            self._package_channel = self._safe_get_channel(BoardShim.get_package_num_channel, resolved_board_id)
            self._connection_name = connection_name
        except BrainFlowError as exc:
            self.logger.exception("Failed to connect to the board.")
            self.board = None
            self.board_id = None
            if sim_endpoint is None:
                raise RuntimeError(f"Could not connect to board on {params.serial_port}: {exc}") from exc
            raise RuntimeError(
                f"Could not connect to simulator stream {sim_endpoint[0]}:{sim_endpoint[1]}: {exc}"
            ) from exc

        self.connected = True
        self.streaming = False
        self.logger.info(
            "Connected to %s with %s EEG channels at %s Hz",
            self._connection_name,
            len(self._eeg_channels),
            self._sample_rate,
        )

    def start_stream(self) -> None:
        if not self.connected:
            raise RuntimeError("Connect to the board before starting the stream.")
        if self.streaming:
            raise RuntimeError("Stream is already running.")
        if self._is_csv_replay():
            self._csv_replay_started_at = time.perf_counter()
            self._csv_replay_emitted = 0
            self.streaming = True
            self.logger.info("CSV replay stream started.")
            return
        if self.board is None:
            raise RuntimeError("Connect to the board before starting the stream.")

        try:
            self.board.start_stream(DEFAULT_STREAM_BUFFER_SIZE)
        except BrainFlowError as exc:
            self.logger.exception("Failed to start the stream.")
            raise RuntimeError(f"Failed to start stream: {exc}") from exc

        self.streaming = True
        self.logger.info("Board stream started.")

    def stop_stream(self) -> None:
        if not self.connected:
            return
        if not self.streaming:
            return
        if self._is_csv_replay():
            self.streaming = False
            self.logger.info("CSV replay stream stopped.")
            return
        if self.board is None:
            return

        try:
            self.board.stop_stream()
            self.logger.info("Board stream stopped.")
        except BrainFlowError as exc:
            self.logger.exception("Failed to stop the stream cleanly.")
            raise RuntimeError(f"Failed to stop stream: {exc}") from exc
        finally:
            self.streaming = False

    def disconnect(self) -> None:
        if self._is_csv_replay():
            self.connected = False
            self.streaming = False
            self.board = None
            self.board_id = None
            self._eeg_channels = []
            self._sample_rate = 0
            self._timestamp_channel = None
            self._package_channel = None
            self._connection_name = "Unknown"
            self._csv_replay_path = None
            self._csv_replay_data = None
            self._csv_replay_timestamps = None
            self._csv_replay_sample_index = None
            self._csv_replay_cursor = 0
            self._csv_replay_emitted = 0
            self._csv_replay_started_at = 0.0
            self.logger.info("CSV replay session released.")
            return

        if self.board is None:
            self.connected = False
            self.streaming = False
            return

        stream_error = None
        if self.streaming:
            try:
                self.stop_stream()
            except RuntimeError as exc:
                stream_error = exc

        try:
            self.board.release_session()
            self.logger.info("Board session released.")
        except BrainFlowError as exc:
            self.logger.exception("Failed to release the board session cleanly.")
            raise RuntimeError(f"Failed to release board session: {exc}") from exc
        finally:
            self.board = None
            self.board_id = None
            self.connected = False
            self.streaming = False
            self._eeg_channels = []
            self._sample_rate = 0
            self._timestamp_channel = None
            self._package_channel = None
            self._connection_name = "Unknown"

        if stream_error is not None:
            raise stream_error

    def get_eeg_channels(self) -> list[int]:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        return list(self._eeg_channels)

    def get_sample_rate(self) -> int:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        return int(self._sample_rate)

    def get_timestamp_channel(self) -> int | None:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        return self._timestamp_channel

    def get_package_channel(self) -> int | None:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        return self._package_channel

    def get_connection_name(self) -> str:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        return str(self._connection_name)

    def is_impedance_supported(self) -> bool:
        if not self.connected or self.board_id is None or BoardIds is None:
            return False
        return int(self.board_id) in {
            int(BoardIds.CYTON_BOARD.value),
            int(BoardIds.CYTON_DAISY_BOARD.value),
        }

    def config_board(self, command: str) -> str:
        if not self.connected or self.board is None:
            raise RuntimeError("Board is not connected.")
        try:
            result = self.board.config_board(str(command))
        except UnicodeError as exc:
            self.logger.warning(
                "Board config command response could not be decoded after command %s: %s",
                command,
                exc,
            )
            return ""
        except BrainFlowError as exc:
            self.logger.exception("Board config command failed.")
            raise RuntimeError(f"Board config command failed: {exc}") from exc
        return "" if result is None else str(result)

    def restore_all_channels_on(self) -> None:
        if not self.is_impedance_supported():
            return
        # Restore normal electrode input, 24x gain, bias enabled, SRB2 on, SRB1 off.
        # This mirrors Cyton's typical default channel configuration and clears
        # any unusual state left after lead-off impedance commands.
        channel_tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "Q", "W", "E", "R", "T", "Y", "U", "I"]
        channel_on_commands = ["!", "@", "#", "$", "%", "^", "&", "*", "Q", "W", "E", "R", "T", "Y", "U", "I"]
        channel_limit = max(0, min(len(self._eeg_channels), len(channel_tokens)))
        for token in channel_tokens[:channel_limit]:
            try:
                self.config_board(f"x{token}060110X")
            except Exception:
                self.logger.debug("Failed to restore default channel settings for %s", token, exc_info=True)
        # Cyton channel-on commands. 1-8 use punctuation; Daisy 9-16 use uppercase letters.
        for command in channel_on_commands[:channel_limit]:
            try:
                self.config_board(command)
            except Exception:
                self.logger.debug("Failed to restore channel with command %s", command, exc_info=True)

    def get_current_data(self, sample_count: int):
        if not self.connected or self.board is None:
            raise RuntimeError("Board is not connected.")
        if not self.streaming:
            raise RuntimeError("Stream is not running.")

        try:
            return self.board.get_current_board_data(sample_count)
        except BrainFlowError as exc:
            self.logger.exception("Failed to read current board data.")
            raise RuntimeError(f"Failed to read board data: {exc}") from exc

    def get_available_sample_count(self) -> int:
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        if not self.streaming:
            raise RuntimeError("Stream is not running.")
        if self._is_csv_replay():
            elapsed = max(0.0, time.perf_counter() - self._csv_replay_started_at)
            should_have_emitted = int(elapsed * float(self._sample_rate))
            return max(0, should_have_emitted - int(self._csv_replay_emitted))
        if self.board is None:
            raise RuntimeError("Board is not connected.")

        try:
            return int(self.board.get_board_data_count())
        except BrainFlowError as exc:
            self.logger.exception("Failed to get available board sample count.")
            raise RuntimeError(f"Failed to inspect board data count: {exc}") from exc

    def get_pending_data(self, sample_count: int):
        if not self.connected:
            raise RuntimeError("Board is not connected.")
        if not self.streaming:
            raise RuntimeError("Stream is not running.")
        if self._is_csv_replay():
            return self._read_csv_replay_chunk(sample_count)
        if self.board is None:
            raise RuntimeError("Board is not connected.")

        try:
            return self.board.get_board_data(sample_count)
        except BrainFlowError as exc:
            self.logger.exception("Failed to drain pending board data.")
            raise RuntimeError(f"Failed to read board data: {exc}") from exc

    def _ensure_brainflow(self) -> None:
        if BRAINFLOW_IMPORT_ERROR is not None:
            raise RuntimeError(
                "BrainFlow is not installed. Run `pip install -r requirements.txt` first."
            ) from BRAINFLOW_IMPORT_ERROR

    def _safe_get_channel(self, getter, board_id: int) -> int | None:
        try:
            return int(getter(board_id))
        except Exception:
            return None

    def _parse_sim_stream_url(self, value: str) -> tuple[str, int] | None:
        text = str(value).strip()
        if not text.lower().startswith("sim://"):
            return None
        endpoint = text[len("sim://") :].strip()
        if not endpoint:
            raise ValueError("Simulator endpoint is empty. Expected format sim://host:port")
        if " " in endpoint:
            endpoint = endpoint.split(" ", 1)[0].strip()
        if ":" not in endpoint:
            raise ValueError("Invalid simulator endpoint. Expected format sim://host:port")
        host, port_text = endpoint.rsplit(":", 1)
        host = host.strip()
        if not host:
            raise ValueError("Invalid simulator endpoint host.")
        try:
            port_num = int(port_text.strip())
        except Exception as exc:
            raise ValueError("Invalid simulator endpoint port.") from exc
        if port_num <= 0:
            raise ValueError("Simulator endpoint port must be positive.")
        return host, port_num

    def _parse_csv_stream_url(self, value: str) -> Path | None:
        text = str(value).strip()
        if not text.lower().startswith("csv://"):
            return None
        raw_path = text[len("csv://") :].strip().strip('"')
        if not raw_path:
            raise ValueError("CSV replay endpoint is empty. Expected format csv://path/to/eeg.csv")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        return path

    def _connect_csv_replay(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"CSV replay file not found: {path}")
        eeg_data, timestamps, sample_index = self._load_csv_replay_file(path)
        self.board = None
        self.board_id = None
        self._csv_replay_path = path
        self._csv_replay_data = eeg_data
        self._csv_replay_timestamps = timestamps
        self._csv_replay_sample_index = sample_index
        self._csv_replay_cursor = 0
        self._csv_replay_emitted = 0
        self._csv_replay_started_at = 0.0
        self._eeg_channels = list(range(eeg_data.shape[0]))
        self._sample_rate = int(DEFAULT_SAMPLE_RATE)
        self._timestamp_channel = int(eeg_data.shape[0])
        self._package_channel = int(eeg_data.shape[0] + 1)
        self._connection_name = f"CSV Replay ({path.name})"
        self.connected = True
        self.streaming = False
        self.logger.info(
            "Connected to CSV replay file %s with %s EEG channels at %s Hz",
            path,
            eeg_data.shape[0],
            self._sample_rate,
        )

    def _load_csv_replay_file(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV replay file has no header row.")
            channel_cols = [name for name in reader.fieldnames if name.lower().startswith("ch")]
            channel_cols = sorted(channel_cols, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or "0"))
            if len(channel_cols) < 1:
                raise ValueError("CSV replay file has no EEG channel columns named ch1..chN.")
            if len(channel_cols) != DEFAULT_CHANNEL_COUNT:
                self.logger.warning(
                    "CSV replay has %s EEG columns; app default is %s.",
                    len(channel_cols),
                    DEFAULT_CHANNEL_COUNT,
                )

            rows: list[list[float]] = []
            timestamps: list[float] = []
            sample_index: list[float] = []
            for row_num, row in enumerate(reader):
                values = []
                for col in channel_cols:
                    values.append(float(row.get(col, "nan")))
                if not all(np.isfinite(values)):
                    continue
                rows.append(values)
                try:
                    timestamps.append(float(row.get("board_timestamp", "")))
                except Exception:
                    timestamps.append(row_num / float(DEFAULT_SAMPLE_RATE))
                try:
                    sample_index.append(float(row.get("sample_index", row_num)))
                except Exception:
                    sample_index.append(float(row_num))

        if not rows:
            raise ValueError("CSV replay file has no valid EEG samples.")
        eeg = np.asarray(rows, dtype=np.float64).T
        ts = np.asarray(timestamps, dtype=np.float64)
        idx = np.asarray(sample_index, dtype=np.float64)
        return eeg, ts, idx

    def _is_csv_replay(self) -> bool:
        return self._csv_replay_data is not None

    def _read_csv_replay_chunk(self, sample_count: int) -> np.ndarray:
        if self._csv_replay_data is None:
            raise RuntimeError("CSV replay data is not loaded.")
        count = max(0, int(sample_count))
        channel_count, total_samples = self._csv_replay_data.shape
        if count <= 0 or total_samples <= 0:
            return np.zeros((channel_count + 2, 0), dtype=np.float64)

        indices = (np.arange(count, dtype=np.int64) + int(self._csv_replay_cursor)) % int(total_samples)
        eeg_chunk = self._csv_replay_data[:, indices]
        ts_source = self._csv_replay_timestamps
        idx_source = self._csv_replay_sample_index
        if ts_source is not None and ts_source.size == total_samples:
            timestamp_chunk = ts_source[indices].reshape(1, -1)
        else:
            start = int(self._csv_replay_emitted)
            timestamp_chunk = ((np.arange(count, dtype=np.float64) + start) / float(self._sample_rate)).reshape(1, -1)
        if idx_source is not None and idx_source.size == total_samples:
            sample_index_chunk = idx_source[indices].reshape(1, -1)
        else:
            sample_index_chunk = (np.arange(count, dtype=np.float64) + int(self._csv_replay_emitted)).reshape(1, -1)

        self._csv_replay_cursor = int((self._csv_replay_cursor + count) % total_samples)
        self._csv_replay_emitted += count
        return np.vstack([eeg_chunk, timestamp_chunk, sample_index_chunk])
