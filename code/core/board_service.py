from __future__ import annotations

import logging
from typing import List

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

from config import DEFAULT_STREAM_BUFFER_SIZE


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

    def connect(self, port: str) -> None:
        self._ensure_brainflow()
        if self.connected:
            raise RuntimeError("Board is already connected.")
        if not port.strip():
            raise ValueError("A serial COM port is required.")

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
        if not self.connected or self.board is None:
            raise RuntimeError("Connect to the board before starting the stream.")
        if self.streaming:
            raise RuntimeError("Stream is already running.")

        try:
            self.board.start_stream(DEFAULT_STREAM_BUFFER_SIZE)
        except BrainFlowError as exc:
            self.logger.exception("Failed to start the stream.")
            raise RuntimeError(f"Failed to start stream: {exc}") from exc

        self.streaming = True
        self.logger.info("Board stream started.")

    def stop_stream(self) -> None:
        if not self.connected or self.board is None:
            return
        if not self.streaming:
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
        if not self.connected or self.board is None:
            raise RuntimeError("Board is not connected.")
        if not self.streaming:
            raise RuntimeError("Stream is not running.")

        try:
            return int(self.board.get_board_data_count())
        except BrainFlowError as exc:
            self.logger.exception("Failed to get available board sample count.")
            raise RuntimeError(f"Failed to inspect board data count: {exc}") from exc

    def get_pending_data(self, sample_count: int):
        if not self.connected or self.board is None:
            raise RuntimeError("Board is not connected.")
        if not self.streaming:
            raise RuntimeError("Stream is not running.")

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
