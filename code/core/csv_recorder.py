from __future__ import annotations

import csv
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class AsyncCsvBatchWriter:
    """Background CSV writer that drains queued row batches on a dedicated thread."""

    _STOP = object()

    def __init__(self, path: str | Path, header: list[str], mode: str = "w") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._queue: queue.Queue[object] = queue.Queue()
        self._rows_written = 0
        self._error: Exception | None = None
        self._closed = False

        if mode == "w" or self.path.stat().st_size == 0:
            self._writer.writerow(list(header))
            self._handle.flush()

        self._thread = threading.Thread(target=self._run, name=f"csv-writer-{self.path.name}", daemon=True)
        self._thread.start()

    @property
    def rows_written(self) -> int:
        return int(self._rows_written)

    def submit_rows(self, rows: list[list[object]]) -> None:
        self._raise_if_failed()
        if self._closed or not rows:
            return
        self._queue.put(rows)

    def close(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        try:
            self._handle.flush()
        finally:
            self._handle.close()
        self._raise_if_failed()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        return
                    rows = item
                    if rows:
                        self._writer.writerows(rows)
                        self._rows_written += len(rows)
                        self._handle.flush()
                finally:
                    self._queue.task_done()
        except Exception as exc:  # pylint: disable=broad-except
            self._error = exc
            try:
                while True:
                    self._queue.get_nowait()
                    self._queue.task_done()
            except queue.Empty:
                pass

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"CSV writer thread failed for {self.path}: {self._error}") from self._error


class CsvRecorder:
    """Write streamed EEG samples into a CSV file with one row per sample."""

    def __init__(self, path: str, channel_count: int) -> None:
        self.path = Path(path)
        self.channel_count = int(channel_count)
        header = ["utc_iso", "board_timestamp", "sample_index"]
        header.extend(f"ch{i + 1}" for i in range(self.channel_count))
        self._writer_thread = AsyncCsvBatchWriter(self.path, header=header, mode="w")
        self._rows_enqueued = 0

    @property
    def rows_written(self) -> int:
        return int(self._rows_enqueued)

    def write_chunk(
        self,
        eeg_chunk: np.ndarray,
        timestamps: np.ndarray | None = None,
        sample_index: np.ndarray | None = None,
    ) -> None:
        if eeg_chunk.ndim != 2:
            raise ValueError("eeg_chunk must be [channels, samples]")
        if eeg_chunk.shape[0] != self.channel_count:
            raise ValueError("channel count mismatch while writing CSV chunk")

        chunk = np.asarray(eeg_chunk, dtype=np.float64)
        sample_count = chunk.shape[1]
        if sample_count <= 0:
            return

        channel_major = chunk.T
        utc_now = datetime.now(timezone.utc).isoformat()
        board_ts_values = [""] * sample_count
        if timestamps is not None:
            ts_arr = np.asarray(timestamps).reshape(-1)
            limit = min(sample_count, ts_arr.shape[0])
            board_ts_values[:limit] = [
                f"{float(value):.6f}" if np.isfinite(value) else ""
                for value in ts_arr[:limit]
            ]

        sample_values = [""] * sample_count
        if sample_index is not None:
            idx_arr = np.asarray(sample_index).reshape(-1)
            limit = min(sample_count, idx_arr.shape[0])
            sample_values[:limit] = [
                str(int(float(value))) if np.isfinite(value) else ""
                for value in idx_arr[:limit]
            ]

        rows = [
            [utc_now, board_ts_values[i], sample_values[i], *channel_major[i].tolist()]
            for i in range(sample_count)
        ]
        self._writer_thread.submit_rows(rows)
        self._rows_enqueued += sample_count

    def close(self) -> None:
        self._writer_thread.close()
