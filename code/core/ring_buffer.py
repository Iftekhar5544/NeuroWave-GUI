from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer storing channel-major float samples."""

    def __init__(self, channels: int, capacity: int) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.channels = channels
        self.capacity = capacity
        self._buffer = np.zeros((channels, capacity), dtype=np.float64)
        self._write_index = 0
        self._filled = 0

    def append(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        if chunk.ndim != 2:
            raise ValueError("chunk must be a 2D array")
        if chunk.shape[0] != self.channels:
            raise ValueError("chunk channel count does not match the buffer")

        chunk = np.asarray(chunk, dtype=np.float64)
        sample_count = chunk.shape[1]

        if sample_count >= self.capacity:
            self._buffer[:, :] = chunk[:, -self.capacity :]
            self._write_index = 0
            self._filled = self.capacity
            return

        end_index = self._write_index + sample_count
        if end_index <= self.capacity:
            self._buffer[:, self._write_index:end_index] = chunk
        else:
            first_part = self.capacity - self._write_index
            self._buffer[:, self._write_index :] = chunk[:, :first_part]
            self._buffer[:, : end_index % self.capacity] = chunk[:, first_part:]

        self._write_index = end_index % self.capacity
        self._filled = min(self.capacity, self._filled + sample_count)

    def get_window(self) -> np.ndarray:
        if self._filled == 0:
            return np.zeros((self.channels, self.capacity), dtype=np.float64)
        if self._filled < self.capacity:
            padded = np.zeros((self.channels, self.capacity), dtype=np.float64)
            padded[:, -self._filled :] = self._buffer[:, : self._filled]
            return padded
        return np.concatenate(
            (self._buffer[:, self._write_index :], self._buffer[:, : self._write_index]),
            axis=1,
        )

    def clear(self) -> None:
        self._buffer.fill(0.0)
        self._write_index = 0
        self._filled = 0

    def get_filled_count(self) -> int:
        return int(self._filled)
