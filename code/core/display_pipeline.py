from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from config import (
    DISPLAY_AUTOSCALE_DECAY,
    DISPLAY_AUTOSCALE_MARGIN,
    DISPLAY_AUTOSCALE_MIN_UV,
    DISPLAY_FILTER_HISTORY_SECONDS,
    DISPLAY_FILTER_ORDER,
    DISPLAY_FIXED_SCALE_UV,
    DISPLAY_HIGHPASS_HZ,
    DISPLAY_LOWPASS_HZ,
    DISPLAY_NOTCH_FREQ_HZ,
)

try:
    from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
except ImportError as exc:  # pragma: no cover - runtime guard
    DataFilter = None
    FilterTypes = None
    NoiseTypes = None
    DATAFILTER_IMPORT_ERROR = exc
else:
    DATAFILTER_IMPORT_ERROR = None


@dataclass
class FilterConfig:
    enabled: bool = True
    preset_name: str = "Raw"
    scope: str = "all"  # all | selected
    selected_channels: list[int] = field(default_factory=list)
    notch_enabled: bool = False
    notch_freq: int = int(DISPLAY_NOTCH_FREQ_HZ)
    notch_q: float = 30.0
    highpass_enabled: bool = False
    highpass_hz: float = float(DISPLAY_HIGHPASS_HZ)
    highpass_order: int = int(DISPLAY_FILTER_ORDER)
    lowpass_enabled: bool = False
    lowpass_hz: float = float(DISPLAY_LOWPASS_HZ)
    lowpass_order: int = int(DISPLAY_FILTER_ORDER)
    bandpass_enabled: bool = False
    bandpass_low_hz: float = 8.0
    bandpass_high_hz: float = 13.0
    bandpass_order: int = int(DISPLAY_FILTER_ORDER)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "FilterConfig":
        data = dict(payload or {})
        data.setdefault("selected_channels", [])
        data.setdefault("scope", "all")
        data.setdefault("preset_name", "Custom")
        return cls(**data)


@dataclass
class ActiveFilterPlan:
    config: FilterConfig
    apply_mask: np.ndarray
    channel_ops: list[list[dict]]
    summary: str


class DisplayPipeline:
    """Display-only chunk processor using a compiled per-channel filter plan."""

    def __init__(self, sample_rate: int, channel_count: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self._history_len = 0
        self._history = np.zeros((self.channel_count, 1), dtype=np.float64)
        self._dc_tracker = np.zeros(self.channel_count, dtype=np.float64)
        self._auto_scales = np.full(self.channel_count, DISPLAY_FIXED_SCALE_UV, dtype=np.float64)
        self._peak_tracker = np.full(self.channel_count, DISPLAY_AUTOSCALE_MIN_UV, dtype=np.float64)
        self.active_plan = self.compile_plan(FilterConfig(), self.sample_rate, self.channel_count)
        self._reset_history()

    def reset(self, sample_rate: int, channel_count: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        self._reset_history()
        self._dc_tracker = np.zeros(self.channel_count, dtype=np.float64)
        self._auto_scales = np.full(self.channel_count, DISPLAY_FIXED_SCALE_UV, dtype=np.float64)
        self._peak_tracker = np.full(self.channel_count, DISPLAY_AUTOSCALE_MIN_UV, dtype=np.float64)
        self.active_plan = self.compile_plan(self.active_plan.config, self.sample_rate, self.channel_count)

    def set_active_plan(self, plan: ActiveFilterPlan) -> None:
        self.active_plan = plan
        self._reset_history()
        self._dc_tracker = np.zeros(self.channel_count, dtype=np.float64)
        self._peak_tracker = np.full(self.channel_count, DISPLAY_AUTOSCALE_MIN_UV, dtype=np.float64)

    def process_chunk(self, raw_chunk: np.ndarray) -> np.ndarray:
        if raw_chunk.ndim != 2:
            raise ValueError("raw_chunk must be 2D [channels, samples]")
        if raw_chunk.shape[0] != self.channel_count:
            raise ValueError("raw_chunk channel count does not match pipeline state")
        if raw_chunk.shape[1] == 0:
            return np.zeros_like(raw_chunk, dtype=np.float64)

        processed = np.array(raw_chunk, dtype=np.float64, copy=True, order="C")
        has_data_filter = DATAFILTER_IMPORT_ERROR is None
        for channel_index in range(self.channel_count):
            channel_data = processed[channel_index]
            if (
                has_data_filter
                and self.active_plan.config.enabled
                and self.active_plan.apply_mask[channel_index]
                and self.active_plan.channel_ops[channel_index]
            ):
                channel_data = self._process_channel_with_history(
                    channel_index,
                    channel_data,
                    self.active_plan.channel_ops[channel_index],
                )

            # Always center each displayed channel for stable zero-centered live traces.
            processed[channel_index] = self._center_channel(channel_index, channel_data)

        self._history = self._build_next_history(raw_chunk)
        return processed

    def update_auto_scales_from_chunk(self, processed_chunk: np.ndarray) -> np.ndarray:
        if processed_chunk.ndim != 2 or processed_chunk.shape[0] != self.channel_count:
            raise ValueError("processed_chunk must be 2D [channels, samples]")
        if processed_chunk.shape[1] == 0:
            return self._auto_scales.copy()

        chunk_peaks = np.max(np.abs(processed_chunk), axis=1)
        self._peak_tracker = np.maximum(chunk_peaks, self._peak_tracker * DISPLAY_AUTOSCALE_DECAY)
        targets = np.maximum(self._peak_tracker * DISPLAY_AUTOSCALE_MARGIN, DISPLAY_AUTOSCALE_MIN_UV)
        self._auto_scales = targets
        return self._auto_scales.copy()

    def get_fixed_scales(self) -> np.ndarray:
        return np.full(self.channel_count, DISPLAY_FIXED_SCALE_UV, dtype=np.float64)

    @staticmethod
    def compile_plan(config: FilterConfig, sample_rate: int, channel_count: int) -> ActiveFilterPlan:
        sr = float(sample_rate)
        nyquist = sr / 2.0
        errors: list[str] = []

        cfg = FilterConfig.from_dict(config.to_dict())
        cfg.scope = (cfg.scope or "all").strip().lower()
        if cfg.scope not in {"all", "selected"}:
            errors.append("Apply Scope must be All Channels or Selected Channels.")

        if cfg.notch_freq not in (50, 60):
            errors.append("Notch frequency must be 50 or 60.")
        if cfg.notch_q <= 0:
            errors.append("Notch Q must be greater than 0.")

        if cfg.highpass_enabled and cfg.highpass_hz <= 0:
            errors.append("High-pass cutoff must be greater than 0.")
        if cfg.lowpass_enabled and cfg.lowpass_hz <= 0:
            errors.append("Low-pass cutoff must be greater than 0.")
        if cfg.bandpass_enabled and (cfg.bandpass_low_hz <= 0 or cfg.bandpass_high_hz <= 0):
            errors.append("Band-pass low and high cutoffs must be greater than 0.")
        if cfg.bandpass_enabled and cfg.bandpass_low_hz >= cfg.bandpass_high_hz:
            errors.append("Band-pass low cutoff must be lower than high cutoff.")

        if cfg.highpass_enabled and cfg.highpass_hz >= nyquist:
            errors.append("High-pass cutoff must be below Nyquist.")
        if cfg.lowpass_enabled and cfg.lowpass_hz >= nyquist:
            errors.append("Low-pass cutoff must be below Nyquist.")
        if cfg.bandpass_enabled and cfg.bandpass_low_hz >= nyquist:
            errors.append("Band-pass low cutoff must be below Nyquist.")
        if cfg.bandpass_enabled and cfg.bandpass_high_hz >= nyquist:
            errors.append("Band-pass high cutoff must be below Nyquist.")

        if cfg.highpass_order not in (2, 4):
            errors.append("High-pass order must be 2 or 4.")
        if cfg.lowpass_order not in (2, 4):
            errors.append("Low-pass order must be 2 or 4.")
        if cfg.bandpass_order not in (2, 4):
            errors.append("Band-pass order must be 2 or 4.")

        selected = sorted(set(int(i) for i in cfg.selected_channels))
        selected = [i for i in selected if 0 <= i < int(channel_count)]
        cfg.selected_channels = selected

        if cfg.scope == "selected" and not cfg.selected_channels:
            errors.append("Select at least one channel when Apply Scope is Selected Channels.")

        if errors:
            raise ValueError("\n".join(errors))

        if cfg.scope == "all":
            apply_mask = np.ones(int(channel_count), dtype=bool)
        else:
            apply_mask = np.zeros(int(channel_count), dtype=bool)
            apply_mask[cfg.selected_channels] = True

        base_ops: list[dict] = []
        if cfg.enabled:
            if cfg.highpass_enabled:
                base_ops.append({"type": "highpass", "cutoff": float(cfg.highpass_hz), "order": int(cfg.highpass_order)})
            if cfg.notch_enabled:
                base_ops.append({"type": "notch", "freq": int(cfg.notch_freq), "q": float(cfg.notch_q), "order": 2})
            if cfg.lowpass_enabled:
                base_ops.append({"type": "lowpass", "cutoff": float(cfg.lowpass_hz), "order": int(cfg.lowpass_order)})
            if cfg.bandpass_enabled:
                base_ops.append(
                    {
                        "type": "bandpass",
                        "low": float(cfg.bandpass_low_hz),
                        "high": float(cfg.bandpass_high_hz),
                        "order": int(cfg.bandpass_order),
                    }
                )

        channel_ops: list[list[dict]] = []
        for ch in range(int(channel_count)):
            channel_ops.append(list(base_ops) if apply_mask[ch] else [])

        scope_label = "All" if cfg.scope == "all" else f"Selected({len(cfg.selected_channels)})"
        if not cfg.enabled:
            chain = "OFF"
        else:
            parts = []
            if cfg.highpass_enabled:
                parts.append(f"HP({cfg.highpass_hz:g})")
            if cfg.notch_enabled:
                parts.append(f"Notch({cfg.notch_freq},Q={cfg.notch_q:g})")
            if cfg.lowpass_enabled:
                parts.append(f"LP({cfg.lowpass_hz:g})")
            if cfg.bandpass_enabled:
                parts.append(f"BP({cfg.bandpass_low_hz:g}-{cfg.bandpass_high_hz:g})")
            chain = "-".join(parts) if parts else "Raw"
        summary = f"Filters: {chain} | Scope: {scope_label}"
        return ActiveFilterPlan(config=cfg, apply_mask=apply_mask, channel_ops=channel_ops, summary=summary)

    def _reset_history(self) -> None:
        self._history_len = max(64, int(self.sample_rate * DISPLAY_FILTER_HISTORY_SECONDS))
        self._history = np.zeros((self.channel_count, self._history_len), dtype=np.float64)

    def _build_next_history(self, raw_chunk: np.ndarray) -> np.ndarray:
        chunk_len = raw_chunk.shape[1]
        if chunk_len >= self._history_len:
            return np.array(raw_chunk[:, -self._history_len :], dtype=np.float64, copy=True, order="C")

        history = np.zeros((self.channel_count, self._history_len), dtype=np.float64)
        if self._history.shape == history.shape:
            history[:, : self._history_len - chunk_len] = self._history[:, chunk_len:]
        history[:, -chunk_len:] = raw_chunk
        return history

    def _process_channel_with_history(self, channel_index: int, chunk: np.ndarray, operations: list[dict]) -> np.ndarray:
        history = self._history[channel_index]
        merged = np.concatenate((history, chunk)).astype(np.float64, copy=False)
        merged = np.ascontiguousarray(merged, dtype=np.float64)

        for op in operations:
            op_type = op["type"]
            if op_type == "highpass":
                self._call_filter(
                    DataFilter.perform_highpass,
                    merged,
                    self.sample_rate,
                    float(op["cutoff"]),
                    int(op["order"]),
                    FilterTypes.BUTTERWORTH.value,
                    0.0,
                )
            elif op_type == "notch":
                freq = float(op["freq"])
                q = max(0.001, float(op["q"]))
                band_width = max(0.1, freq / q)
                try:
                    nyquist = (self.sample_rate / 2.0) - 0.1
                    half_bw = band_width / 2.0
                    start_freq = max(0.1, freq - half_bw)
                    stop_freq = min(nyquist, freq + half_bw)
                    if stop_freq <= start_freq:
                        raise ValueError("Invalid notch band edges for current sample rate.")
                    self._call_filter(
                        DataFilter.perform_bandstop,
                        merged,
                        self.sample_rate,
                        start_freq,
                        stop_freq,
                        int(op.get("order", 2)),
                        FilterTypes.BUTTERWORTH.value,
                        0.0,
                    )
                except Exception:
                    noise_type = NoiseTypes.FIFTY.value if int(freq) == 50 else NoiseTypes.SIXTY.value
                    self._call_filter(
                        DataFilter.remove_environmental_noise,
                        merged,
                        self.sample_rate,
                        noise_type,
                    )
            elif op_type == "lowpass":
                self._call_filter(
                    DataFilter.perform_lowpass,
                    merged,
                    self.sample_rate,
                    float(op["cutoff"]),
                    int(op["order"]),
                    FilterTypes.BUTTERWORTH.value,
                    0.0,
                )
            elif op_type == "bandpass":
                self._call_filter(
                    DataFilter.perform_bandpass,
                    merged,
                    self.sample_rate,
                    float(op["low"]),
                    float(op["high"]),
                    int(op["order"]),
                    FilterTypes.BUTTERWORTH.value,
                    0.0,
                )

        filtered_chunk = merged[-chunk.size :]
        return filtered_chunk

    def _center_channel(self, channel_index: int, channel: np.ndarray) -> np.ndarray:
        centered = np.asarray(channel, dtype=np.float64).copy()
        target = float(np.median(centered))
        self._dc_tracker[channel_index] = 0.99 * self._dc_tracker[channel_index] + 0.01 * target
        centered -= self._dc_tracker[channel_index]
        return centered

    @staticmethod
    def _call_filter(filter_func, data: np.ndarray, *args) -> None:
        result = filter_func(data, *args)
        if result is not None:
            data[:] = result
