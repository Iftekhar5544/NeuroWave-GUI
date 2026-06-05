from __future__ import annotations

import json
from pathlib import Path

from core.display_pipeline import FilterConfig


BUILTIN_PRESET_ORDER = [
    "Raw",
    "OpenBCI Default",
    "EEG Lab",
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
    "Custom",
]


class PresetStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._user_presets: dict[str, dict] = {}
        self._builtin_presets = self._build_builtin_presets()
        self.load()

    def _build_builtin_presets(self) -> dict[str, FilterConfig]:
        presets: dict[str, FilterConfig] = {}
        presets["Raw"] = FilterConfig(
            enabled=True,
            preset_name="Raw",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=False,
        )
        presets["OpenBCI Default"] = FilterConfig(
            enabled=True,
            preset_name="OpenBCI Default",
            highpass_enabled=True,
            highpass_hz=1.0,
            highpass_order=2,
            notch_enabled=True,
            notch_freq=50,
            notch_q=30.0,
            lowpass_enabled=True,
            lowpass_hz=45.0,
            lowpass_order=2,
            bandpass_enabled=False,
        )
        presets["EEG Lab"] = FilterConfig(
            enabled=True,
            preset_name="EEG Lab",
            highpass_enabled=True,
            highpass_hz=1.0,
            highpass_order=2,
            notch_enabled=True,
            notch_freq=50,
            notch_q=30.0,
            lowpass_enabled=True,
            lowpass_hz=40.0,
            lowpass_order=2,
            bandpass_enabled=False,
        )
        # Backward-compatible alias for older configs.
        presets["EEG Default"] = FilterConfig(
            enabled=True,
            preset_name="EEG Default",
            highpass_enabled=True,
            highpass_hz=1.0,
            highpass_order=2,
            notch_enabled=True,
            notch_freq=50,
            notch_q=30.0,
            lowpass_enabled=True,
            lowpass_hz=40.0,
            lowpass_order=2,
            bandpass_enabled=False,
        )
        presets["Delta"] = FilterConfig(
            enabled=True,
            preset_name="Delta",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=True,
            bandpass_low_hz=0.5,
            bandpass_high_hz=4.0,
            bandpass_order=2,
        )
        presets["Theta"] = FilterConfig(
            enabled=True,
            preset_name="Theta",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=True,
            bandpass_low_hz=4.0,
            bandpass_high_hz=8.0,
            bandpass_order=2,
        )
        presets["Alpha"] = FilterConfig(
            enabled=True,
            preset_name="Alpha",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=True,
            bandpass_low_hz=8.0,
            bandpass_high_hz=13.0,
            bandpass_order=2,
        )
        presets["Beta"] = FilterConfig(
            enabled=True,
            preset_name="Beta",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=True,
            bandpass_low_hz=13.0,
            bandpass_high_hz=30.0,
            bandpass_order=2,
        )
        presets["Gamma"] = FilterConfig(
            enabled=True,
            preset_name="Gamma",
            highpass_enabled=False,
            lowpass_enabled=False,
            notch_enabled=False,
            bandpass_enabled=True,
            bandpass_low_hz=30.0,
            bandpass_high_hz=45.0,
            bandpass_order=2,
        )
        presets["Custom"] = FilterConfig(
            enabled=True,
            preset_name="Custom",
            highpass_enabled=True,
            highpass_hz=1.0,
            notch_enabled=True,
            notch_freq=50,
            notch_q=30.0,
            lowpass_enabled=True,
            lowpass_hz=40.0,
            bandpass_enabled=False,
        )
        return presets

    def load(self) -> None:
        if not self.path.exists():
            self._user_presets = {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._user_presets = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
            else:
                self._user_presets = {}
        except Exception:
            self._user_presets = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._user_presets, indent=2), encoding="utf-8")

    def list_preset_names(self) -> list[str]:
        builtins = BUILTIN_PRESET_ORDER[:]
        users = sorted(self._user_presets.keys(), key=lambda x: x.lower())
        return builtins + users

    def is_builtin(self, name: str) -> bool:
        return name in self._builtin_presets

    def get(self, name: str) -> FilterConfig | None:
        if name in self._builtin_presets:
            return FilterConfig.from_dict(self._builtin_presets[name].to_dict())
        payload = self._user_presets.get(name)
        if payload is None:
            return None
        cfg = FilterConfig.from_dict(payload)
        cfg.preset_name = name
        return cfg

    def save_user_preset(self, name: str, config: FilterConfig) -> None:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Preset name cannot be empty.")
        if clean_name in self._builtin_presets:
            raise ValueError("This name is reserved by a built-in preset.")
        payload = config.to_dict()
        payload["preset_name"] = clean_name
        self._user_presets[clean_name] = payload
        self.save()

    def delete_user_preset(self, name: str) -> None:
        if name in self._builtin_presets:
            raise ValueError("Built-in presets cannot be deleted.")
        if name in self._user_presets:
            del self._user_presets[name]
            self.save()
