from __future__ import annotations

import numpy as np


EEG_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


def feature_names(channel_count: int) -> list[str]:
    names: list[str] = []
    for ch in range(int(channel_count)):
        ch_id = ch + 1
        names.extend(
            [
                f"ch{ch_id}_rms",
                f"ch{ch_id}_std",
                f"ch{ch_id}_line_length",
                f"ch{ch_id}_hjorth_activity",
                f"ch{ch_id}_hjorth_mobility",
                f"ch{ch_id}_hjorth_complexity",
                f"ch{ch_id}_spec_entropy",
            ]
        )
        for band_name, _, _ in EEG_BANDS:
            names.append(f"ch{ch_id}_abs_{band_name}")
        for band_name, _, _ in EEG_BANDS:
            names.append(f"ch{ch_id}_rel_{band_name}")

    names.extend(["global_rms_mean", "global_rms_std"])
    for band_name, _, _ in EEG_BANDS:
        names.append(f"global_rel_{band_name}_mean")
    return names


def extract_eeg_window_features(window: np.ndarray, sample_rate: int) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("window must be 2D [channels, samples]")
    n_ch, n_samples = arr.shape
    if n_ch <= 0 or n_samples <= 0:
        raise ValueError("window must contain channels and samples")

    arr = arr - np.mean(arr, axis=1, keepdims=True)
    if n_samples < 16:
        return np.zeros(len(feature_names(n_ch)), dtype=np.float32)

    win = np.hanning(n_samples).astype(np.float64)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate))
    eps = 1e-12

    features: list[float] = []
    rel_band_stack: list[list[float]] = []
    rms_vals = []

    for ch in range(n_ch):
        x = arr[ch]
        dx = np.diff(x)

        rms = float(np.sqrt(np.mean(np.square(x))))
        std = float(np.std(x))
        line_length = float(np.sum(np.abs(dx)))
        rms_vals.append(rms)

        var0 = float(np.var(x)) + eps
        var1 = float(np.var(dx)) + eps
        ddx = np.diff(dx) if dx.size > 1 else np.array([0.0], dtype=np.float64)
        var2 = float(np.var(ddx)) + eps
        mobility = float(np.sqrt(var1 / var0))
        complexity = float(np.sqrt(var2 / var1) / (mobility + eps))

        pxx = np.abs(np.fft.rfft(x * win)) ** 2
        valid_mask = (freqs >= 0.5) & (freqs <= 45.0)
        pxx_valid = pxx[valid_mask]
        total_power = float(np.sum(pxx_valid) + eps)

        if pxx_valid.size > 1:
            p_norm = pxx_valid / total_power
            spec_entropy = float(-np.sum(p_norm * np.log2(p_norm + eps)) / np.log2(p_norm.size + eps))
        else:
            spec_entropy = 0.0

        abs_band = []
        rel_band = []
        for _, lo, hi in EEG_BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            bp = float(np.sum(pxx[mask]))
            abs_band.append(bp)
            rel_band.append(bp / total_power)

        rel_band_stack.append(rel_band)
        features.extend([rms, std, line_length, var0, mobility, complexity, spec_entropy])
        features.extend(abs_band)
        features.extend(rel_band)

    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    features.append(float(np.mean(rms_arr)))
    features.append(float(np.std(rms_arr)))

    rel_band_arr = np.asarray(rel_band_stack, dtype=np.float64)
    if rel_band_arr.ndim == 2 and rel_band_arr.shape[1] == len(EEG_BANDS):
        features.extend(np.mean(rel_band_arr, axis=0).tolist())
    else:
        features.extend([0.0] * len(EEG_BANDS))

    feat = np.asarray(features, dtype=np.float64)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)

