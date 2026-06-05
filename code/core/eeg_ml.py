from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from core.csv_recorder import AsyncCsvBatchWriter
from core.eeg_features import extract_eeg_window_features, feature_names

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    joblib = None
    JOBLIB_IMPORT_ERROR = exc
else:
    JOBLIB_IMPORT_ERROR = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    RandomForestClassifier = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    StratifiedKFold = None
    cross_val_score = None
    train_test_split = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


DEFAULT_ML_DATA_DIR = "dataset"
DEFAULT_ML_MODEL_DIR = "trained_model"
DEFAULT_ML_MODEL_ARTIFACT = "eeg_realtime_model.joblib"
DEFAULT_ML_RUN_NAME = "eeg_training"


def _ensure_ml_dependencies() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError("scikit-learn is not installed. Install with `pip install scikit-learn`.") from SKLEARN_IMPORT_ERROR
    if JOBLIB_IMPORT_ERROR is not None:
        raise RuntimeError("joblib is not installed. Install with `pip install joblib`.") from JOBLIB_IMPORT_ERROR


def _sanitize_filename_token(text: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip())
    safe = safe.strip("_")
    return safe or fallback


def _to_project_relative_path(path: Path | str) -> str:
    abs_path = Path(path).expanduser().resolve()
    root = PROJECT_ROOT.resolve()
    try:
        rel = abs_path.relative_to(root)
    except ValueError:
        return str(abs_path)
    return rel.as_posix()


def _normalize_dataset_csv_paths(csv_path: str | None, csv_paths: list[str] | None) -> list[Path]:
    raw_inputs: list[str] = []
    if csv_path:
        raw_inputs.append(str(csv_path))
    raw_inputs.extend(str(x) for x in list(csv_paths or []) if str(x).strip())

    normalized: list[Path] = []
    seen: set[str] = set()
    for raw in raw_inputs:
        token = str(raw).strip()
        if not token:
            continue
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_dir():
            files = sorted(candidate.rglob("*.csv"))
            for file_path in files:
                key = str(file_path).lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(file_path)
            continue
        if candidate.is_file():
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                normalized.append(candidate)
            continue
        raise FileNotFoundError(f"Dataset path not found: {candidate}")
    return normalized


def _unique_run_dir(output_dir: str, run_name: str) -> Path:
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    else:
        output_root = output_root.resolve()
    safe_name = _sanitize_filename_token(run_name, DEFAULT_ML_RUN_NAME)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{safe_name}_{stamp}"
    run_dir = output_root / base
    suffix = 2
    while run_dir.exists():
        run_dir = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    return run_dir


@dataclass
class ModelBundle:
    artifact: dict

    @property
    def model(self):
        return self.artifact["model"]

    @property
    def class_names(self) -> list[str]:
        return [str(x) for x in self.artifact["class_names"]]

    @property
    def sample_rate(self) -> int:
        return int(self.artifact["sample_rate"])

    @property
    def window_samples(self) -> int:
        return int(self.artifact["window_samples"])

    @property
    def stride_samples(self) -> int:
        return int(self.artifact["stride_samples"])

    @property
    def channel_count(self) -> int:
        return int(self.artifact["channel_count"])


class LabeledEegRecorder:
    def __init__(self, path: str, channel_count: int, label: str, trial_id: str, phase: str = "") -> None:
        self.path = Path(path)
        self.channel_count = int(channel_count)
        self.label = str(label).strip()
        self.trial_id = str(trial_id).strip()
        self.phase = str(phase).strip()
        if not self.label:
            raise ValueError("Label is required for data collection.")
        if not self.trial_id:
            raise ValueError("Trial ID is required for data collection.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = ["utc_iso", "board_timestamp", "sample_index", "label", "trial_id", "phase"]
        header.extend(f"ch{i + 1}" for i in range(self.channel_count))
        self._writer_thread = AsyncCsvBatchWriter(self.path, header=header, mode="a")
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
            raise ValueError("channel count mismatch while writing labeled EEG chunk")
        sample_count = eeg_chunk.shape[1]
        if sample_count <= 0:
            return

        chunk = np.asarray(eeg_chunk, dtype=np.float64)
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
            [utc_now, board_ts_values[i], sample_values[i], self.label, self.trial_id, self.phase, *channel_major[i].tolist()]
            for i in range(sample_count)
        ]
        self._writer_thread.submit_rows(rows)
        self._rows_enqueued += sample_count

    def close(self) -> None:
        self._writer_thread.close()


def _find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def load_labeled_segments(csv_path: str) -> tuple[list[tuple[str, np.ndarray]], int]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV has no header row.")
        fieldnames = list(reader.fieldnames)

        label_key = _find_column(fieldnames, ["label", "Label"])
        trial_key = _find_column(fieldnames, ["trial_id", "Trial_ID", "trial"])
        if label_key is None or trial_key is None:
            raise ValueError("Dataset CSV must contain `label` and `trial_id` columns.")

        channel_cols = [name for name in fieldnames if name.lower().startswith("ch")]
        if not channel_cols:
            raise ValueError("Dataset CSV has no EEG channel columns (expected ch1..chN).")
        channel_cols = sorted(channel_cols, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or "0"))

        groups: dict[tuple[str, str], list[list[float]]] = {}
        for row in reader:
            label = str(row.get(label_key, "")).strip()
            trial_id = str(row.get(trial_key, "")).strip()
            if not label or not trial_id:
                continue
            values = []
            valid_row = True
            for col in channel_cols:
                raw = row.get(col, "")
                try:
                    values.append(float(raw))
                except Exception:
                    valid_row = False
                    break
            if not valid_row:
                continue
            key = (label, trial_id)
            groups.setdefault(key, []).append(values)

    segments: list[tuple[str, np.ndarray]] = []
    for (label, _trial_id), samples in groups.items():
        if not samples:
            continue
        arr = np.asarray(samples, dtype=np.float32)  # [samples, channels]
        segments.append((label, arr))

    if not segments:
        raise ValueError("No valid labeled samples found in dataset CSV.")
    return segments, len(channel_cols)


def build_dataset(
    segments: list[tuple[str, np.ndarray]],
    sample_rate: int,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for label, seq in segments:
        if seq.ndim != 2:
            continue
        n_samples, _n_channels = seq.shape
        if n_samples < window_samples:
            continue
        for start in range(0, n_samples - window_samples + 1, stride_samples):
            window = seq[start : start + window_samples, :].T  # [channels, samples]
            feats = extract_eeg_window_features(window, sample_rate)
            x_rows.append(feats)
            y_rows.append(label)

    if not x_rows:
        raise ValueError("No training windows generated. Add more data or reduce window size.")
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    return x, y


def train_eeg_model(
    sample_rate: int,
    window_ms: int,
    stride_ms: int,
    n_estimators: int,
    max_depth: int,
    test_size: float,
    random_seed: int,
    csv_path: str | None = None,
    csv_paths: list[str] | None = None,
    model_out_path: str | None = None,
    output_dir: str | None = None,
    run_name: str = DEFAULT_ML_RUN_NAME,
    model_filename: str = DEFAULT_ML_MODEL_ARTIFACT,
) -> dict:
    _ensure_ml_dependencies()

    sr = int(sample_rate)
    window_samples = max(16, int((float(window_ms) / 1000.0) * sr))
    stride_samples = max(1, int((float(stride_ms) / 1000.0) * sr))

    dataset_paths = _normalize_dataset_csv_paths(csv_path, csv_paths)
    if len(dataset_paths) == 0:
        raise ValueError("No dataset CSV files selected.")

    segments: list[tuple[str, np.ndarray]] = []
    segment_count_by_file: dict[str, int] = {}
    channel_count_by_file: dict[str, int] = {}
    for dataset_path in dataset_paths:
        file_segments, file_channel_count = load_labeled_segments(str(dataset_path))
        segment_count_by_file[str(dataset_path)] = int(len(file_segments))
        channel_count_by_file[str(dataset_path)] = int(file_channel_count)
        segments.extend(file_segments)

    if len(segments) == 0:
        raise ValueError("No valid segments found across selected dataset files.")

    unique_channel_counts = sorted(set(int(v) for v in channel_count_by_file.values()))
    if len(unique_channel_counts) != 1:
        details = ", ".join(
            f"{Path(path).name}:{count}" for path, count in list(channel_count_by_file.items())[:8]
        )
        if len(channel_count_by_file) > 8:
            details = f"{details}, +{len(channel_count_by_file) - 8} more"
        raise ValueError(
            "Channel count mismatch across selected datasets. "
            f"Detected: {details}"
        )
    channel_count = int(unique_channel_counts[0])

    x, y = build_dataset(segments, sr, window_samples, stride_samples)
    classes = sorted(str(v) for v in np.unique(y))
    if len(classes) < 2:
        raise ValueError("Need at least 2 labels/classes for training.")

    class_to_idx = {name: i for i, name in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[str(v)] for v in y], dtype=np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_idx,
        test_size=float(test_size),
        random_state=int(random_seed),
        stratify=y_idx,
    )

    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None if int(max_depth) <= 0 else int(max_depth),
        class_weight="balanced_subsample",
        random_state=int(random_seed),
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", rf),
        ]
    )

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_seconds = float(time.perf_counter() - t0)

    acc = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

    # CV for robustness estimate (limited by smallest class count)
    unique, counts = np.unique(y_idx, return_counts=True)
    min_class_count = int(np.min(counts)) if counts.size > 0 else 0
    cv_folds = max(2, min(5, min_class_count))
    cv_mean = 0.0
    cv_std = 0.0
    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(random_seed))
        cv_scores = cross_val_score(model, x, y_idx, cv=cv, scoring="accuracy", n_jobs=1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

    if output_dir is None:
        if model_out_path:
            mo = Path(model_out_path).expanduser()
            if mo.suffix.lower() == ".joblib":
                output_dir = str(mo.parent)
                if not run_name or run_name == DEFAULT_ML_RUN_NAME:
                    run_name = _sanitize_filename_token(mo.stem, DEFAULT_ML_RUN_NAME)
            else:
                output_dir = str(mo)
        else:
            output_dir = DEFAULT_ML_MODEL_DIR

    run_dir = _unique_run_dir(output_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_file_stem = _sanitize_filename_token(Path(model_filename).stem, "eeg_realtime_model")
    model_file = f"{model_file_stem}.joblib"
    out_path = run_dir / model_file
    created_at_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir_rel = _to_project_relative_path(run_dir)
    model_path_rel = _to_project_relative_path(out_path)
    csv_paths_rel = [_to_project_relative_path(path) for path in dataset_paths]

    setup_payload = {
        "created_at_text": created_at_text,
        "dataset_paths": csv_paths_rel,
        "output_dir": _to_project_relative_path(run_dir.parent),
        "run_name": _sanitize_filename_token(run_name, DEFAULT_ML_RUN_NAME),
        "window_ms": int(window_ms),
        "stride_ms": int(stride_ms),
        "window_samples": int(window_samples),
        "stride_samples": int(stride_samples),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "test_size": float(test_size),
        "random_seed": int(random_seed),
        "sample_rate_hz": int(sr),
        "input_channels": int(channel_count),
    }
    class_counts = np.bincount(y_idx, minlength=len(classes))
    results_payload = {
        "created_at_text": created_at_text,
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "num_features": int(x.shape[1]),
        "classes": classes,
        "class_window_counts": {classes[i]: int(class_counts[i]) for i in range(len(classes))},
        "segment_count_by_file": {
            _to_project_relative_path(path): count for path, count in segment_count_by_file.items()
        },
        "channel_count_by_file": {
            _to_project_relative_path(path): count for path, count in channel_count_by_file.items()
        },
        "input_channels": int(channel_count),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report_text": report,
        "classification_report_dict": report_dict,
        "model_file": model_file,
        "run_dir": run_dir_rel,
        "model_path": model_path_rel,
    }

    artifact = {
        "model": model,
        "class_names": classes,
        "sample_rate": sr,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "channel_count": int(channel_count),
        "feature_names": feature_names(channel_count),
        "feature_extractor": "core.eeg_features.extract_eeg_window_features",
        "created_at_unix": time.time(),
        "created_at_text": created_at_text,
        "run_dir": run_dir_rel,
        "setup": setup_payload,
        "metrics": {
            "accuracy": acc,
            "weighted_f1": weighted_f1,
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
        },
    }

    joblib.dump(artifact, out_path)

    setup_path = run_dir / "training_setup.json"
    results_path = run_dir / "training_results.json"
    report_path = run_dir / "classification_report.txt"
    summary_path = run_dir / "training_summary.txt"
    cm_csv_path = run_dir / "confusion_matrix.csv"

    with setup_path.open("w", encoding="utf-8") as handle:
        json.dump(setup_payload, handle, indent=2)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(report.rstrip("\n") + "\n")
    with cm_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted"] + classes)
        cm_rows = cm.astype(int).tolist()
        for idx, row in enumerate(cm_rows):
            writer.writerow([classes[idx]] + [int(v) for v in row])

    summary = (
        f"Run Folder: {run_dir_rel}\n"
        f"Model File: {model_path_rel}\n"
        f"Datasets: {len(dataset_paths)} file(s)\n"
        f"Accuracy: {acc:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"Train windows: {len(y_train)} | Test windows: {len(y_test)}\n"
        f"Window: {window_ms} ms ({window_samples} samples) | Stride: {stride_ms} ms ({stride_samples} samples)\n"
        f"Input channels: {channel_count}\n"
        f"Classes: {', '.join(classes)}"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(summary + "\n")

    return {
        "model_path": str(out_path),
        "run_dir": str(run_dir),
        "summary_text": summary,
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "train_windows": int(len(y_train)),
        "test_windows": int(len(y_test)),
        "feature_dim": int(x.shape[1]),
        "classes": classes,
        "train_seconds": train_seconds,
        "report": report,
        "confusion_matrix": cm.astype(int).tolist(),
        "channel_count": int(channel_count),
        "window_samples": window_samples,
        "stride_samples": stride_samples,
    }


def load_model_bundle(model_path: str) -> ModelBundle:
    _ensure_ml_dependencies()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    artifact = joblib.load(path)
    required = ["model", "class_names", "sample_rate", "window_samples", "stride_samples", "channel_count"]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"Invalid model artifact. Missing keys: {missing}")
    return ModelBundle(artifact=artifact)


def predict_from_window(bundle: ModelBundle, eeg_window: np.ndarray) -> dict:
    window = np.asarray(eeg_window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError("eeg_window must be 2D [channels, samples].")
    if window.shape[0] != bundle.channel_count:
        raise ValueError(f"Model expects {bundle.channel_count} channels, got {window.shape[0]}.")

    t0 = time.perf_counter()
    features = extract_eeg_window_features(window, bundle.sample_rate).reshape(1, -1)
    model = bundle.model
    classes = bundle.class_names

    pred_index = int(model.predict(features)[0])
    pred_index = max(0, min(pred_index, len(classes) - 1))
    pred_label = classes[pred_index]

    probabilities = np.zeros(len(classes), dtype=np.float32)
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(features)[0], dtype=np.float32)
            if probs.shape[0] == len(classes):
                probabilities[:] = probs
                confidence = float(np.max(probs))
        except Exception:
            probabilities[pred_index] = 1.0
            confidence = 1.0
    else:
        probabilities[pred_index] = 1.0
        confidence = 1.0

    latency_ms = float((time.perf_counter() - t0) * 1000.0)
    return {
        "label": pred_label,
        "class_index": pred_index,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "latency_ms": latency_ms,
    }
