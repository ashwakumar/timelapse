#!/usr/bin/env python3
"""Build patient-disjoint VitalDB tensors from the top-N ranked signals.

The positional argument is intentionally the only feature-selection control:
``N`` always means the first N dynamic parameters in the saved reliable
parameter ranking. This makes top-5, top-10, and later runs reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from feature_screen import TARGET_TRACK, TRACKS, first_true_run_start, load_case


LABELS = ["within_3", "within_5", "within_10", "within_15", "none_within_15"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VitalDB LSTM tensors from the best N ranked dynamic parameters."
    )
    parser.add_argument(
        "num_parameters",
        type=int,
        help="Number of top-ranked dynamic parameters to include (for example 5 or 10).",
    )
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--history-sec", type=int, default=20)
    parser.add_argument("--horizon-sec", type=int, default=900)
    parser.add_argument("--stride-sec", type=int, default=60)
    parser.add_argument("--clean-history-sec", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--ranking-file",
        type=Path,
        default=Path("artifacts/feature_screen/reliable_parameter_ranking.csv"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/vitaldb_feature_screen")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/lstm"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("interval", "history_sec", "horizon_sec", "stride_sec", "clean_history_sec"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("history_sec", "horizon_sec", "stride_sec", "clean_history_sec"):
        if getattr(args, name) % args.interval:
            raise ValueError(f"--{name.replace('_', '-')} must be divisible by --interval")
    if args.horizon_sec != 900:
        raise ValueError("The five output classes require --horizon-sec 900")


def select_top_parameters(ranking_path: Path, count: int) -> tuple[list[str], list[dict]]:
    if count < 1:
        raise ValueError("num_parameters must be at least 1")
    ranking = pd.read_csv(ranking_path)
    required = {"parameter", "grouped_permutation_auprc_drop_mean"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Ranking file is missing columns: {sorted(required - set(ranking.columns))}")
    ranking = ranking[ranking["parameter"].isin(TRACKS)].copy()
    ranking = ranking.sort_values(
        "grouped_permutation_auprc_drop_mean", ascending=False, kind="stable"
    )
    if count > len(ranking):
        raise ValueError(
            f"Requested top {count}, but only {len(ranking)} reliable dynamic parameters exist. "
            "Re-run feature_screen.py with a larger cohort before requesting more."
        )
    selected_rows = ranking.head(count)
    return selected_rows["parameter"].tolist(), selected_rows.to_dict(orient="records")


def split_subjects(subject_ids: np.ndarray, seed: int) -> dict[str, set[int]]:
    subjects = np.asarray(sorted(set(int(value) for value in subject_ids)))
    if len(subjects) < 7:
        raise RuntimeError("At least seven subjects are required for train/validation/test splits")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    development_idx, test_idx = next(outer.split(subjects, groups=subjects))
    development = subjects[development_idx]
    test = subjects[test_idx]
    validation_fraction = 0.15 / 0.85
    inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + 1)
    train_idx, validation_idx = next(inner.split(development, groups=development))
    return {
        "train": set(int(value) for value in development[train_idx]),
        "validation": set(int(value) for value in development[validation_idx]),
        "test": set(int(value) for value in test),
    }


def onset_bucket(onset_sec: int | None) -> str:
    if onset_sec is None:
        return "none_within_15"
    if onset_sec <= 180:
        return "within_3"
    if onset_sec <= 300:
        return "within_5"
    if onset_sec <= 600:
        return "within_10"
    return "within_15"


def case_windows(
    frame: pd.DataFrame,
    parameters: list[str],
    case_id: int,
    subject_id: int,
    interval: int,
    history_sec: int,
    horizon_sec: int,
    stride_sec: int,
    clean_history_sec: int,
) -> list[dict]:
    history_n = history_sec // interval
    horizon_n = horizon_sec // interval
    stride_n = stride_sec // interval
    clean_n = clean_history_sec // interval
    required_low_n = 60 // interval
    windows: list[dict] = []
    first_cutoff = max(history_n, clean_n)

    for cutoff in range(first_cutoff, len(frame) - horizon_n + 1, stride_n):
        clean_map = frame["MAP"].iloc[cutoff - clean_n : cutoff].to_numpy(dtype=np.float32)
        # This creates a clean new-onset task: the preceding minute must prove
        # that the patient was observed and never below the target threshold.
        if not (np.isfinite(clean_map).all() and (clean_map >= 65).all()):
            continue

        history_map = frame["MAP"].iloc[cutoff - history_n : cutoff].to_numpy(dtype=np.float32)
        future_map = frame["MAP"].iloc[cutoff : cutoff + horizon_n].to_numpy(dtype=np.float32)
        if np.isfinite(history_map).mean() < 0.8 or np.isfinite(future_map).mean() < 0.8:
            continue

        # Missing values are False and therefore break, rather than bridge, a run.
        low = np.isfinite(future_map) & (future_map < 65)
        onset_index = first_true_run_start(low, required_low_n)
        # The first future sample is one sampling interval after the last input.
        onset_sec = None if onset_index is None else (onset_index + 1) * interval
        label = onset_bucket(onset_sec)
        values = frame[parameters].iloc[cutoff - history_n : cutoff].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        windows.append(
            {
                "values": values,
                "mask": mask,
                "label": LABEL_TO_INDEX[label],
                "case_id": case_id,
                "subject_id": subject_id,
                "cutoff_sec": cutoff * interval,
                "future_map_coverage": float(np.isfinite(future_map).mean()),
            }
        )
    return windows


def training_normalization(windows: list[dict], parameter_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise RuntimeError("The training split produced no eligible windows")
    values = np.concatenate([window["values"] for window in windows], axis=0)
    median = np.full(parameter_count, np.nan, dtype=np.float32)
    scale = np.full(parameter_count, np.nan, dtype=np.float32)
    for column in range(parameter_count):
        observed = values[np.isfinite(values[:, column]), column]
        if observed.size == 0:
            raise RuntimeError(f"Selected parameter {column} has no observed training values")
        median[column] = np.median(observed)
        q25, q75 = np.percentile(observed, [25, 75])
        iqr = q75 - q25
        scale[column] = iqr if np.isfinite(iqr) and iqr > 1e-6 else 1.0
    return median, scale


def save_split(
    path: Path,
    windows: list[dict],
    median: np.ndarray,
    scale: np.ndarray,
    history_n: int,
    parameter_count: int,
) -> dict:
    if windows:
        raw = np.stack([window["values"] for window in windows])
        mask = np.stack([window["mask"] for window in windows])
        normalized = np.where(mask, (raw - median) / scale, 0.0).astype(np.float32)
    else:
        normalized = np.empty((0, history_n, parameter_count), dtype=np.float32)
        mask = np.empty((0, history_n, parameter_count), dtype=bool)
    labels = np.asarray([window["label"] for window in windows], dtype=np.int8)
    case_ids = np.asarray([window["case_id"] for window in windows], dtype=np.int32)
    subject_ids = np.asarray([window["subject_id"] for window in windows], dtype=np.int32)
    cutoff_sec = np.asarray([window["cutoff_sec"] for window in windows], dtype=np.int32)
    coverage = np.asarray([window["future_map_coverage"] for window in windows], dtype=np.float32)
    np.savez_compressed(
        path,
        x_values=normalized,
        x_mask=mask,
        y=labels,
        case_id=case_ids,
        subject_id=subject_ids,
        cutoff_sec=cutoff_sec,
        future_map_coverage=coverage,
    )
    return {
        "windows": int(len(windows)),
        "subjects": int(len(set(subject_ids.tolist()))),
        "cases": int(len(set(case_ids.tolist()))),
        "label_counts": {LABELS[index]: int((labels == index).sum()) for index in range(len(LABELS))},
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    parameters, ranking_rows = select_top_parameters(args.ranking_file, args.num_parameters)
    if "MAP" not in parameters:
        raise RuntimeError("The selected ranking must include MAP to define input quality")

    case_ids = sorted(vitaldb.find_cases([TARGET_TRACK]))
    clinical = vitaldb.load_clinical_data(caseids=case_ids, params=["caseid", "subjectid"])
    subject_by_case = {
        int(case_id): int(subject_id)
        for case_id, subject_id in zip(clinical["caseid"], clinical["subjectid"])
        if pd.notna(subject_id)
    }
    eligible_cases = [case_id for case_id in case_ids if int(case_id) in subject_by_case]
    rng = np.random.default_rng(args.seed)
    selected_cases = rng.permutation(eligible_cases)[: min(args.max_cases, len(eligible_cases))]
    selected_subjects = np.asarray([subject_by_case[int(case_id)] for case_id in selected_cases])
    subjects_by_split = split_subjects(selected_subjects, args.seed)

    split_windows: dict[str, list[dict]] = {name: [] for name in subjects_by_split}
    failures: list[dict] = []
    split_for_subject = {
        subject: split for split, subjects in subjects_by_split.items() for subject in subjects
    }
    for case_id_value in tqdm(selected_cases, desc=f"Preparing top-{args.num_parameters} cases"):
        case_id = int(case_id_value)
        subject_id = subject_by_case[case_id]
        try:
            frame = load_case(case_id, args.interval, args.cache_dir)
            split_windows[split_for_subject[subject_id]].extend(
                case_windows(
                    frame,
                    parameters,
                    case_id,
                    subject_id,
                    args.interval,
                    args.history_sec,
                    args.horizon_sec,
                    args.stride_sec,
                    args.clean_history_sec,
                )
            )
        except Exception as error:
            failures.append({"case_id": case_id, "error": repr(error)})

    median, scale = training_normalization(split_windows["train"], len(parameters))
    output_dir = args.output_root / f"top_{args.num_parameters}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = {
        split: save_split(
            output_dir / f"{split}.npz",
            windows,
            median,
            scale,
            args.history_sec // args.interval,
            len(parameters),
        )
        for split, windows in split_windows.items()
    }
    availability = {}
    for split, windows in split_windows.items():
        if windows:
            masks = np.stack([window["mask"] for window in windows])
            availability[split] = {
                parameter: float(masks[:, :, index].mean())
                for index, parameter in enumerate(parameters)
            }
        else:
            availability[split] = {parameter: None for parameter in parameters}

    manifest = {
        "source": "VitalDB open dataset",
        "num_parameters": args.num_parameters,
        "parameters_in_tensor_order": parameters,
        "selection_rule": "First N dynamic parameters in reliable_parameter_ranking.csv, descending grouped permutation AUPRC drop.",
        "ranking_rows": ranking_rows,
        "tensor_shape": ["windows", args.history_sec // args.interval, args.num_parameters],
        "labels": {str(index): label for index, label in enumerate(LABELS)},
        "target": "First onset of MAP <65 mmHg for >=60 contiguous observed seconds after a fully observed, non-low preceding minute.",
        "bucket_boundaries_seconds": "0 < onset <= 180; 180 < onset <= 300; 300 < onset <= 600; 600 < onset <= 900; otherwise none.",
        "normalization": {
            "method": "training-observation median and IQR; missing values become zero after scaling and are identified by x_mask",
            "median": dict(zip(parameters, median.astype(float).tolist())),
            "iqr_scale": dict(zip(parameters, scale.astype(float).tolist())),
        },
        "interval_sec": args.interval,
        "history_sec": args.history_sec,
        "horizon_sec": args.horizon_sec,
        "stride_sec": args.stride_sec,
        "clean_history_sec": args.clean_history_sec,
        "sampled_cases": int(len(selected_cases)),
        "splits": split_reports,
        "observed_sample_fraction_by_parameter": availability,
        "failed_cases": failures,
        "warning": "The ranking came from the exploratory 40-case validation screen. These files are suitable for model development, but their test split is not an unbiased final evaluation of that feature-selection step.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
