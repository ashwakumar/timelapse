#!/usr/bin/env python3
"""Prepare the real, feature-screen-independent VitalDB onset-v2 corpus.

The default path is the only corpus accepted by ``training.vitaldb``.  Its
labels use an additional observed minute after the 15-minute horizon so an
episode beginning at the horizon boundary can still be confirmed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm

try:
    from scripts.feature_screen import TARGET_TRACK, TRACKS, first_true_run_start, load_case
    from scripts.opentslm_vitaldb_dataset import PreparedVitalDBCorpus
    from scripts.prepare_lstm_dataset import (
        LABELS,
        LABEL_TO_INDEX,
        onset_bucket,
        save_split,
        select_top_parameters,
        training_normalization,
    )
except ImportError:  # direct execution from scripts/
    from feature_screen import TARGET_TRACK, TRACKS, first_true_run_start, load_case
    from opentslm_vitaldb_dataset import PreparedVitalDBCorpus
    from prepare_lstm_dataset import (
        LABELS,
        LABEL_TO_INDEX,
        onset_bucket,
        save_split,
        select_top_parameters,
        training_normalization,
    )


TRAINING_CONTRACT = "vitaldb_onset_v2"
EXPLORATORY_CONTRACT = "vitaldb_onset_v2_exploratory"
TARGET_VERSION = "hypotension_onset_horizon_v2_confirmed_lookahead"
FEATURE_SCREEN_SELECTION_VERSION = "feature_screen_v1_target_hr_seeded_permutation"
FEATURE_SCREEN_SEED = 20260912
FEATURE_SCREEN_CASE_LIMIT = 40

INTERVAL_SEC = 2
HISTORY_SEC = 20
HORIZON_SEC = 900
STRIDE_SEC = 60
CLEAN_HISTORY_SEC = 60
SUSTAIN_SEC = 60


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-limit", type=int, default=1000)
    parser.add_argument("--num-parameters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--ranking-file",
        type=Path,
        default=Path("artifacts/feature_screen/reliable_parameter_ranking.csv"),
    )
    parser.add_argument(
        "--feature-screen-report",
        type=Path,
        default=Path("artifacts/feature_screen/report.json"),
    )
    parser.add_argument(
        "--feature-screen-cohort",
        type=Path,
        default=Path("artifacts/feature_screen/cohort.json"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/cache/vitaldb_onset_v2")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/vitaldb/onset_v2")
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "Permit incomplete feature-screen provenance or target-class coverage. "
            "The resulting contract is deliberately rejected by final training."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared output only after the new corpus validates.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.case_limit < 1:
        raise ValueError("--case-limit must be positive")
    if args.num_parameters < 1:
        raise ValueError("--num-parameters must be positive")
    validate_fractions(
        args.train_fraction, args.validation_fraction, args.test_fraction
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output is not empty: {args.output_dir}. Pass --overwrite to replace it atomically."
        )


def validate_fractions(train: float, validation: float, test: float) -> None:
    fractions = np.asarray([train, validation, test], dtype=float)
    if not (np.isfinite(fractions).all() and (fractions > 0).all()):
        raise ValueError("All split fractions must be finite and positive")
    if not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1")


def split_subjects(
    subject_ids: Sequence[int] | np.ndarray,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, set[int]]:
    """Assign each subject once, with deterministic largest-remainder counts."""

    validate_fractions(train_fraction, validation_fraction, test_fraction)
    subjects = np.asarray(sorted({int(value) for value in subject_ids}), dtype=np.int64)
    if len(subjects) < 3:
        raise RuntimeError("At least three subjects are required for three non-empty splits")

    fractions = np.asarray([train_fraction, validation_fraction, test_fraction])
    exact = fractions * len(subjects)
    counts = np.floor(exact).astype(int)
    for index in sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))[
        : len(subjects) - int(counts.sum())
    ]:
        counts[index] += 1
    for empty in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise RuntimeError("Could not form three non-empty subject splits")
        counts[donor] -= 1
        counts[empty] += 1

    shuffled = np.random.default_rng(seed).permutation(subjects)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return {
        "train": set(int(v) for v in shuffled[:train_end]),
        "validation": set(int(v) for v in shuffled[train_end:validation_end]),
        "test": set(int(v) for v in shuffled[validation_end:]),
    }


def case_windows(
    frame: pd.DataFrame,
    parameters: list[str],
    case_id: int,
    subject_id: int,
    interval: int = INTERVAL_SEC,
    history_sec: int = HISTORY_SEC,
    horizon_sec: int = HORIZON_SEC,
    stride_sec: int = STRIDE_SEC,
    clean_history_sec: int = CLEAN_HISTORY_SEC,
    sustain_sec: int = SUSTAIN_SEC,
) -> list[dict]:
    """Create auditable labels with a fully observed confirmation lookahead."""

    durations = (history_sec, horizon_sec, stride_sec, clean_history_sec, sustain_sec)
    if interval <= 0 or any(value <= 0 or value % interval for value in durations):
        raise ValueError("All durations must be positive multiples of interval")
    missing_columns = sorted(set(["MAP", *parameters]) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Case frame is missing columns: {missing_columns}")

    history_n = history_sec // interval
    horizon_n = horizon_sec // interval
    stride_n = stride_sec // interval
    clean_n = clean_history_sec // interval
    required_low_n = sustain_sec // interval
    target_lookahead_n = horizon_n + required_low_n
    first_cutoff = max(history_n, clean_n)
    windows: list[dict] = []

    for cutoff in range(
        first_cutoff, len(frame) - target_lookahead_n + 1, stride_n
    ):
        clean_map = frame["MAP"].iloc[cutoff - clean_n : cutoff].to_numpy(
            dtype=np.float32
        )
        if not (np.isfinite(clean_map).all() and (clean_map >= 65).all()):
            continue

        target_map = frame["MAP"].iloc[
            cutoff : cutoff + target_lookahead_n
        ].to_numpy(dtype=np.float32)
        # Unknown target samples make both event and no-event labels uncertain.
        # Reject the window instead of silently treating missingness as non-low.
        if target_map.size != target_lookahead_n or not np.isfinite(target_map).all():
            continue

        low = target_map < 65
        onset_index = first_true_run_start(low, required_low_n)
        onset_seconds = None
        if onset_index is not None and onset_index <= horizon_n:
            onset_seconds = int(onset_index * interval)
        label = onset_bucket(onset_seconds)

        values = frame[parameters].iloc[
            cutoff - history_n : cutoff
        ].to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        windows.append(
            {
                "values": values,
                "mask": mask,
                "label": LABEL_TO_INDEX[label],
                "case_id": int(case_id),
                "subject_id": int(subject_id),
                # This sample is excluded from history and is time zero for target.
                "cutoff_sec": int(cutoff * interval),
                "future_map_coverage": 1.0,
                "onset_seconds": onset_seconds,
            }
        )
    return windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_screen_spec(
    cohort_path: Path,
    report_path: Path,
    ranking_path: Path,
    exploratory: bool,
) -> dict:
    """Load and authenticate the immutable feature-screen subject cohort."""

    issues: list[str] = []
    if cohort_path.is_file():
        cohort = json.loads(cohort_path.read_text())
    else:
        cohort = {}
        issues.append(f"missing immutable cohort file: {cohort_path}")

    case_ids = sorted({int(value) for value in cohort.get("case_ids", [])})
    subject_ids = sorted({int(value) for value in cohort.get("subject_ids", [])})
    if cohort.get("schema_version") != 1:
        issues.append("cohort schema_version must be 1")
    if not case_ids or not subject_ids:
        issues.append("cohort must contain case_ids and subject_ids")

    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        actual_report_hash = _sha256(report_path)
        if cohort.get("report_sha256") != actual_report_hash:
            issues.append("feature-screen report checksum does not match cohort.json")
    else:
        actual_report_hash = None
        issues.append(f"missing feature-screen report: {report_path}")

    if ranking_path.is_file():
        actual_ranking_hash = _sha256(ranking_path)
        if cohort.get("ranking_sha256") != actual_ranking_hash:
            issues.append("feature-screen ranking checksum does not match cohort.json")
    else:
        actual_ranking_hash = None
        issues.append(f"missing feature-screen ranking: {ranking_path}")

    if report is not None:
        expected = {
            "usable_cases": len(case_ids),
            "usable_subjects": len(subject_ids),
            "windows": int(cohort.get("windows", -1)),
        }
        for key, value in expected.items():
            if int(report.get(key, -1)) != value:
                issues.append(f"cohort {key} does not match report.json")
    if issues and not exploratory:
        raise RuntimeError(
            "Feature-screen cohort provenance failed closed: " + "; ".join(issues)
        )

    return {
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path) if cohort_path.is_file() else None,
        "schema_version": cohort.get("schema_version"),
        "authoritative_case_ids": case_ids,
        "authoritative_subject_ids": subject_ids,
        "windows": cohort.get("windows"),
        "source_table_name": cohort.get("source_table_name"),
        "source_table_sha256": cohort.get("source_table_sha256"),
        "report_path": str(report_path),
        "report_sha256": actual_report_hash,
        "ranking_path": str(ranking_path),
        "ranking_sha256": actual_ranking_hash,
        "integrity_issues": issues,
    }


def _subject_by_case(api: object, case_ids: Sequence[int]) -> dict[int, int]:
    clinical = api.load_clinical_data(
        caseids=[int(case_id) for case_id in case_ids],
        params=["caseid", "subjectid"],
    )
    required = {"caseid", "subjectid"}
    if not required.issubset(clinical.columns):
        raise ValueError(
            f"VitalDB clinical data is missing columns: {sorted(required - set(clinical.columns))}"
        )
    mapping: dict[int, int] = {}
    for case_id, subject_id in zip(clinical["caseid"], clinical["subjectid"]):
        if pd.notna(case_id) and pd.notna(subject_id):
            mapping[int(case_id)] = int(subject_id)
    return mapping


def discover_cases(api: object, args: argparse.Namespace) -> tuple[list[int], dict[int, int], dict]:
    """Resolve candidate cases and independently reproduce screen exclusions."""

    spec = feature_screen_spec(
        args.feature_screen_cohort,
        args.feature_screen_report,
        args.ranking_file,
        args.exploratory,
    )
    screen_eligible = sorted(
        int(case_id) for case_id in api.find_cases([TARGET_TRACK, TRACKS["HR"]])
    )
    screen_cases = [
        int(case_id)
        for case_id in np.random.default_rng(FEATURE_SCREEN_SEED).permutation(
            screen_eligible
        )[: min(FEATURE_SCREEN_CASE_LIMIT, len(screen_eligible))]
    ]
    candidate_cases = sorted(
        int(case_id) for case_id in api.find_cases([TARGET_TRACK])
    )
    subject_by_case = _subject_by_case(
        api, sorted(set(screen_cases) | set(candidate_cases))
    )

    unresolved_screen = sorted(set(screen_cases) - set(subject_by_case))
    reproduced_subjects = {
        subject_by_case[case_id] for case_id in screen_cases if case_id in subject_by_case
    }
    authoritative_subjects = set(spec["authoritative_subject_ids"])
    screen_subjects = sorted(authoritative_subjects | reproduced_subjects)
    eligible = [
        case_id
        for case_id in candidate_cases
        if case_id in subject_by_case and subject_by_case[case_id] not in screen_subjects
    ]
    selected = [
        int(case_id)
        for case_id in np.random.default_rng(args.seed).permutation(eligible)[
            : min(args.case_limit, len(eligible))
        ]
    ]
    if not selected:
        raise RuntimeError("No VitalDB cases remain after feature-screen subject exclusion")

    selected_subjects = {subject_by_case[case_id] for case_id in selected}
    overlap = selected_subjects & set(screen_subjects)
    if overlap:
        raise AssertionError(f"Feature-screen subjects leaked into preparation: {sorted(overlap)}")

    provenance = {
        **spec,
        "excluded_subject_ids": screen_subjects,
        "conservative_reproduction": {
            "selection_version": FEATURE_SCREEN_SELECTION_VERSION,
            "seed": FEATURE_SCREEN_SEED,
            "case_limit": FEATURE_SCREEN_CASE_LIMIT,
            "eligibility_tracks": [TARGET_TRACK, TRACKS["HR"]],
            "current_eligible_case_count": len(screen_eligible),
            "selected_case_ids": screen_cases,
            "resolved_subject_ids": sorted(reproduced_subjects),
            "additional_subject_ids": sorted(reproduced_subjects - authoritative_subjects),
            "unresolved_case_ids": unresolved_screen,
        },
        "candidate_cases_removed_by_subject_exclusion": int(
            sum(
                case_id in subject_by_case
                and subject_by_case[case_id] in set(screen_subjects)
                for case_id in candidate_cases
            )
        ),
    }
    return selected, subject_by_case, provenance


def _validate_target_coverage(split_windows: dict[str, list[dict]], exploratory: bool) -> None:
    for name, windows in split_windows.items():
        if not windows:
            raise RuntimeError(f"The {name} split produced no eligible windows")
    if exploratory:
        return
    train_labels = {int(window["label"]) for window in split_windows["train"]}
    if train_labels != set(range(len(LABELS))):
        raise RuntimeError(
            f"Strict training split needs all five labels, found {sorted(train_labels)}"
        )
    for name, windows in split_windows.items():
        labels = np.asarray([window["label"] for window in windows])
        has_event = np.any(labels < LABEL_TO_INDEX["none_within_15"])
        has_no_event = np.any(labels == LABEL_TO_INDEX["none_within_15"])
        if not (has_event and has_no_event):
            raise RuntimeError(
                f"Strict {name} split needs both event and no-event windows"
            )


def _write_corpus(
    output_dir: Path,
    manifest: dict,
    split_windows: dict[str, list[dict]],
    median: np.ndarray,
    scale: np.ndarray,
    overwrite: bool,
) -> dict:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        split_reports = {
            name: save_split(
                stage / f"{name}.npz",
                windows,
                median,
                scale,
                HISTORY_SEC // INTERVAL_SEC,
                len(manifest["parameters_in_tensor_order"]),
            )
            for name, windows in split_windows.items()
        }
        manifest["splits"] = split_reports
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        PreparedVitalDBCorpus(stage).validate()

        if output_dir.exists():
            if any(output_dir.iterdir()):
                if not overwrite:
                    raise FileExistsError(f"Output is not empty: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                output_dir.rmdir()
        stage.replace(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return split_reports


def prepare_corpus(
    args: argparse.Namespace,
    *,
    api: object | None = None,
    case_loader: Callable[[int, int, Path], pd.DataFrame] | None = None,
    show_progress: bool = True,
) -> dict:
    """Download, label, split, normalize, validate, and atomically save a corpus."""

    validate_args(args)
    api = vitaldb if api is None else api
    case_loader = load_case if case_loader is None else case_loader
    parameters, ranking_rows = select_top_parameters(
        args.ranking_file, args.num_parameters
    )
    if "MAP" not in parameters:
        raise RuntimeError("Selected parameters must include MAP")

    selected_cases, subject_by_case, exclusion = discover_cases(api, args)
    subjects_by_split = split_subjects(
        [subject_by_case[case_id] for case_id in selected_cases],
        args.seed,
        args.train_fraction,
        args.validation_fraction,
        args.test_fraction,
    )
    split_for_subject = {
        subject: name for name, subjects in subjects_by_split.items() for subject in subjects
    }
    split_windows: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    failures: list[dict] = []
    iterator = tqdm(
        selected_cases,
        desc="Preparing real VitalDB onset-v2 cases",
        disable=not show_progress,
    )
    for case_id in iterator:
        subject_id = subject_by_case[case_id]
        try:
            frame = case_loader(case_id, INTERVAL_SEC, args.cache_dir)
            split_windows[split_for_subject[subject_id]].extend(
                case_windows(frame, parameters, case_id, subject_id)
            )
        except Exception as error:
            failures.append({"case_id": case_id, "error": repr(error)})

    _validate_target_coverage(split_windows, args.exploratory)
    median, scale = training_normalization(split_windows["train"], len(parameters))
    availability: dict[str, dict[str, float | None]] = {}
    for name, windows in split_windows.items():
        if windows:
            masks = np.stack([window["mask"] for window in windows])
            availability[name] = {
                parameter: float(masks[:, :, index].mean())
                for index, parameter in enumerate(parameters)
            }
        else:
            availability[name] = {parameter: None for parameter in parameters}

    manifest = {
        "training_contract": (
            EXPLORATORY_CONTRACT if args.exploratory else TRAINING_CONTRACT
        ),
        "target_version": TARGET_VERSION,
        "mode": "exploratory" if args.exploratory else "final_candidate",
        "source": "Real VitalDB open dataset downloaded directly with the vitaldb Python library",
        "source_is_synthetic": False,
        "num_parameters": len(parameters),
        "parameters_in_tensor_order": parameters,
        "selection_rule": "First N dynamic parameters in the committed reliable feature-screen ranking.",
        "ranking_file": str(args.ranking_file),
        "ranking_file_sha256": _sha256(args.ranking_file),
        "ranking_rows": ranking_rows,
        "tensor_shape": ["windows", HISTORY_SEC // INTERVAL_SEC, len(parameters)],
        "labels": {str(index): label for index, label in enumerate(LABELS)},
        "target": (
            "First onset at or after cutoff of MAP <65 mmHg for 60 contiguous "
            "observed seconds, after a fully observed non-low preceding minute."
        ),
        "bucket_boundaries_seconds": (
            "0 <= onset <= 180; 180 < onset <= 300; 300 < onset <= 600; "
            "600 < onset <= 900; otherwise none."
        ),
        "target_observation_policy": (
            "Every MAP sample in the 900-second horizon plus 60-second "
            "confirmation lookahead must be observed; uncertain windows are excluded."
        ),
        "cutoff_definition": "Timestamp of the first sample excluded from input history; target time zero.",
        "normalization": {
            "method": (
                "training-observation median and IQR; missing inputs become zero after "
                "scaling and are identified by x_mask"
            ),
            "median": dict(zip(parameters, median.astype(float).tolist())),
            "iqr_scale": dict(zip(parameters, scale.astype(float).tolist())),
        },
        "interval_sec": INTERVAL_SEC,
        "history_sec": HISTORY_SEC,
        "horizon_sec": HORIZON_SEC,
        "confirmation_lookahead_sec": SUSTAIN_SEC,
        "sustained_low_sec": SUSTAIN_SEC,
        "stride_sec": STRIDE_SEC,
        "clean_history_sec": CLEAN_HISTORY_SEC,
        "case_limit": args.case_limit,
        "sampled_cases": len(selected_cases),
        "sampled_case_ids": selected_cases,
        "selection_seed": args.seed,
        "requested_split_fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "assigned_subject_ids_by_split": {
            name: sorted(subjects) for name, subjects in subjects_by_split.items()
        },
        "feature_screen_exclusion": exclusion,
        "observed_sample_fraction_by_parameter": availability,
        "failed_cases": failures,
        "provenance_note": (
            "Feature-screen subjects are excluded before case sampling and subject-disjoint splitting."
        ),
    }
    split_reports = _write_corpus(
        args.output_dir,
        manifest,
        split_windows,
        median,
        scale,
        args.overwrite,
    )
    result = dict(manifest)
    result["splits"] = split_reports
    result["output_dir"] = str(args.output_dir)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    manifest = prepare_corpus(parse_args(argv))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
