"""Audit prepared VitalDB splits for subject, case, and time leakage."""

from __future__ import annotations

from scripts.opentslm_vitaldb_dataset import SPLITS, PreparedVitalDBCorpus


def audit_prepared_splits(corpus: PreparedVitalDBCorpus) -> dict:
    """Confirm train/validation/test do not share patients or cases.

    VitalDB public metadata does not identify physical monitor units, so a
    device-disjoint split cannot be proven. Windows from one subject stay in
    one split, which also keeps overlapping case time from crossing the cut.
    """

    splits = {name: corpus.load_split(name) for name in SPLITS}
    subjects = {
        name: set(split.arrays["subject_id"].astype(int).tolist())
        for name, split in splits.items()
    }
    cases = {
        name: set(split.arrays["case_id"].astype(int).tolist())
        for name, split in splits.items()
    }
    subject_overlap = _pairwise_overlap(subjects)
    case_overlap = _pairwise_overlap(cases)
    if subject_overlap:
        raise ValueError(f"Subject identifiers leak across splits: {subject_overlap}")
    if case_overlap:
        raise ValueError(f"Case identifiers leak across splits: {case_overlap}")

    cutoffs: dict[str, dict[int, list[int]]] = {}
    for name, split in splits.items():
        by_case: dict[int, list[int]] = {}
        for case_id, cutoff in zip(
            split.arrays["case_id"].astype(int),
            split.arrays["cutoff_sec"].astype(int),
            strict=True,
        ):
            by_case.setdefault(int(case_id), []).append(int(cutoff))
        cutoffs[name] = {case_id: sorted(values) for case_id, values in by_case.items()}

    return {
        "ok": True,
        "subject_disjoint": True,
        "case_disjoint": True,
        "device_disjoint": False,
        "device_note": (
            "VitalDB open files do not expose physical device-unit IDs, so "
            "device-disjoint evaluation is not claimed."
        ),
        "time_note": (
            "Each subject and case appears in exactly one split, so overlapping "
            "20-second windows from the same recording cannot appear in both "
            "train and held-out sets. Unrelated cases are not ordered on a "
            "global clock."
        ),
        "subjects_by_split": {name: len(values) for name, values in subjects.items()},
        "cases_by_split": {name: len(values) for name, values in cases.items()},
        "windows_by_split": {name: len(split) for name, split in splits.items()},
        "cutoff_sec_by_split_and_case": {
            name: {str(case_id): values for case_id, values in per_case.items()}
            for name, per_case in cutoffs.items()
        },
    }


def _pairwise_overlap(groups: dict[str, set[int]]) -> dict[str, list[int]]:
    names = list(groups)
    overlaps: dict[str, list[int]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(groups[left] & groups[right])
            if shared:
                overlaps[f"{left}/{right}"] = shared
    return overlaps
