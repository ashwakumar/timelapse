#!/usr/bin/env python3
"""Build and verify the local VitalDB TimeNet dataset.

This runner pins the public TimeNet 0.1.0 contract.  It invokes the official
``run_pipeline`` producer API and then reads the committed TimeF version back
through the official ``TimeNet`` client.  No endpoint or credentials are needed:
TimeNet 0.1.0 publishes builds to a local registry.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import numpy as np

from connectors.registry import connector_class_for


REQUIRED_TIMENET_VERSION = "0.1.0"
DATASET_VERSION = "0.1.0"
SPLITS = ("train", "validation", "test")


def _require_timenet() -> None:
    try:
        installed = importlib.metadata.version("timenet")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "TimeNet is not installed. Install the released SDK with "
            "`python -m pip install 'timenet[build]==0.1.0'`."
        ) from error
    if installed != REQUIRED_TIMENET_VERSION:
        raise RuntimeError(
            f"This connector is pinned to timenet=={REQUIRED_TIMENET_VERSION}; found {installed}."
        )


def _expected_counts(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    return {
        "records": sum(int(manifest["splits"][split]["windows"]) for split in SPLITS),
        "subjects": {
            split: int(manifest["splits"][split]["subjects"])
            for split in SPLITS
        },
        "parameters": len(manifest["parameters_in_tensor_order"]),
    }


def audit_dataset(dataset, expected: dict) -> dict:
    """Check masks, tasks, counts, and subject isolation after TimeF readback."""

    subjects_by_split = {split: set() for split in SPLITS}
    null_values = 0
    mask_values = 0
    mismatches: list[str] = []
    classification_tasks = 0
    answer_tasks = 0

    task_by_id = {task.id: task for task in dataset.tasks}
    for record in dataset.records:
        annotations = {annotation.key: annotation.value for annotation in record.annotations}
        split = annotations.get("source_split")
        if split not in subjects_by_split:
            mismatches.append(f"{record.id}: missing or invalid source_split annotation")
            continue
        subjects_by_split[split].update(record.subject_ids)

        by_signal = {series.signal: series for series in record.time_series}
        physical = [series for series in record.time_series if not series.signal.endswith("_observed")]
        if len(record.time_series) != 2 * expected["parameters"]:
            mismatches.append(
                f"{record.id}: expected {2 * expected['parameters']} signal/mask series, "
                f"found {len(record.time_series)}"
            )
        if len(physical) != expected["parameters"]:
            mismatches.append(
                f"{record.id}: expected {expected['parameters']} physical series, found {len(physical)}"
            )
        for series in physical:
            companion = by_signal.get(f"{series.signal}_observed")
            if companion is None:
                mismatches.append(f"{record.id}: missing mask for {series.signal}")
                continue
            values = series.to_arrow()
            observed = companion.to_numpy().astype(bool)
            present = ~values.is_null().to_numpy(zero_copy_only=False)
            null_values += int((~present).sum())
            mask_values += int((~observed).sum())
            if not np.array_equal(present, observed):
                mismatches.append(f"{record.id}: nullable values disagree with {companion.signal}")

        tasks = [task_by_id[task_id] for task_id in record.task_ids]
        classification_tasks += sum(task.task_type.value == "classification" for task in tasks)
        answer_tasks += sum(task.task_type.value == "answer" for task in tasks)
        if len(tasks) != 2:
            mismatches.append(f"{record.id}: expected two tasks, found {len(tasks)}")

    overlaps = {
        f"{left}/{right}": sorted(subjects_by_split[left] & subjects_by_split[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
        if subjects_by_split[left] & subjects_by_split[right]
    }
    if overlaps:
        mismatches.append(f"subjects cross split boundaries: {overlaps}")
    if len(dataset.records) != expected["records"]:
        mismatches.append(
            f"record count {len(dataset.records)} does not match source {expected['records']}"
        )
    if null_values != mask_values:
        mismatches.append(
            f"nullable missing count {null_values} does not match mask-zero count {mask_values}"
        )
    for split, count in expected["subjects"].items():
        if len(subjects_by_split[split]) != count:
            mismatches.append(
                f"{split} subject count {len(subjects_by_split[split])} does not match source {count}"
            )
    if classification_tasks != expected["records"] or answer_tasks != expected["records"]:
        mismatches.append(
            f"task counts classification={classification_tasks}, answer={answer_tasks}, "
            f"expected each={expected['records']}"
        )
    if mismatches:
        raise RuntimeError("TimeNet readback audit failed: " + "; ".join(mismatches[:20]))

    return {
        "records": len(dataset.records),
        "time_series": sum(len(record.time_series) for record in dataset.records),
        "classification_tasks": classification_tasks,
        "answer_tasks": answer_tasks,
        "missing_signal_values": null_values,
        "mask_zero_values": mask_values,
        "subjects_by_split": {
            split: len(subjects) for split, subjects in subjects_by_split.items()
        },
        "subject_overlap": {},
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/lstm/top_10"))
    parser.add_argument("--out", type=Path, default=Path("data/timenet_registry"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Run download/convert/schema validation without writing a registry.",
    )
    parser.add_argument(
        "--sourced-dataset-id",
        default="vitaldb",
        help="Challenge-1 dataset_id used to look up the TimeNet connector.",
    )
    return parser.parse_args(argv)


def ingest_prepared(
    data_dir: Path,
    *,
    registry: Path | None = None,
    force: bool = False,
    convert_only: bool = False,
    sourced_dataset_id: str = "vitaldb",
) -> dict:
    """Convert a prepared corpus through the registered TimeNet connector."""

    data_dir = data_dir.resolve()
    expected = _expected_counts(data_dir)
    _require_timenet()

    os.environ["VITALDB_PREPARED_DIR"] = str(data_dir)
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    connector_cls = connector_class_for(sourced_dataset_id)
    connector = connector_cls()
    metadata = connector.metadata()
    if convert_only:
        raw_refs = connector.download(data_dir)
        dataset = connector.convert(raw_refs)
        schema = dataset.derive_schema()
        return {
            "mode": "convert-only",
            "sourced_dataset_id": sourced_dataset_id,
            "dataset_id": metadata.dataset_id,
            "records": len(dataset.records),
            "tasks": len(dataset.tasks),
            "time_series_specs": len(schema.time_series_specs),
            "annotations": len(schema.annotations),
            "audit": {
                "records": len(dataset.records),
                "tasks": len(dataset.tasks),
            },
        }

    if registry is None:
        raise ValueError("registry output path is required unless --convert-only")

    from timenet.client import TimeNet
    from timenet.engine import run_pipeline

    registry = registry.resolve()
    with tempfile.TemporaryDirectory(prefix="timenet-vitaldb-cache-") as cache_dir:
        version_dir = run_pipeline(
            connector,
            registry,
            cache_dir=Path(cache_dir),
            force=force,
        )
    readback = TimeNet(registry).load(f"{metadata.dataset_id}@{DATASET_VERSION}")
    return {
        "mode": "build-and-readback",
        "sourced_dataset_id": sourced_dataset_id,
        "timenet_version": REQUIRED_TIMENET_VERSION,
        "dataset_id": metadata.dataset_id,
        "dataset_version": DATASET_VERSION,
        "version_dir": str(version_dir),
        "manifest": str(Path(version_dir) / "manifest.json"),
        "audit": audit_dataset(readback, expected),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = ingest_prepared(
        args.data_dir,
        registry=None if args.convert_only else args.out,
        force=args.force,
        convert_only=args.convert_only,
        sourced_dataset_id=args.sourced_dataset_id,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
