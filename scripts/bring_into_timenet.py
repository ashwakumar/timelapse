#!/usr/bin/env python3
"""Bring the challenge-1 selected dataset into TimeNet.

Requires artifacts/data_sourcing/selection.json from
``python scripts/source_datasets.py``. Prepares standardized windows, then
runs the registered reusable connector through TimeNet's official engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectors.registry import load_selection
from scripts.ingest_timenet import ingest_prepared
from scripts.prepare_lstm_dataset import parse_args as parse_prepare_args
from scripts.prepare_lstm_dataset import prepare_corpus


def prepared_corpus_ready(data_dir: Path) -> bool:
    required = ("manifest.json", "train.npz", "validation.npz", "test.npz")
    return all((data_dir / name).is_file() for name in required)


def bring_into_timenet(
    *,
    selection_path: Path,
    num_parameters: int,
    max_cases: int,
    data_root: Path,
    registry: Path,
    output_dir: Path,
    force: bool,
    convert_only: bool,
    skip_prepare: bool,
) -> dict:
    selection = load_selection(selection_path)
    data_dir = data_root / f"top_{num_parameters}"
    prepared = None
    if skip_prepare and not prepared_corpus_ready(data_dir):
        raise FileNotFoundError(
            f"--skip-prepare was set but {data_dir} is missing train/validation/test NPZs"
        )
    if not skip_prepare and not prepared_corpus_ready(data_dir):
        prepare_args = parse_prepare_args(
            [
                str(num_parameters),
                "--max-cases",
                str(max_cases),
                "--output-root",
                str(data_root),
            ]
        )
        prepared = prepare_corpus(prepare_args)
    elif not prepared_corpus_ready(data_dir):
        raise FileNotFoundError(f"Prepared corpus is missing: {data_dir}")

    ingestion = ingest_prepared(
        data_dir,
        registry=None if convert_only else registry,
        force=force,
        convert_only=convert_only,
        sourced_dataset_id=selection["dataset_id"],
    )
    report = {
        "challenge_1_selection": selection,
        "prepared_corpus": str(data_dir),
        "prepared_manifest_summary": None
        if prepared is None
        else {
            "windows": {
                split: prepared["splits"][split]["windows"] for split in prepared["splits"]
            },
            "parameters": prepared["parameters_in_tensor_order"],
            "labels": prepared["labels"],
        },
        "timenet": ingestion,
        "targets": {
            "classification_schema": "hypotension_onset_horizon_v1",
            "labels": [
                "within_3",
                "within_5",
                "within_10",
                "within_15",
                "none_within_15",
            ],
            "question": (
                "Using only the supplied 20-second signal history and observation masks, "
                "predict the tightest available horizon containing the first onset of "
                "sustained MAP below 65 mmHg."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("artifacts/data_sourcing/selection.json"),
        help="Challenge-1 selection written by scripts/source_datasets.py",
    )
    parser.add_argument("--num-parameters", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("data/lstm"))
    parser.add_argument("--registry", type=Path, default=Path("data/timenet_registry"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/timenet_ingestion"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Do not download VitalDB; require an existing prepared corpus.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    report = bring_into_timenet(
        selection_path=args.selection,
        num_parameters=args.num_parameters,
        max_cases=args.max_cases,
        data_root=args.data_root,
        registry=args.registry,
        output_dir=args.output_dir,
        force=args.force,
        convert_only=args.convert_only,
        skip_prepare=args.skip_prepare,
    )
    print(json.dumps({"dataset_id": report["timenet"]["dataset_id"], "mode": report["timenet"]["mode"]}, indent=2))
    print(f"Wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
