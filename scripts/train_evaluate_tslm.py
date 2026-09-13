#!/usr/bin/env python3
"""Train the patch TSLM, a window-summary baseline, and compare them on test.

Uses the patient/case-disjoint splits already written by the VitalDB prepare
step. Does not resplit, and does not use the validation split for the reported
numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.compare import compare_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/lstm/top_5"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tslm_evaluation"),
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.data_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Missing prepared corpus {args.data_dir}. Run challenge 2 "
            "(`python scripts/bring_into_timenet.py`) first; this script does not "
            "replace data sourcing or TimeNet ingestion."
        )
    report = compare_models(
        args.data_dir, l2=args.l2, patch_size=args.patch_size
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "held_out_split": report["held_out_split"],
        "leakage_ok": report["leakage"]["ok"],
        "baseline": report["models"]["baseline"],
        "tslm": report["models"]["tslm"],
        "wrote": str(path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
