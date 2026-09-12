"""Write a deterministic synthetic manifest for pipeline verification, not clinical training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_example(output: Path, *, patients: int = 10, seed: int = 42) -> Path:
    """Create four-channel, 1 Hz histories with unique synthetic patient/device IDs.

    Refuse to overwrite an existing directory. NaNs exercise observation masks;
    two windows per patient exercise grouped splitting. Values and targets are
    synthetic fixtures and must not be interpreted as treatment annotations.
    """
    if patients < 5:
        raise ValueError("Use at least five patients for the example 80/20 split")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    records = []
    for patient in range(patients):
        for window in range(2):
            length = 300 - window * 17
            baseline = np.array([90.0, 72.0, 98.0, 36.0])[:, None]
            scales = np.array([5.0, 3.0, 0.3, 1.0])[:, None]
            values = (baseline + scales * rng.normal(size=(4, length))).astype(
                np.float32
            )
            values[2, 20:24] = np.nan
            sample_id = f"synthetic-p{patient:03d}-w{window}"
            filename = f"{sample_id}.npy"
            np.save(output / filename, values, allow_pickle=False)
            end = float(300 + window * 300)
            records.append(
                {
                    "sample_id": sample_id,
                    "patient_id": f"synthetic-p{patient:03d}",
                    "case_id": f"synthetic-case{patient:03d}",
                    "device_id": f"synthetic-monitor{patient:03d}",
                    "recording_id": f"synthetic-recording{patient:03d}",
                    "signal_path": filename,
                    "channels": ["ABP", "HR", "SpO2", "EtCO2"],
                    "sample_rate_hz": 1.0,
                    "time_base": "seconds_since_case_start",
                    "window_start": end - length,
                    "window_end": end,
                    "query_time": end,
                    "prompt": "Describe this synthetic telemetry fixture and its data quality.",
                    "target": {
                        "Observation": "This is a synthetic four-channel fixture with missing SpO2 samples.",
                        "Rationale": "These artificial values test the software and do not describe a patient.",
                        "Recommendation": "Verify the pipeline output; no clinical intervention is implied.",
                    },
                }
            )
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Generate a fresh example directory with explicit seed and size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(make_example(args.output, patients=args.patients, seed=args.seed))


if __name__ == "__main__":
    main()
