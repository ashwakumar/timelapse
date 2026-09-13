"""Preprocess local FD001 data with deterministic windows and synthetic labels.

This stage does not download data or invoke an LLM agent.
"""

import argparse
import json
import tempfile
from collections import Counter, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from timenet.errors import TimeFFormatError, TimeFValidationError

from scripts.download_data import file_sha256, inspect_raw_data, iter_raw_rows

ACTIVE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]
SPLIT_UNITS = {
    "train": list(range(1, 71)),
    "validation": list(range(71, 81)),
    "test": list(range(81, 101)),
}


def generate_engineering_rationale(
    unit_id: int,
    cycle: int,
    rul: int,
    drifts: dict[str, float],
    window_size: int = 30,
) -> tuple[str, str, str]:
    """Create observational text and synthetic RUL-band supervision.

    Args:
        unit_id: Engine identifier retained as record metadata, not prompt input.
        cycle: Current observed cycle.
        rul: Known remaining recorded lifetime used only in target text.
        drifts: Last-minus-first sensor values within this window.
        window_size: Number of observed cycles.

    Returns:
        Prompt, target response, and sensor-only observation text.
    """
    prompt = (
        f"Using these 14 sensor channels over the past {window_size} cycles "
        f"(current cycle: {cycle}), estimate remaining useful life in cycles, "
        "assign a RUL band (CRITICAL: 0–30, WARNING: 31–75, NORMAL: above 75), "
        "and summarize the observed sensor changes."
    )
    status = "CRITICAL" if rul <= 30 else "WARNING" if rul <= 75 else "NORMAL"
    rationale = (
        f"Over this {window_size}-cycle window, sensor_4 changed by "
        f"{drifts['sensor_4']:+.2f}, sensor_11 by {drifts['sensor_11']:+.2f}, "
        f"and sensor_15 by {drifts['sensor_15']:+.4f} "
        "(last minus first, in each channel's raw units). "
        "These observations alone do not identify a component fault."
    )
    target = (
        f"Projected RUL: {rul} cycles. Status: {status} (RUL-derived band). "
        "Component diagnosis: not labeled. Maintenance action: not labeled."
    )
    return prompt, target, rationale


def iter_windows(
    raw_file: Path,
    lifetimes: dict[int, int],
    window_size: int,
    stride: int,
) -> Iterator[dict[str, Any]]:
    """Yield bounded-memory sensor windows, always including each terminal window.

    Args:
        raw_file: Validated raw telemetry file.
        lifetimes: Final recorded cycle per engine.
        window_size: Number of sensor samples per window.
        stride: Spacing between regular window end cycles.

    Yields:
        JSON-compatible records with observation inputs and explicit targets.
    """
    window: deque[list[float]] = deque(maxlen=window_size)
    previous_unit = 0
    split_by_unit = {unit: split for split, units in SPLIT_UNITS.items() for unit in units}
    for unit, cycle, sensors in iter_raw_rows(raw_file):
        if unit != previous_unit:
            window.clear()
            previous_unit = unit
        window.append(sensors)
        if len(window) < window_size:
            continue
        if (cycle - window_size) % stride != 0 and cycle != lifetimes[unit]:
            continue
        series = {
            sensor: [row[int(sensor.split("_")[1]) - 1] for row in window]
            for sensor in ACTIVE_SENSORS
        }
        drifts = {sensor: values[-1] - values[0] for sensor, values in series.items()}
        rul = lifetimes[unit] - cycle
        prompt, target, rationale = generate_engineering_rationale(
            unit,
            cycle,
            rul,
            drifts,
            window_size,
        )
        yield {
            "schema_version": 2,
            "record_id": f"FD001-U{unit:03d}-C{cycle:04d}",
            "unit_number": unit,
            "cycle": cycle,
            "window_start": cycle - window_size + 1,
            "split": split_by_unit[unit],
            "rul": rul,
            "status": "CRITICAL" if rul <= 30 else "WARNING" if rul <= 75 else "NORMAL",
            "series": series,
            "prompt": prompt,
            "target": target,
            "rationale": rationale,
            "label_provenance": {
                "rul": "final_recorded_cycle_minus_current_cycle",
                "status": "synthetic_rul_threshold_rule",
                "rationale": "template_of_observed_sensor_differences",
                "component_diagnosis": None,
                "maintenance_action": None,
            },
        }


def preprocess_data(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    window_size: int = 30,
    stride: int = 5,
) -> Path:
    """Validate FD001 and write version-2 JSONL plus a reproducibility manifest.

    Args:
        raw_dir: Raw telemetry cache directory.
        output_dir: Destination directory; existing output files are replaced.
        window_size: Positive observation length no longer than any engine history.
        stride: Positive spacing between regular windows; terminal windows are added.

    Returns:
        Generated JSONL path.

    Raises:
        TimeFValidationError: Window/stride settings are invalid.
        TimeFFormatError: Raw telemetry is invalid or incomplete.
    """
    if type(window_size) is not int or type(stride) is not int or min(window_size, stride) < 1:
        raise TimeFValidationError("window_size and stride must be positive integers")
    raw_file = Path(raw_dir) / "train_FD001.txt"
    if not raw_file.is_file():
        raise FileNotFoundError(
            f"Raw data missing at {raw_file}. Run: uv run python -m scripts.download_data"
        )
    lifetimes, row_count = inspect_raw_data(raw_file)
    if min(lifetimes.values()) < window_size:
        raise TimeFValidationError("window_size exceeds at least one engine's history")
    raw_hash = file_sha256(raw_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out_file = output / "windows.jsonl"
    counts: Counter[str] = Counter()
    statuses: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_UNITS}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for record in iter_windows(raw_file, lifetimes, window_size, stride):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                counts[record["split"]] += 1
                statuses[record["split"]][record["status"]] += 1
            handle.flush()
            if file_sha256(raw_file) != raw_hash:
                raise TimeFFormatError("Raw file changed during preparation")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    source_url = None
    source_file = raw_file.with_suffix(".source.json")
    if source_file.exists():
        source = json.loads(source_file.read_text(encoding="utf-8"))
        if source.get("sha256") == raw_hash:
            source_url = source.get("url")
    manifest = {
        "schema_version": 2,
        "dataset": "C-MAPSS FD001",
        "source": {
            "file": raw_file.name,
            "sha256": raw_hash,
            "rows": row_count,
            "url": source_url,
            "license_status": "not verified",
        },
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "raw_validation_script_sha256": file_sha256(Path(__file__).with_name("download_data.py")),
        "window_size": window_size,
        "stride": stride,
        "include_terminal_window": True,
        "channels": ACTIVE_SENSORS,
        "split_units": SPLIT_UNITS,
        "window_counts": dict(counts),
        "status_counts": statuses,
        "rul_definition": "final_recorded_cycle_minus_current_cycle",
        "rul_cap": None,
        "status_thresholds": {"critical_max": 30, "warning_max": 75},
        "text_labels": "synthetic; no component or maintenance ground truth",
        "evaluation_protocol": "engine holdout within train_FD001; not the official test set",
        "normalization": "none; raw sensor values",
        "output": {"file": out_file.name, "sha256": file_sha256(temporary)},
    }
    temporary.replace(out_file)
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {sum(counts.values())} windows: {dict(counts)} -> {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    preprocess_data(args.raw_dir, args.output_dir, args.window_size, args.stride)
