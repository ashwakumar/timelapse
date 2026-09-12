"""Download and validate raw C-MAPSS FD001 telemetry without preprocessing it."""

import argparse
import hashlib
import json
import math
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from timenet.errors import TimeFFormatError

RAW_DATA_URLS = [
    "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data/train_FD001.txt",
    "https://raw.githubusercontent.com/sahilkhanna/CMAPSS-NASA/master/train_FD001.txt",
]
COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def iter_raw_rows(path: Path) -> Iterator[tuple[int, int, list[float]]]:
    """Read finite, ordered 26-column rows without loading the file into memory.

    Args:
        path: Raw FD001 text file, ordered by engine then cycle.

    Yields:
        Engine ID, cycle, and the 21 sensor measurements.

    Raises:
        TimeFFormatError: Rows are malformed, duplicated, missing, or out of order.
    """
    previous_unit, previous_cycle = 0, 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            context = f"{path}:{line_number}"
            if len(fields) != len(COLUMN_NAMES):
                raise TimeFFormatError(f"{context}: expected 26 columns, got {len(fields)}")
            try:
                values = [float(value) for value in fields]
            except ValueError as exc:
                raise TimeFFormatError(f"{context}: non-numeric value") from exc
            if not all(math.isfinite(value) for value in values):
                raise TimeFFormatError(f"{context}: non-finite value")
            if values[0] != int(values[0]) or values[1] != int(values[1]):
                raise TimeFFormatError(f"{context}: engine ID and cycle must be integers")
            unit, cycle = int(values[0]), int(values[1])
            if not 1 <= unit <= 100 or cycle < 1:
                raise TimeFFormatError(f"{context}: invalid FD001 engine ID or cycle")
            if unit < previous_unit:
                raise TimeFFormatError(f"{context}: engines must be in ascending order")
            expected_cycle = previous_cycle + 1 if unit == previous_unit else 1
            if cycle != expected_cycle:
                raise TimeFFormatError(f"{context}: expected cycle {expected_cycle}, got {cycle}")
            previous_unit, previous_cycle = unit, cycle
            yield unit, cycle, values[5:]
    if previous_unit == 0:
        raise TimeFFormatError(f"{path}: no telemetry rows")


def inspect_raw_data(path: Path) -> tuple[dict[int, int], int]:
    """Validate a complete FD001 training file and collect engine lifetimes.

    Args:
        path: Raw telemetry file.

    Returns:
        Final cycle per engine and total row count.

    Raises:
        TimeFFormatError: The file is invalid or does not contain engines 1–100.
    """
    lifetimes: dict[int, int] = {}
    row_count = 0
    for unit, cycle, _ in iter_raw_rows(path):
        lifetimes[unit] = cycle
        row_count += 1
    if set(lifetimes) != set(range(1, 101)):
        raise TimeFFormatError("FD001 training input must contain exactly engines 1–100")
    return lifetimes, row_count


def download_raw_data(raw_dir: Path) -> Path:
    """Use cached raw data or validate a temporary download before publishing it.

    Args:
        raw_dir: Cache directory for train_FD001.txt.

    Returns:
        Validated cached raw file path.

    Raises:
        TimeFFormatError: Every download source fails or contains invalid data.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "train_FD001.txt"
    if raw_file.exists():
        inspect_raw_data(raw_file)
        return raw_file
    errors = []
    for url in RAW_DATA_URLS:
        with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            inspect_raw_data(temporary)
            temporary.replace(raw_file)
            raw_file.with_suffix(".source.json").write_text(
                json.dumps({"url": url, "sha256": file_sha256(raw_file)}, indent=2) + "\n",
                encoding="utf-8",
            )
            return raw_file
        except (OSError, urllib.error.URLError, TimeFFormatError) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
    raise TimeFFormatError("Could not obtain valid FD001 data: " + "; ".join(errors))


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    print(f"Validated raw data: {download_raw_data(Path(args.raw_dir))}")
