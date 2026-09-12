"""Regression tests for data integrity, engine holdouts, and target provenance."""

import json
from pathlib import Path

import pytest
from timenet.errors import TimeFFormatError, TimeFValidationError

from scripts.download_data import file_sha256, iter_raw_rows
from scripts.preprocess_data import (
    generate_engineering_rationale,
    preprocess_data,
)


def raw_line(unit: int, cycle: int) -> str:
    """Build a deterministic 26-column telemetry fixture row.

    Args:
        unit: Fixture engine identifier.
        cycle: Fixture observation cycle.

    Returns:
        Whitespace-separated row ending in a newline.
    """
    return " ".join(map(str, [unit, cycle, 0, 0, 100] + [cycle + i for i in range(21)])) + "\n"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Create a complete miniature FD001 fixture with 100 engines.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Raw data directory.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train_FD001.txt").write_text(
        "".join(raw_line(unit, cycle) for unit in range(1, 101) for cycle in range(1, 9)),
        encoding="utf-8",
    )
    return raw


def test_windows_splits_targets_and_manifest(raw_dir: Path, tmp_path: Path) -> None:
    """Check terminal inclusion, truthful labels, deterministic output, and isolation."""
    output = tmp_path / "processed"
    path = preprocess_data(str(raw_dir), str(output), window_size=3, stride=3)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 300
    first = [r for r in records if r["unit_number"] == 1]
    assert [r["cycle"] for r in first] == [3, 6, 8]
    assert [r["rul"] for r in first] == [5, 2, 0]
    assert first[-1]["window_start"] == 6
    assert first[-1]["series"]["sensor_2"] == [7, 8, 9]
    assert "past 3 cycles" in first[0]["prompt"]
    assert first[0]["label_provenance"]["component_diagnosis"] is None
    assert "overhaul" not in first[0]["target"].lower()
    assert len({r["record_id"] for r in records}) == len(records)
    split_units = {
        split: {r["unit_number"] for r in records if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert split_units == {
        "train": set(range(1, 71)),
        "validation": set(range(71, 81)),
        "test": set(range(81, 101)),
    }
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["window_counts"] == {"train": 210, "validation": 30, "test": 60}
    assert manifest["source"]["sha256"] == file_sha256(raw_dir / "train_FD001.txt")
    assert manifest["source"]["url"] is None
    assert manifest["output"]["sha256"] == file_sha256(path)
    previous = path.read_bytes(), (output / "dataset_manifest.json").read_bytes()
    preprocess_data(str(raw_dir), str(output), window_size=3, stride=3)
    assert previous == (path.read_bytes(), (output / "dataset_manifest.json").read_bytes())


def test_exact_length_engine_has_one_window(raw_dir: Path, tmp_path: Path) -> None:
    """Keep exactly one terminal window when the history equals the window length."""
    path = preprocess_data(str(raw_dir), str(tmp_path / "out"), window_size=8)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 100
    assert all(record["rul"] == 0 for record in records)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "1 1 0\n",
        raw_line(1, 1).replace("100", "nan"),
        raw_line(1, 1).replace("100", "inf"),
        raw_line(1, 1).replace("100", "bad"),
        raw_line(1, 1) + raw_line(1, 1),
        raw_line(1, 1) + raw_line(1, 3),
        raw_line(2, 1) + raw_line(1, 1),
        raw_line(101, 1),
        raw_line(1, 1).replace("1 1 ", "1.5 1 ", 1),
    ],
)
def test_reject_corrupt_raw_rows(tmp_path: Path, content: str) -> None:
    """Reject malformed values, duplicate cycles, gaps, and invalid engine IDs."""
    path = tmp_path / "raw.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TimeFFormatError):
        list(iter_raw_rows(path))


@pytest.mark.parametrize("window,stride", [(0, 5), (3, 0), (-1, 1), (3, -1), (True, 1)])
def test_invalid_configuration(tmp_path: Path, window: int, stride: int) -> None:
    """Reject invalid configuration before attempting a download."""
    with pytest.raises(TimeFValidationError):
        preprocess_data(str(tmp_path / "missing"), str(tmp_path / "out"), window, stride)


def test_failed_validation_preserves_output(raw_dir: Path, tmp_path: Path) -> None:
    """Keep prior processed data when raw input is incomplete or window too large."""
    output = tmp_path / "out"
    path = preprocess_data(str(raw_dir), str(output), 3, 3)
    previous = path.read_bytes()
    with pytest.raises(TimeFValidationError):
        preprocess_data(str(raw_dir), str(output), 9, 3)
    (raw_dir / "train_FD001.txt").write_text(raw_line(1, 1), encoding="utf-8")
    with pytest.raises(TimeFFormatError):
        preprocess_data(str(raw_dir), str(output), 3, 3)
    assert path.read_bytes() == previous


@pytest.mark.parametrize(
    "rul,status",
    [(0, "CRITICAL"), (30, "CRITICAL"), (31, "WARNING"), (75, "WARNING"), (76, "NORMAL")],
)
def test_thresholds_and_observation_separation(rul: int, status: str) -> None:
    """Ground truth changes targets but never the prompt or sensor observations."""
    drifts = {"sensor_4": 2.0, "sensor_11": -3.0, "sensor_15": 0.1}
    prompt, target, rationale = generate_engineering_rationale(1, 30, rul, drifts)
    other_prompt, _, other_rationale = generate_engineering_rationale(99, 30, 1234, drifts)
    assert f"Status: {status}" in target
    assert prompt == other_prompt
    assert rationale == other_rationale
    assert "1234" not in prompt + rationale


def test_download_validates_before_caching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Try the next mirror when a successful response contains invalid telemetry."""
    import io

    from scripts.download_data import RAW_DATA_URLS, download_raw_data

    valid = "".join(raw_line(unit, 1) for unit in range(1, 101)).encode()
    visited: list[str] = []

    def response(url: str, timeout: int) -> io.BytesIO:
        visited.append(url)
        assert timeout == 30
        return io.BytesIO(b"not telemetry" if len(visited) == 1 else valid)

    monkeypatch.setattr("urllib.request.urlopen", response)
    cached = download_raw_data(tmp_path / "raw")
    assert visited == RAW_DATA_URLS
    assert cached.read_bytes() == valid
    receipt = json.loads(cached.with_suffix(".source.json").read_text())
    assert receipt == {"url": RAW_DATA_URLS[1], "sha256": file_sha256(cached)}
    assert len(list(cached.parent.iterdir())) == 2


def test_invalid_downloads_leave_no_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed validation must not leave a partial cache that later runs would trust."""
    import io

    from scripts.download_data import download_raw_data

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))
    raw = tmp_path / "raw"
    with pytest.raises(TimeFFormatError):
        download_raw_data(raw)
    assert list(raw.iterdir()) == []


def test_preprocessing_never_downloads(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process local telemetry even when network use is forbidden."""

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Preprocessing must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    output = preprocess_data(str(raw_dir), str(tmp_path / "out"), 3, 3)
    assert output.is_file()


def test_missing_raw_requires_download_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing raw file gives an explicit command without creating partial data."""

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing raw data must not trigger an implicit download")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    output = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="scripts.download_data"):
        preprocess_data(str(tmp_path / "missing"), str(output))
    assert not output.exists()


def test_download_stage_validates_cache_without_preprocessing(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache validation does not create processed files or silently replace bad data."""
    from scripts.download_data import download_raw_data

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Existing raw data must not trigger network requests")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    raw_file = download_raw_data(raw_dir)
    assert set(raw_dir.parent.iterdir()) == {raw_dir}
    raw_file.write_text("invalid\n", encoding="utf-8")
    with pytest.raises(TimeFFormatError):
        download_raw_data(raw_dir)
    assert raw_file.read_text() == "invalid\n"
