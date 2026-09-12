"""Tests for exact dataset fingerprints and restart-safe artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.provenance import dataset_fingerprint, verify_run_artifacts


def _dataset(tmp_path: Path):
    first = tmp_path / "z.npy"
    second = tmp_path / "a.npy"
    np.save(first, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    np.save(second, np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"target":"first"}\n', encoding="utf-8")
    records = [
        {"signal_path": str(first)},
        {"signal_path": str(second)},
        {"signal_path": str(first)},  # distinct files are hashed only once
    ]
    return records, manifest, first, second


def test_fingerprint_detects_target_manifest_change(tmp_path):
    records, manifest, _, _ = _dataset(tmp_path)
    before = dataset_fingerprint(records, manifest)
    manifest.write_text('{"target":"changed"}\n', encoding="utf-8")
    after = dataset_fingerprint(records, manifest)
    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert before["content_sha256"] != after["content_sha256"]
    assert before["signal_files"] == after["signal_files"]


def test_fingerprint_detects_signal_change_with_identical_statistics(tmp_path):
    records, manifest, first, _ = _dataset(tmp_path)
    before = dataset_fingerprint(records, manifest)
    original = np.load(first)
    np.save(first, original[:, ::-1])  # same mean/std, different source bytes
    after = dataset_fingerprint(records, manifest)
    assert before["manifest_sha256"] == after["manifest_sha256"]
    assert before["content_sha256"] != after["content_sha256"]
    before_hashes = {item["path"]: item["sha256"] for item in before["signal_files"]}
    after_hashes = {item["path"]: item["sha256"] for item in after["signal_files"]}
    assert before_hashes[str(first.resolve())] != after_hashes[str(first.resolve())]
    assert [item["path"] for item in after["signal_files"]] == sorted(after_hashes)


def test_resume_rejects_missing_source_artifact_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    destination = tmp_path / "new-run"
    with pytest.raises(RuntimeError, match="Resume artifact is missing"):
        verify_run_artifacts(
            destination,
            {"dataset_fingerprint.json": {"value": 1}},
            resume_from=checkpoint,
        )
    assert not destination.exists()


def test_resume_rejects_changed_artifact_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    verify_run_artifacts(
        source, {"dataset_fingerprint.json": {"content_sha256": "old"}}, None
    )
    destination = tmp_path / "new-run"
    with pytest.raises(RuntimeError, match="does not match current inputs"):
        verify_run_artifacts(
            destination,
            {"dataset_fingerprint.json": {"content_sha256": "changed"}},
            resume_from=checkpoint,
        )
    assert not destination.exists()


def test_resume_to_separate_output_copies_verified_artifacts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    artifacts = {
        "dataset_fingerprint.json": {"sha256": "abc"},
        "split_audit.json": {"seed": 42},
    }
    verify_run_artifacts(source, artifacts, resume_from=None)

    destination = tmp_path / "continued"
    paths = verify_run_artifacts(destination, artifacts, resume_from=checkpoint)
    assert set(paths) == set(artifacts)
    for name, payload in artifacts.items():
        assert json.loads((destination / name).read_text()) == payload

    conflicting = destination / "normalization.json"
    conflicting.write_text('{"mean":[999]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        verify_run_artifacts(
            destination,
            {**artifacts, "normalization.json": {"mean": [0]}},
            resume_from=None,
        )
    assert json.loads(conflicting.read_text()) == {"mean": [999]}
