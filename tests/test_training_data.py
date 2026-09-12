"""Focused tests for leakage-safe telemetry data preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.data import (
    CHANNELS,
    ChannelNormalizer,
    TelemetryCollator,
    TelemetryDataset,
    load_manifest,
    split_records,
)


def _record(path: Path, sample: str, patient: str, case: str, device: str, **extra):
    signal = np.stack([np.arange(8, dtype=np.float32) + offset for offset in range(4)])
    signal_path = path / f"{sample}.npy"
    np.save(signal_path, signal)
    record = {
        "sample_id": sample,
        "patient_id": patient,
        "case_id": case,
        "device_id": device,
        "recording_id": f"r-{sample}",
        "signal_path": signal_path.name,
        "channels": list(CHANNELS),
        "sample_rate_hz": 2,
        "time_base": "seconds-from-case-start",
        "window_start": 0,
        "window_end": 4,
        "query_time": 4,
        "prompt": "Assess this window.",
        "target": {
            "Observation": "A trend is present.",
            "Rationale": "The channels changed together.",
            "Recommendation": "Review the patient and verify the signal.",
        },
    }
    record.update(extra)
    return record


def _manifest(tmp_path: Path, records) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_connected_components_block_transitive_leakage(tmp_path):
    records = [
        _record(tmp_path, "a", "p1", "c1", "d1"),
        _record(tmp_path, "b", "p1", "c2", "d2"),  # patient bridge to a
        _record(tmp_path, "c", "p2", "c3", "d2"),  # device bridge to b
        _record(tmp_path, "d", "p3", "c4", "d3"),
        _record(tmp_path, "e", "p4", "c5", "d4", signal_path="d.npy"),
    ]
    loaded = load_manifest(_manifest(tmp_path, records))
    audit_path = tmp_path / "audit" / "split.json"
    train, validation, audit = split_records(loaded, seed=7, audit_path=audit_path)
    locations = {record["sample_id"]: "train" for record in train} | {
        record["sample_id"]: "validation" for record in validation
    }
    assert len({locations[name] for name in ("a", "b", "c")}) == 1
    assert locations["d"] == locations["e"]  # canonical signal-path identity
    assert audit["disjoint_verification"]["passed"] is True
    assert audit["actual_train_fraction"] + audit["actual_val_fraction"] == 1
    assert (
        json.loads(audit_path.read_text())["train_sample_ids"]
        == audit["train_sample_ids"]
    )


def test_missing_identifiers_are_rejected_or_explicitly_case_only(tmp_path):
    record = _record(tmp_path, "a", "p1", "c1", "d1")
    record["patient_id"] = None
    path = _manifest(tmp_path, [record])
    with pytest.raises(ValueError, match="requires patient_id"):
        load_manifest(path)
    assert load_manifest(path, allow_case_only=True)[0]["patient_id"] is None

    record["device_id"] = None
    path = _manifest(tmp_path, [record])
    with pytest.raises(ValueError, match="requires device_id"):
        load_manifest(path, allow_case_only=True)


def test_missing_optional_ids_do_not_form_a_shared_group(tmp_path):
    records = [
        _record(tmp_path, "a", "p1", "c1", "d1", case_id=None, recording_id=None),
        _record(tmp_path, "b", "p2", "c2", "d2", case_id=None, recording_id=None),
    ]
    loaded = load_manifest(_manifest(tmp_path, records))
    train, validation, audit = split_records(loaded)
    assert len(train) == len(validation) == 1
    assert audit["disjoint_verification"]["passed"]


def test_rejects_inconsistent_patient_metadata_and_future_window(tmp_path):
    first = _record(tmp_path, "a", "p1", "shared-case", "d1")
    second = _record(tmp_path, "b", "p2", "shared-case", "d2")
    with pytest.raises(ValueError, match="Inconsistent metadata"):
        load_manifest(_manifest(tmp_path, [first, second]))

    future = _record(tmp_path, "c", "p3", "c3", "d3", query_time=3)
    with pytest.raises(ValueError, match="post-query"):
        load_manifest(_manifest(tmp_path, [future]))


def test_rejects_unstructured_targets_complex_signals_and_zero_duration(tmp_path):
    invalid_target = _record(tmp_path, "a", "p1", "c1", "d1")
    invalid_target["target"]["Observation"] = None
    with pytest.raises(ValueError, match="sections must be strings"):
        load_manifest(_manifest(tmp_path, [invalid_target]))

    complex_signal = _record(tmp_path, "b", "p2", "c2", "d2")
    np.save(tmp_path / "b.npy", np.ones((4, 8), dtype=np.complex64))
    with pytest.raises(ValueError, match="real numeric"):
        load_manifest(_manifest(tmp_path, [complex_signal]))

    zero_duration = _record(tmp_path, "c", "p3", "c3", "d3", window_end=0, query_time=0)
    with pytest.raises(ValueError, match="window_start < window_end"):
        load_manifest(_manifest(tmp_path, [zero_duration]))


def test_train_only_normalizer_does_not_see_validation(tmp_path):
    train_record = _record(tmp_path, "train", "p1", "c1", "d1")
    validation_record = _record(tmp_path, "val", "p2", "c2", "d2")
    np.save(tmp_path / "train.npy", np.full((4, 8), 10, dtype=np.float32))
    np.save(tmp_path / "val.npy", np.full((4, 8), 10_000, dtype=np.float32))
    records = load_manifest(_manifest(tmp_path, [train_record, validation_record]))
    normalizer = ChannelNormalizer.fit([records[0]])
    assert normalizer.mean == (10.0, 10.0, 10.0, 10.0)
    assert normalizer.count == (8, 8, 8, 8)


def test_latest_crop_missing_mask_and_patch_padding(tmp_path):
    first = _record(tmp_path, "a", "p1", "c1", "d1")
    second = _record(tmp_path, "b", "p2", "c2", "d2")
    values = np.stack([np.arange(8, dtype=np.float32) + offset for offset in range(4)])
    values[1, 6] = np.nan
    np.save(tmp_path / "a.npy", values)
    records = load_manifest(_manifest(tmp_path, [first, second]))
    normalizer = ChannelNormalizer.fit(records, max_signal_length=5)
    assert normalizer.count[1] == 9  # NaN is excluded from finite-only estimates.
    dataset = TelemetryDataset(
        records, normalizer, max_signal_length=5, eos_token="<eos>"
    )
    item = dataset[0]
    assert len(item["time_series"]) == 4
    assert item["time_series"][0].shape == (5,)
    assert item["signal_valid_mask"].shape == (4, 5)
    assert item["signal_valid_mask"][1, 3].item() is False
    assert item["time_series"][1][3].item() == 0
    assert item["answer"].endswith("<eos>")
    assert "training-only clinical-scale" in item["time_series_text"][0]
    assert "retained duration 2.5 seconds" in item["pre_prompt"]
    assert "[1.5, 4)" in item["pre_prompt"]

    batch = TelemetryCollator(patch_size=4)([item, dataset[1]])
    assert batch["signals"].shape == (2, 4, 8)
    assert batch["signal_attention_mask"].shape == (2, 4, 8)
    assert batch["patch_attention_mask"].shape == (2, 4, 2)
    assert not batch["signal_attention_mask"][:, :, 5:].any()
    assert batch["patch_attention_mask"][:, :, 1].all()
    assert batch["opentslm_batch"][0]["time_series"].shape == (4, 8)
    assert batch["opentslm_batch"][0]["signal_length"] == 5


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, *, add_special_tokens, truncation):
        ids = [1] if add_special_tokens else []
        ids.extend(range(3, 3 + len(text.split())))
        return {"input_ids": ids}


def test_token_labels_mask_prompt_and_preserve_target(tmp_path):
    record = _record(tmp_path, "a", "p1", "c1", "d1")
    loaded = load_manifest(_manifest(tmp_path, [record]))
    item = TelemetryDataset(loaded, ChannelNormalizer.fit(loaded))[0]
    collator = TelemetryCollator(
        patch_size=4, tokenizer=_TinyTokenizer(), max_text_length=30
    )
    batch = collator([item])
    target_length = len(item["answer"].split())
    supervised = batch["labels"][0].ne(-100)
    assert supervised.sum().item() == target_length
    assert supervised.nonzero()[0].item() == 30 - target_length
    assert batch["attention_mask"].sum().item() == 30
