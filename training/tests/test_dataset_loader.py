"""Offline tests for training-only statistics, lazy access, and text masking."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from timenet.errors import TimeFFormatError, TimeFValidationError
from torch import nn
from torch.utils.data import DataLoader

from training.dataset_loader import CMAPSSCoTDataset, NormalizationStats, prepare_inputs

CHANNELS = ("sensor_2", "sensor_3")


class TinyTokenizer:
    """Predictable offline tokenizer with identical padding and EOS IDs."""

    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Map whitespace-separated words to IDs for masking tests."""
        return ([1] if add_special_tokens else []) + list(range(10, 10 + len(text.split())))


@pytest.fixture
def records() -> list[dict[str, Any]]:
    """Supply two training windows and extreme held-out windows."""
    result = []
    for index, (unit, split, values) in enumerate(
        [
            (1, "train", [1.0, 2.0, 3.0]),
            (1, "train", [3.0, 4.0, 5.0]),
            (71, "validation", [100.0, 200.0, 300.0]),
            (81, "test", [1000.0, 2000.0, 3000.0]),
        ]
    ):
        result.append(
            {
                "record_id": f"window-{index}",
                "unit_number": unit,
                "cycle": index + 3,
                "split": split,
                "series": {"sensor_2": values, "sensor_3": [7.0, 7.0, 7.0]},
                "prompt": "Estimate remaining life",
                "rul": 20,
                "rationale": "Observed sensor drift.",
                "target": "Projected RUL: 20 cycles.",
            }
        )
    return result


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write fixture JSONL and return its path."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def dataset(path: Path, **kwargs: Any) -> CMAPSSCoTDataset:
    """Construct a three-cycle fixture loader."""
    return CMAPSSCoTDataset(str(path), channels=CHANNELS, window_size=3, **kwargs)


def test_training_only_population_statistics_and_roundtrip(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    train = dataset(path)
    expected = np.asarray([1, 2, 3, 3, 4, 5], dtype=float)
    np.testing.assert_allclose(train.normalization.means, [expected.mean(), 7])
    np.testing.assert_allclose(train.normalization.scales, [expected.std(), 1])
    assert train.normalization.sample_count_per_channel == 6
    assert train.normalization.training_units == (1,)
    saved = tmp_path / "model/preprocessing.json"
    train.normalization.save(saved)
    reloaded = NormalizationStats.load(saved)
    assert reloaded == train.normalization
    assert torch.equal(
        dataset(path, normalization=saved)[0]["sensor_series"], train[0]["sensor_series"]
    )
    assert torch.equal(train[0]["sensor_series"][1], torch.zeros(3))
    for split in ("validation", "test"):
        heldout = dataset(path, split=split, normalization=saved)
        assert heldout.normalization == reloaded
        assert "rul" not in heldout[0] and "target" not in heldout[0]
        raw = records[2 if split == "validation" else 3]["series"]
        assert torch.equal(heldout[0]["sensor_series"], reloaded.transform(raw))


def test_heldout_changes_cannot_affect_statistics(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    original = dataset(path).normalization
    records[-1]["series"]["sensor_2"] = [-100000.0, 0.0, 500000.0]
    records[-2]["rul"] = 999
    write_records(path, records)
    changed = dataset(path).normalization
    assert changed.means == original.means and changed.scales == original.scales
    assert changed.training_sha256 == original.training_sha256
    assert changed.dataset_sha256 != original.dataset_sha256
    with pytest.raises(TimeFValidationError, match="different dataset"):
        dataset(path, split="test", normalization=original)


def test_heldout_requires_training_statistics(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    for split in ("validation", "test"):
        with pytest.raises(TimeFValidationError, match="require normalization"):
            dataset(path, split=split)


def test_lazy_access_and_batching(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    train = dataset(path, tokenizer=TinyTokenizer(), max_length=40)
    assert not hasattr(train, "records")
    assert all(isinstance(offset, int) for offset in train._offsets)
    batch = next(iter(DataLoader(train, batch_size=2)))
    assert batch["sensor_series"].shape == (2, 2, 3)
    assert batch["input_ids"].shape == (2, 40)
    assert train[-1]["record_id"] == records[1]["record_id"]
    with pytest.raises(IndexError):
        train[2]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(TimeFFormatError, match="changed after indexing"):
        train[0]


def test_prompt_padding_and_eos_masking(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    tokenizer = TinyTokenizer()
    sample = dataset(path, tokenizer=tokenizer, max_length=40)[0]
    prompt_len = len(tokenizer.encode(f"<|prompt|>{records[0]['prompt']}\n<|response|>"))
    active = int(sample["attention_mask"].sum())
    assert (sample["labels"][:prompt_len] == -100).all()
    assert torch.equal(sample["labels"][prompt_len:active], sample["input_ids"][prompt_len:active])
    assert sample["labels"][active - 1] == tokenizer.eos_token_id
    assert (sample["labels"][active:] == -100).all()
    assert (sample["attention_mask"][active:] == 0).all()
    # EOS is trainable even when padding uses that same token ID.
    assert sample["attention_mask"][active - 1] == 1
    with pytest.raises(TimeFValidationError, match="silently dropping"):
        dataset(path, tokenizer=tokenizer, max_length=6)[0]


def test_prediction_inputs_do_not_depend_on_targets(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    stats = dataset(path).normalization
    inference = dataset(path, split="test", normalization=stats, tokenizer=TinyTokenizer())
    sample = inference[0]
    assert not {"rul", "target", "labels", "rationale", "status"} & sample.keys()
    observed = prepare_inputs(records[-1]["series"], records[-1]["prompt"], stats, TinyTokenizer())
    assert torch.equal(sample["input_ids"], observed["input_ids"])
    assert torch.equal(sample["sensor_series"], observed["sensor_series"])
    supervised = dataset(
        path,
        split="validation",
        normalization=stats,
        include_targets=True,
        tokenizer=TinyTokenizer(),
    )
    assert {"labels", "rul", "target"} <= supervised[0].keys()
    # Standalone inference requires no dataset file or labels.
    path.unlink()
    assert torch.equal(observed["sensor_series"], stats.transform(records[-1]["series"]))


@pytest.mark.parametrize("bad_values", [[1, 2], [1, 2, float("nan")], [1, 2, "bad"]])
def test_invalid_sensor_values(tmp_path: Path, records: list, bad_values: list) -> None:
    records[0]["series"]["sensor_2"] = bad_values
    with pytest.raises((TimeFValidationError, TimeFFormatError)):
        dataset(write_records(tmp_path / "bad.jsonl", records))


def test_invalid_channels_splits_and_metadata(tmp_path: Path, records: list) -> None:
    path = write_records(tmp_path / "windows.jsonl", records)
    stats = dataset(path).normalization
    with pytest.raises(TimeFValidationError, match="channel order"):
        CMAPSSCoTDataset(str(path), channels=CHANNELS[::-1], window_size=3, normalization=stats)
    with pytest.raises(TimeFValidationError):
        replace(stats, scales=(0.0, 1.0))
    with pytest.raises(TimeFValidationError):
        replace(stats, training_units=(81,))
    records[-1]["split"] = "train"
    with pytest.raises((TimeFValidationError, TimeFFormatError)):
        dataset(write_records(path, records))


def test_model_forwards_prefix_and_padding_masks() -> None:
    """Use a tiny local model stub to check integration without downloading weights."""
    from training.train_opentslm import AeroGuardTSLM, TimeSeriesPatchEncoder

    class RecordingLLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(30, 4)
            self.received: dict[str, Any] = {}

        def get_input_embeddings(self) -> nn.Embedding:
            return self.embedding

        def forward(self, **kwargs: Any) -> SimpleNamespace:
            self.received = kwargs
            return SimpleNamespace(
                loss=kwargs["inputs_embeds"].sum() * 0, logits=torch.zeros(1, 5, 30)
            )

    model = AeroGuardTSLM.__new__(AeroGuardTSLM)
    nn.Module.__init__(model)
    model_as_any: Any = model
    model_as_any.llm = RecordingLLM()
    model.ts_encoder = TimeSeriesPatchEncoder(2, 3, 4)
    model_as_any.rul_head = nn.Linear(4, 1)
    model(
        torch.ones(1, 2, 3),
        torch.tensor([[1, 10, 2, 2]]),
        labels=torch.tensor([[-100, 10, 2, -100]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )
    assert model.llm.received["attention_mask"].tolist() == [[1, 1, 1, 1, 0]]
    assert model.llm.received["labels"].tolist() == [[-100, -100, 10, 2, -100]]
