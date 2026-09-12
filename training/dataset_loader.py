"""Lazy sensor-window loading with persisted training-only normalization."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import torch
from timenet.errors import TimeFFormatError, TimeFValidationError
from torch.utils.data import Dataset

from scripts.preprocess_data import ACTIVE_SENSORS, SPLIT_UNITS


class Tokenizer(Protocol):
    """Tokenizer operations required by the dataset and inference preparation."""

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token ID, when configured."""

    @property
    def pad_token_id(self) -> int | None:
        """Return the padding token ID, when configured."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text as token IDs."""


def _sensor_array(
    series: dict[str, Any], channels: tuple[str, ...], window_size: int
) -> np.ndarray:
    try:
        values = np.asarray([series[channel] for channel in channels], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise TimeFValidationError("Missing or non-numeric sensor channels") from exc
    if values.shape != (len(channels), window_size) or not np.isfinite(values).all():
        raise TimeFValidationError(
            f"Expected finite sensor values of shape {(len(channels), window_size)}"
        )
    return values


@dataclass(frozen=True)
class NormalizationStats:
    """Immutable training preprocessing metadata; constant channels use scale 1."""

    channels: tuple[str, ...]
    window_size: int
    means: tuple[float, ...]
    scales: tuple[float, ...]
    dataset_sha256: str
    training_sha256: str
    training_units: tuple[int, ...]
    sample_count_per_channel: int
    schema_version: int = 1
    fit_policy: str = "training_windows_with_overlap_population_std"

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.fit_policy != "training_windows_with_overlap_population_std"
        ):
            raise TimeFValidationError("Unsupported normalization schema or fit policy")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise TimeFValidationError("Normalization channels must be unique and nonempty")
        if type(self.window_size) is not int or self.window_size < 1:
            raise TimeFValidationError("window_size must be a positive integer")
        if len(self.means) != len(self.channels) or len(self.scales) != len(self.channels):
            raise TimeFValidationError("Normalization statistics do not match channel count")
        if (
            not np.isfinite(self.means).all()
            or not np.isfinite(self.scales).all()
            or min(self.scales) <= 0
        ):
            raise TimeFValidationError(
                "Normalization means/scales must be finite with positive scales"
            )
        if not self.training_units or not set(self.training_units) <= set(SPLIT_UNITS["train"]):
            raise TimeFValidationError("Normalization may only be fitted on training engines")
        if self.sample_count_per_channel < 1:
            raise TimeFValidationError("Normalization requires training samples")
        for digest in (self.dataset_sha256, self.training_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise TimeFValidationError("Invalid dataset/training SHA-256 digest")

    def save(self, path: str | Path) -> None:
        """Write preprocessing metadata beside a model or as a standalone artifact.

        Args:
            path: Destination JSON file.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and validate saved normalization without fitting any data.

        Args:
            path: Saved preprocessing JSON.

        Returns:
            Validated immutable normalization metadata.
        """
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in ("channels", "means", "scales", "training_units"):
                payload[key] = tuple(payload[key])
            return cls(**payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise TimeFFormatError(f"Invalid normalization artifact: {path}") from exc

    def transform(self, series: dict[str, Any]) -> torch.Tensor:
        """Normalize one sensor window without needing labels or a dataset file.

        Args:
            series: Raw channel names mapped to observed sensor values.

        Returns:
            Finite float32 tensor with shape [channels, window_size].
        """
        values = _sensor_array(series, self.channels, self.window_size)
        normalized = (values - np.asarray(self.means)[:, None]) / np.asarray(self.scales)[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            values32 = normalized.astype(np.float32)
        if not np.isfinite(values32).all():
            raise TimeFValidationError("Normalized sensor values exceed float32 range")
        return torch.from_numpy(values32)


def _tokenize(
    tokenizer: Tokenizer,
    prompt: str,
    target: str | None,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer.encode(f"<|prompt|>{prompt}\n<|response|>", add_special_tokens=True)
    ids = list(prompt_ids)
    labels = [-100] * len(ids)
    if target is not None:
        if tokenizer.eos_token_id is None:
            raise TimeFValidationError("Supervised tokenization requires an EOS token")
        response_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids.extend(response_ids)
        labels.extend(response_ids)
    if len(ids) > max_length:
        raise TimeFValidationError(
            f"Text requires {len(ids)} tokens, exceeds max_length={max_length}; "
            "increase max_length rather than silently dropping supervision"
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise TimeFValidationError("Tokenization requires a padding or EOS token")
    padding = max_length - len(ids)
    result = {
        "input_ids": torch.tensor(ids + [pad_id] * padding, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(ids) + [0] * padding, dtype=torch.long),
    }
    if target is not None:
        result["labels"] = torch.tensor(labels + [-100] * padding, dtype=torch.long)
    return result


def prepare_inputs(
    series: dict[str, Any],
    prompt: str,
    normalization: NormalizationStats,
    tokenizer: Tokenizer | None = None,
    max_length: int = 640,
) -> dict[str, Any]:
    """Prepare label-free inference inputs using saved training preprocessing.

    Args:
        series: Observed raw sensor window.
        prompt: Observation-only question, without the answer.
        normalization: Previously fitted training statistics.
        tokenizer: Optional language-model tokenizer.
        max_length: Padded text length; overflow raises instead of truncating.

    Returns:
        Sensor tensor and prompt, plus optional input IDs and attention mask.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise TimeFValidationError("A nonempty observation prompt is required")
    if type(max_length) is not int or max_length < 1:
        raise TimeFValidationError("max_length must be a positive integer")
    result = {"sensor_series": normalization.transform(series), "prompt": prompt}
    if tokenizer is not None:
        result.update(_tokenize(tokenizer, prompt, None, max_length))
    return result


class CMAPSSCoTDataset(Dataset[dict[str, Any]]):
    """Index JSONL byte offsets; read sensor windows only when requested.

    Training fits statistics if none are supplied. Validation/test require saved
    training statistics. By default only training returns targets; opt into
    include_targets for supervised validation loss or scoring, never generation.
    """

    def __init__(
        self,
        jsonl_path: str = "data/processed/windows.jsonl",
        split: str = "train",
        tokenizer: Tokenizer | None = None,
        max_length: int = 640,
        channels: list[str] | tuple[str, ...] = tuple(ACTIVE_SENSORS),
        window_size: int = 30,
        normalization: NormalizationStats | str | Path | None = None,
        include_targets: bool | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.channels = tuple(channels)
        self.window_size = window_size
        self.include_targets = split == "train" if include_targets is None else include_targets
        self._offsets: list[int] = []
        if split not in SPLIT_UNITS:
            raise TimeFValidationError(f"Unknown split: {split}")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise TimeFValidationError("Channels must be unique and nonempty")
        if (
            type(window_size) is not int
            or window_size < 1
            or type(max_length) is not int
            or max_length < 1
        ):
            raise TimeFValidationError("window_size and max_length must be positive integers")
        if isinstance(normalization, (str, Path)):
            normalization = NormalizationStats.load(normalization)
        if normalization is None and split != "train":
            raise TimeFValidationError(
                "Validation/test require normalization fitted on training data"
            )
        if normalization is not None and (
            normalization.channels != self.channels or normalization.window_size != window_size
        ):
            raise TimeFValidationError("Saved channel order or window size does not match loader")
        if not self.path.is_file():
            raise FileNotFoundError(
                "Processed data missing. Run: uv run python -m scripts.preprocess_data"
            )
        self._file_signature = self._signature()
        data_hash, train_hash = hashlib.sha256(), hashlib.sha256()
        count = 0
        means, m2 = np.zeros(len(self.channels)), np.zeros(len(self.channels))
        train_units: set[int] = set()
        seen_ids: set[str] = set()
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                data_hash.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_split, unit = record["split"], record["unit_number"]
                    if (
                        record_split not in SPLIT_UNITS
                        or type(unit) is not int
                        or unit not in SPLIT_UNITS[record_split]
                    ):
                        raise TimeFValidationError(
                            "Record engine ID does not belong to its declared split"
                        )
                    record_id = record["record_id"]
                    if not isinstance(record_id, str) or record_id in seen_ids:
                        raise TimeFValidationError("Record IDs must be unique strings")
                    seen_ids.add(record_id)
                    values = _sensor_array(record["series"], self.channels, window_size)
                except (KeyError, TypeError, ValueError) as exc:
                    raise TimeFFormatError(f"Invalid JSONL record at byte {offset}") from exc
                if record_split == split:
                    self._offsets.append(offset)
                if record_split == "train":
                    train_units.add(unit)
                    train_hash.update(line)
                    if normalization is None:
                        # Parallel-variance merge: stable, bounded-memory population moments.
                        batch_mean = values.mean(axis=1)
                        batch_m2 = ((values - batch_mean[:, None]) ** 2).sum(axis=1)
                        new_count = count + window_size
                        delta = batch_mean - means
                        m2 += batch_m2 + delta**2 * count * window_size / new_count
                        means += delta * window_size / new_count
                        count = new_count
        if self._signature() != self._file_signature:
            raise TimeFFormatError("Dataset changed while indexing")
        if not self._offsets:
            raise TimeFValidationError(f"No records in split {split}")
        if normalization is None:
            scales = np.sqrt(m2 / count)
            scales[scales < 1e-6] = 1.0
            normalization = NormalizationStats(
                channels=self.channels,
                window_size=window_size,
                means=tuple(means.tolist()),
                scales=tuple(scales.tolist()),
                dataset_sha256=data_hash.hexdigest(),
                training_sha256=train_hash.hexdigest(),
                training_units=tuple(sorted(train_units)),
                sample_count_per_channel=count,
            )
        elif (
            normalization.dataset_sha256 != data_hash.hexdigest()
            or normalization.training_sha256 != train_hash.hexdigest()
        ):
            raise TimeFValidationError(
                "Normalization artifact belongs to a different dataset; do not silently refit"
            )
        self.normalization = normalization

    def _signature(self) -> tuple[int, int]:
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._signature() != self._file_signature:
            raise TimeFFormatError("Dataset changed after indexing; construct a new loader")
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            item = json.loads(handle.readline())
        result = prepare_inputs(
            item["series"], item["prompt"], self.normalization, self.tokenizer, self.max_length
        )
        result.update({key: item[key] for key in ("record_id", "unit_number", "cycle")})
        if self.include_targets:
            try:
                rul = float(item["rul"])
                if not _valid_rul(rul):
                    raise ValueError("invalid RUL")
                if (
                    not isinstance(item["target"], str)
                    or not item["target"].strip()
                    or not isinstance(item["rationale"], str)
                ):
                    raise ValueError("invalid target text")
                target = f"{item['rationale']} {item['target']}"
            except (KeyError, TypeError, ValueError) as exc:
                raise TimeFValidationError(
                    "Supervised records require finite nonnegative RUL and target text"
                ) from exc
            result.update({"rul": torch.tensor(rul, dtype=torch.float32), "target": target})
            if self.tokenizer is not None:
                result.update(_tokenize(self.tokenizer, item["prompt"], target, self.max_length))
        return result


def _valid_rul(value: float) -> bool:
    """Check that a RUL label can be represented by a nonnegative float32.

    Args:
        value: Numeric label.

    Returns:
        Whether the label is finite, nonnegative, and representable.
    """
    return bool(np.isfinite(value) and 0 <= value <= np.finfo(np.float32).max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/windows.jsonl")
    parser.add_argument("--save-normalization", default="models/preprocessing.json")
    args = parser.parse_args()
    dataset = CMAPSSCoTDataset(jsonl_path=args.data_path)
    dataset.normalization.save(args.save_normalization)
    print(f"Fitted on {len(dataset)} training windows; saved {args.save_normalization}")
