"""Leakage-safe data preparation for surgical telemetry language models.

Manifest format
---------------
The input is JSON Lines, one object per decision window. Required fields are::

    {
      "sample_id": "case-17-at-300s",
      "patient_id": "patient-8",
      "case_id": "case-17",
      "device_id": "monitor-4",
      "recording_id": "recording-17",
      "signal_path": "signals/case-17-at-300s.npy",
      "channels": ["ABP", "HR", "SpO2", "EtCO2"],
      "sample_rate_hz": 1.0,
      "time_base": "seconds-from-case-start",
      "window_start": 0.0,
      "window_end": 300.0,
      "query_time": 300.0,
      "prompt": "Analyze the last 5 minutes ...",
      "target": {
        "Observation": "...",
        "Rationale": "...",
        "Recommendation": "..."
      }
    }

``signal_path`` is relative to the manifest and stores a finite-or-missing
float array with shape ``[4, time]``. Patient identifiers are required by
default. Passing ``allow_case_only=True`` explicitly documents the weaker
assumption that a case identifier uniquely identifies one patient. Device
identity is also required by default because strict device-disjoint evaluation
is impossible without it.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F
from torch.utils.data import Dataset


CHANNELS = ("ABP", "HR", "SpO2", "EtCO2")
CHANNEL_UNITS = ("mmHg", "beats/min", "%", "mmHg")
TARGET_SECTIONS = ("Observation", "Rationale", "Recommendation")
IDENTITY_FIELDS = ("patient_id", "case_id", "device_id", "recording_id")


def _present(value: Any) -> str | None:
    """Return a normalized identifier, treating missing values as absent."""

    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"Identifiers must be strings or integers, got {value!r}")
    value = str(value).strip()
    return value or None


def _load_signal(path: str | os.PathLike[str]) -> np.ndarray:
    """Memory-map a signal array after checking its fixed channel layout."""

    try:
        signal = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load signal array {path}: {exc}") from exc
    if signal.ndim != 2 or signal.shape[0] != len(CHANNELS) or signal.shape[1] < 1:
        raise ValueError(
            f"Signal {path} must have shape [{len(CHANNELS)}, T>0], got {signal.shape}"
        )
    if not np.issubdtype(signal.dtype, np.number) or np.issubdtype(
        signal.dtype, np.complexfloating
    ):
        raise ValueError(f"Signal {path} must contain real numeric values")
    return signal


def _structured_target(value: Any, *, location: str) -> dict[str, str]:
    """Validate a target and normalize it to the three required sections."""

    if isinstance(value, Mapping):
        if set(value) != set(TARGET_SECTIONS):
            raise ValueError(
                f"{location}.target must contain exactly {list(TARGET_SECTIONS)}"
            )
        if any(not isinstance(value[section], str) for section in TARGET_SECTIONS):
            raise ValueError(f"{location}.target sections must be strings")
        target = {section: value[section].strip() for section in TARGET_SECTIONS}
    elif isinstance(value, str):
        pattern = re.compile(
            r"^Observation:\s*(.*?)\s*Rationale:\s*(.*?)\s*Recommendation:\s*(.*?)\s*$",
            flags=re.DOTALL,
        )
        match = pattern.match(value.strip())
        if match is None:
            raise ValueError(
                f"{location}.target text must have Observation, Rationale, and Recommendation sections"
            )
        target = dict(zip(TARGET_SECTIONS, (part.strip() for part in match.groups())))
    else:
        raise ValueError(f"{location}.target must be an object or structured string")
    if any(not target[section] for section in TARGET_SECTIONS):
        raise ValueError(f"{location}.target sections must all be non-empty")
    return target


def _validate_metadata_consistency(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject contradictory patient/case/recording identity metadata."""

    def assert_single(key_field: str, value_field: str) -> None:
        associations: dict[str, set[str | None]] = {}
        for record in records:
            key = record.get(key_field)
            if key is not None:
                associations.setdefault(key, set()).add(record.get(value_field))
        bad = {key: values for key, values in associations.items() if len(values) > 1}
        if bad:
            example, values = next(iter(bad.items()))
            raise ValueError(
                f"Inconsistent metadata: {key_field}={example!r} maps to multiple "
                f"{value_field} values: {sorted(map(str, values))}"
            )

    assert_single("case_id", "patient_id")
    for field in ("patient_id", "case_id", "device_id"):
        assert_single("recording_id", field)


def load_manifest(
    path: str | os.PathLike[str],
    *,
    require_device_id: bool = True,
    allow_case_only: bool = False,
) -> list[dict[str, Any]]:
    """Load and validate a telemetry JSONL manifest.

    Signal paths are resolved relative to the manifest. Temporal fields must be
    finite values on the named common time base and satisfy
    ``window_start < window_end <= query_time``. The half-open signal duration
    must agree tightly with the timestamps and sampling rate.

    Args:
        path: JSONL manifest path.
        require_device_id: Require device identity for device-disjoint splits.
        allow_case_only: Permit missing ``patient_id`` only when ``case_id`` is
            present. This is an explicit assertion that each case belongs to a
            single, otherwise unrepresented patient.
    """

    manifest_path = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read manifest {manifest_path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{manifest_path}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {location}: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{location} must contain a JSON object")
        record = dict(raw)
        for field in IDENTITY_FIELDS:
            record[field] = _present(record.get(field))

        if record["patient_id"] is None:
            if not allow_case_only or record["case_id"] is None:
                raise ValueError(
                    f"{location} requires patient_id; pass allow_case_only=True only "
                    "when case_id uniquely identifies a patient"
                )
        if require_device_id and record["device_id"] is None:
            raise ValueError(f"{location} requires device_id for strict splitting")

        sample_id = _present(record.get("sample_id")) or f"line-{line_number}"
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id {sample_id!r} at {location}")
        seen_samples.add(sample_id)
        record["sample_id"] = sample_id

        signal_name = record.get("signal_path")
        if not isinstance(signal_name, str) or not signal_name.strip():
            raise ValueError(f"{location}.signal_path must be a non-empty string")
        signal_path = Path(signal_name).expanduser()
        if not signal_path.is_absolute():
            signal_path = manifest_path.parent / signal_path
        signal_path = signal_path.resolve()
        signal = _load_signal(signal_path)
        record["signal_path"] = str(signal_path)

        if tuple(record.get("channels", ())) != CHANNELS:
            raise ValueError(f"{location}.channels must be exactly {list(CHANNELS)}")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{location}.prompt must be a non-empty string")
        record["prompt"] = prompt.strip()
        record["target"] = _structured_target(record.get("target"), location=location)

        time_base = record.get("time_base")
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"{location}.time_base must name the common time base")
        record["time_base"] = time_base.strip()
        for field in ("sample_rate_hz", "window_start", "window_end", "query_time"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{location}.{field} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{location}.{field} must be finite")
            record[field] = value
        if record["sample_rate_hz"] <= 0:
            raise ValueError(f"{location}.sample_rate_hz must be positive")
        if not record["window_start"] < record["window_end"] <= record["query_time"]:
            raise ValueError(
                f"{location} violates window_start < window_end <= query_time; "
                "post-query samples would leak future information"
            )
        duration = record["window_end"] - record["window_start"]
        signal_duration = signal.shape[1] / record["sample_rate_hz"]
        if not math.isclose(duration, signal_duration, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(
                f"{location} has {signal.shape[1]} samples but timestamps imply "
                f"{duration * record['sample_rate_hz']:g} at "
                f"{record['sample_rate_hz']:g} Hz for a half-open window"
            )
        records.append(record)

    if not records:
        raise ValueError(f"Manifest {manifest_path} contains no records")
    _validate_metadata_consistency(records)
    return records


class _UnionFind:
    """Small union-find used to bridge every shared leakage identifier."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _identity_sets(records: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    sets = {field: set() for field in (*IDENTITY_FIELDS, "signal_path")}
    for record in records:
        for field in sets:
            value = record.get(field)
            if value is not None:
                sets[field].add(str(value))
    return sets


def split_records(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    audit_path: str | os.PathLike[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split records through connected leakage groups with GroupShuffleSplit.

    Two rows enter the same connected component if they share any patient,
    case, device, recording, or canonical signal path. Transitive bridges are
    retained, so a shared device cannot leak through otherwise distinct cases.
    Missing optional identifiers are ignored rather than collapsed into a fake
    shared group. This randomized grouped split makes no claim about global
    chronological ordering.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required for a split")

    union_find = _UnionFind(len(records))
    owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field in (*IDENTITY_FIELDS, "signal_path"):
            value = record.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            key = (field, str(value))
            if key in owner:
                union_find.union(index, owner[key])
            else:
                owner[key] = index

    roots = [union_find.find(index) for index in range(len(records))]
    unique_roots = sorted(set(roots))
    if len(unique_roots) < 2:
        raise ValueError(
            "Leakage-safe splitting is impossible: all records form one connected identity group"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_indices, val_indices = next(
        splitter.split(np.zeros(len(records)), groups=roots)
    )
    train = [dict(records[index]) for index in train_indices]
    val = [dict(records[index]) for index in val_indices]

    train_identities = _identity_sets(train)
    val_identities = _identity_sets(val)
    intersections = {
        field: sorted(train_identities[field] & val_identities[field])
        for field in train_identities
    }
    passed = all(not values for values in intersections.values())
    if not passed:  # Defensive: this indicates a bug in component construction.
        raise RuntimeError(f"Split leakage verification failed: {intersections}")

    component_labels = {
        root: f"component-{position}" for position, root in enumerate(unique_roots)
    }
    audit = {
        "schema_version": 1,
        "strategy": "connected_components_then_group_shuffle_split",
        "seed": seed,
        "requested_val_fraction": val_fraction,
        "actual_train_fraction": len(train) / len(records),
        "actual_val_fraction": len(val) / len(records),
        "train_sample_ids": [record["sample_id"] for record in train],
        "val_sample_ids": [record["sample_id"] for record in val],
        "train_component_ids": sorted(
            {component_labels[roots[index]] for index in train_indices}
        ),
        "val_component_ids": sorted(
            {component_labels[roots[index]] for index in val_indices}
        ),
        "disjoint_verification": {"passed": passed, "intersections": intersections},
        "chronology": "No global chronological-order claim; all windows from linked identities stay together.",
    }
    if audit_path is not None:
        destination = Path(audit_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return train, val, audit


@dataclass(frozen=True)
class ChannelNormalizer:
    """Train-fitted per-channel z-score statistics.

    Call :meth:`fit` with training records only. Non-finite observations do not
    contribute to the streaming estimates and are zero-filled after scaling.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: tuple[int, ...]
    channels: tuple[str, ...] = CHANNELS
    max_signal_length: int | None = None

    def __post_init__(self) -> None:
        if tuple(self.channels) != CHANNELS:
            raise ValueError(f"Normalizer channels must be exactly {list(CHANNELS)}")
        size = len(self.channels)
        if not (len(self.mean) == len(self.std) == len(self.count) == size):
            raise ValueError("Normalizer statistics must match channel count")
        if any(not math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("Normalizer statistics must be finite")
        if any(value <= 0 for value in self.std) or any(
            value <= 0 for value in self.count
        ):
            raise ValueError(
                "Normalizer standard deviations and counts must be positive"
            )

    @classmethod
    def fit(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        max_signal_length: int | None = None,
        chunk_size: int = 1_000_000,
        min_std: float = 1e-6,
    ) -> "ChannelNormalizer":
        """Fit finite-only statistics by streaming over the supplied records."""

        if not records:
            raise ValueError("Cannot fit normalization on an empty training split")
        if max_signal_length is not None and max_signal_length <= 0:
            raise ValueError("max_signal_length must be positive")
        if chunk_size <= 0 or min_std <= 0:
            raise ValueError("chunk_size and min_std must be positive")

        counts = np.zeros(len(CHANNELS), dtype=np.int64)
        means = np.zeros(len(CHANNELS), dtype=np.float64)
        m2 = np.zeros(len(CHANNELS), dtype=np.float64)
        for record in records:
            signal = _load_signal(record["signal_path"])
            start = (
                max(0, signal.shape[1] - max_signal_length) if max_signal_length else 0
            )
            for channel_index in range(len(CHANNELS)):
                channel = signal[channel_index, start:]
                for offset in range(0, channel.shape[0], chunk_size):
                    block = np.asarray(
                        channel[offset : offset + chunk_size], dtype=np.float64
                    )
                    finite = block[np.isfinite(block)]
                    block_count = finite.size
                    if block_count == 0:
                        continue
                    block_mean = float(finite.mean(dtype=np.float64))
                    block_m2 = float(
                        np.square(finite - block_mean).sum(dtype=np.float64)
                    )
                    old_count = int(counts[channel_index])
                    total = old_count + block_count
                    delta = block_mean - means[channel_index]
                    means[channel_index] += delta * block_count / total
                    m2[channel_index] += (
                        block_m2 + delta * delta * old_count * block_count / total
                    )
                    counts[channel_index] = total
        missing = [CHANNELS[index] for index, count in enumerate(counts) if count == 0]
        if missing:
            raise ValueError(f"No finite training values for channels: {missing}")
        stds = np.sqrt(m2 / counts)
        stds = np.maximum(stds, min_std)
        return cls(
            mean=tuple(float(value) for value in means),
            std=tuple(float(value) for value in stds),
            count=tuple(int(value) for value in counts),
            max_signal_length=max_signal_length,
        )

    def transform(self, signal: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize one ``[C,T]`` signal and return data plus finite mask."""

        array = np.asarray(signal)
        if array.ndim != 2 or array.shape[0] != len(self.channels):
            raise ValueError(
                f"Expected [{len(self.channels)}, T] signal, got {array.shape}"
            )
        valid = np.isfinite(array)
        normalized64 = np.zeros(array.shape, dtype=np.float64)
        means = np.asarray(self.mean, dtype=np.float64)[:, None]
        stds = np.asarray(self.std, dtype=np.float64)[:, None]
        np.subtract(array, means, out=normalized64, where=valid)
        np.divide(normalized64, stds, out=normalized64, where=valid)
        float32_max = np.finfo(np.float32).max
        if not np.isfinite(normalized64).all() or np.any(
            np.abs(normalized64) > float32_max
        ):
            raise ValueError(
                "Normalization produced values outside the finite float32 range"
            )
        normalized = normalized64.astype(np.float32)
        return torch.from_numpy(normalized), torch.from_numpy(valid.copy())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable training normalization metadata."""

        return {
            "schema_version": 1,
            "channels": list(self.channels),
            "mean": list(self.mean),
            "std": list(self.std),
            "count": list(self.count),
            "max_signal_length": self.max_signal_length,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialize statistics for exact reuse by validation and inference."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ChannelNormalizer":
        """Load statistics previously written by :meth:`save`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported normalization metadata schema")
        return cls(
            channels=tuple(payload["channels"]),
            mean=tuple(payload["mean"]),
            std=tuple(payload["std"]),
            count=tuple(payload["count"]),
            max_signal_length=payload.get("max_signal_length"),
        )


class TelemetryDataset(Dataset):
    """Load normalized four-channel windows as OpenTSLM prompt dictionaries."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        normalizer: ChannelNormalizer,
        *,
        max_signal_length: int | None = None,
        eos_token: str = "",
    ) -> None:
        if not records:
            raise ValueError("TelemetryDataset requires at least one record")
        if max_signal_length is not None and max_signal_length <= 0:
            raise ValueError("max_signal_length must be positive")
        self.records = [dict(record) for record in records]
        self.normalizer = normalizer
        self.max_signal_length = max_signal_length
        self.eos_token = eos_token

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        signal = _load_signal(record["signal_path"])
        if self.max_signal_length is not None:
            signal = signal[:, -self.max_signal_length :]
        normalized, valid = self.normalizer.transform(signal)
        target = record["target"]
        answer = "\n".join(
            f"{section}: {target[section]}" for section in TARGET_SECTIONS
        )
        if self.eos_token and not answer.endswith(self.eos_token):
            answer += self.eos_token
        descriptions = [
            (
                f"{channel} ({unit}); normalized using training-only clinical-scale "
                f"mean={mean:.6g}, std={std:.6g}; zero denotes missing or the training mean:"
            )
            for channel, unit, mean, std in zip(
                self.normalizer.channels,
                CHANNEL_UNITS,
                self.normalizer.mean,
                self.normalizer.std,
            )
        ]
        retained_duration = normalized.shape[1] / record["sample_rate_hz"]
        retained_start = record["window_end"] - retained_duration
        query_gap = record["query_time"] - record["window_end"]
        temporal_context = (
            f"Telemetry context: {record['sample_rate_hz']:g} Hz sampling; retained "
            f"half-open window [{retained_start:g}, {record['window_end']:g}) on "
            f"{record['time_base']}; retained duration {retained_duration:g} seconds; "
            f"query is {query_gap:g} seconds after the window end."
        )
        return {
            "pre_prompt": f"{record['prompt']}\n{temporal_context}",
            "post_prompt": "Respond with Observation, Rationale, and Recommendation sections.",
            "answer": answer,
            "time_series_text": descriptions,
            "time_series": [normalized[channel] for channel in range(len(CHANNELS))],
            "signal_valid_mask": valid,
            "signal_length": normalized.shape[1],
            "sample_id": record["sample_id"],
            "normalization": self.normalizer.to_dict(),
        }


class TelemetryCollator:
    """Patch-align OpenTSLM signals and optionally tokenize supervised text.

    The result contains ``opentslm_batch`` for ``OpenTSLM.compute_loss`` and
    dense tensor aliases for adapter implementations. Masks are per channel so
    internal missing observations remain distinguishable from real normalized
    zeros. Token labels mask both prompt and padding with ``-100``. When a text
    limit is reached, old prompt tokens are removed first so the full target is
    preserved; targets longer than the limit are rejected.
    """

    def __init__(
        self,
        *,
        patch_size: int = 4,
        max_signal_length: int | None = None,
        tokenizer: Any | None = None,
        max_text_length: int | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if max_signal_length is not None and max_signal_length <= 0:
            raise ValueError("max_signal_length must be positive")
        if tokenizer is not None and (max_text_length is None or max_text_length <= 0):
            raise ValueError("A positive max_text_length is required with a tokenizer")
        self.patch_size = patch_size
        self.max_signal_length = max_signal_length
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    @staticmethod
    def _prompt_text(item: Mapping[str, Any]) -> str:
        chunks = [item["pre_prompt"]]
        for description in item["time_series_text"]:
            chunks.extend((description, "<time_series>"))
        chunks.append(item["post_prompt"])
        return "\n".join(chunks)

    @staticmethod
    def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = tokenizer(
            text, add_special_tokens=add_special_tokens, truncation=False
        )
        ids = (
            encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _tokenize(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        assert self.tokenizer is not None and self.max_text_length is not None
        rows: list[tuple[list[int], list[int]]] = []
        for item in batch:
            prompt_ids = self._token_ids(
                self.tokenizer, self._prompt_text(item), add_special_tokens=True
            )
            target_ids = self._token_ids(
                self.tokenizer, item["answer"], add_special_tokens=False
            )
            if len(target_ids) > self.max_text_length:
                raise ValueError(
                    f"Target for sample {item['sample_id']!r} exceeds max_text_length; "
                    "increase the limit instead of truncating clinical rationale"
                )
            prompt_budget = self.max_text_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            rows.append((prompt_ids, target_ids))
        width = max(len(prompt) + len(target) for prompt, target in rows)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id")
        input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for row_index, (prompt, target) in enumerate(rows):
            combined = prompt + target
            input_ids[row_index, : len(combined)] = torch.tensor(combined)
            attention_mask[row_index, : len(combined)] = 1
            labels[row_index, len(prompt) : len(combined)] = torch.tensor(target)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        prepared: list[dict[str, Any]] = []
        lengths: list[int] = []
        for source in batch:
            item = dict(source)
            series = torch.stack(
                [torch.as_tensor(ts, dtype=torch.float32) for ts in item["time_series"]]
            )
            valid = torch.as_tensor(item["signal_valid_mask"], dtype=torch.bool)
            if (
                series.shape != valid.shape
                or series.ndim != 2
                or series.shape[0] != len(CHANNELS)
            ):
                raise ValueError(
                    "time_series and signal_valid_mask must both have shape [4, T]"
                )
            if self.max_signal_length is not None:
                series = series[:, -self.max_signal_length :]
                valid = valid[:, -self.max_signal_length :]
            item["time_series"] = series
            item["signal_valid_mask"] = valid
            item["signal_length"] = series.shape[1]
            lengths.append(series.shape[1])
            prepared.append(item)

        padded_length = math.ceil(max(lengths) / self.patch_size) * self.patch_size
        for item in prepared:
            padding = padded_length - item["time_series"].shape[1]
            item["time_series"] = F.pad(item["time_series"], (0, padding), value=0.0)
            item["signal_valid_mask"] = F.pad(
                item["signal_valid_mask"], (0, padding), value=False
            )
        signals = torch.stack([item["time_series"] for item in prepared])
        valid_mask = torch.stack([item["signal_valid_mask"] for item in prepared])
        result: dict[str, Any] = {
            "opentslm_batch": prepared,
            "signals": signals,
            "signal_attention_mask": valid_mask,
            "patch_attention_mask": valid_mask.reshape(
                len(prepared), len(CHANNELS), -1, self.patch_size
            ).any(dim=-1),
            "sample_ids": [item["sample_id"] for item in prepared],
        }
        if self.tokenizer is not None:
            result.update(self._tokenize(prepared))
        return result


__all__ = [
    "CHANNELS",
    "ChannelNormalizer",
    "TelemetryCollator",
    "TelemetryDataset",
    "load_manifest",
    "split_records",
]
