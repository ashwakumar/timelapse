#!/usr/bin/env python3
"""OpenTSLM dataset adapter for the prepared VitalDB hypotension windows.

The adapter returns the dictionary contract consumed by OpenTSLM at upstream
commit 2968f4b891baab4307f7e9d0043e87677b593a30.  It deliberately keeps the
future-derived class in ``answer`` only.  Input tensors contain the normalized
20-second histories and explicit observation-mask channels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


UPSTREAM_OPENTSLM_COMMIT = "2968f4b891baab4307f7e9d0043e87677b593a30"
SPLITS = ("train", "validation", "test")
REQUIRED_ARRAYS = (
    "x_values",
    "x_mask",
    "y",
    "case_id",
    "subject_id",
    "cutoff_sec",
    "future_map_coverage",
)

# Units are taken from the VitalDB Open Dataset parameter table.  These describe
# the physical values before the saved train-only median/IQR normalization.
PARAMETER_UNITS = {
    "MAP": "mmHg",
    "RemiEffectSite": "ng/mL",
    "RR_CO2": "/min",
    "RemiRate": "mL/hr",
    "BIS_SQI": "%",
    "InCO2": "mmHg",
    "EtCO2": "mmHg",
    "BIS_EMG": "dB",
    "BIS": "unitless",
    "BIS_SR": "%",
    "ST_II": "mm",
    "Temperature": "degC",
    "FeO2": "%",
    "FiO2": "%",
    "BIS_SEF": "Hz",
    "SpO2": "%",
    "HR": "/min",
}

ANSWER_TEXT = {
    "within_3": "hypotension within 3 minutes",
    "within_5": "hypotension within 5 minutes",
    "within_10": "hypotension within 10 minutes",
    "within_15": "hypotension within 15 minutes",
    "none_within_15": "no hypotension within 15 minutes",
}

POST_PROMPT = """Respond in four lines headed INTERPRET, ANTICIPATE, ACT, and
Answer. ACT should request a signal-quality and patient reassessment, not a
treatment. The final answer must be one of: hypotension within 3 minutes;
hypotension within 5 minutes; hypotension within 10 minutes; hypotension within
15 minutes; no hypotension within 15 minutes."""


class PreparedVitalDBCorpus:
    """Validated access to one ``data/lstm/top_N`` directory."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing prepared dataset manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        self.parameters = list(self.manifest.get("parameters_in_tensor_order", ()))
        if not self.parameters:
            raise ValueError("manifest.json has no parameters_in_tensor_order")
        unknown_units = sorted(set(self.parameters) - set(PARAMETER_UNITS))
        if unknown_units:
            raise ValueError(
                "No verified VitalDB unit is recorded for: " + ", ".join(unknown_units)
            )

        labels = self.manifest.get("labels", {})
        self.labels = {int(index): str(label) for index, label in labels.items()}
        if set(self.labels.values()) != set(ANSWER_TEXT):
            raise ValueError(f"Unexpected label mapping: {self.labels}")

        normalization = self.manifest.get("normalization", {})
        medians = normalization.get("median", {})
        scales = normalization.get("iqr_scale", {})
        self.median = np.asarray([medians[p] for p in self.parameters], dtype=np.float32)
        self.scale = np.asarray([scales[p] for p in self.parameters], dtype=np.float32)
        if not (np.isfinite(self.median).all() and np.isfinite(self.scale).all()):
            raise ValueError("Normalization statistics must be finite")
        if not (self.scale > 0).all():
            raise ValueError("Every normalization scale must be positive")

        self.interval_sec = int(self.manifest.get("interval_sec", 0))
        self.history_sec = int(self.manifest.get("history_sec", 0))
        if self.interval_sec <= 0 or self.history_sec <= 0:
            raise ValueError("interval_sec and history_sec must be positive")

    def load_split(self, split: str) -> "PreparedVitalDBSplit":
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return PreparedVitalDBSplit(self, split)

    def validate(self) -> dict:
        """Validate shapes, masks, labels, and subject-disjoint splits."""

        split_objects = {split: self.load_split(split) for split in SPLITS}
        subject_sets = {
            split: set(obj.arrays["subject_id"].astype(int).tolist())
            for split, obj in split_objects.items()
        }
        overlaps = {
            f"{left}/{right}": sorted(subject_sets[left] & subject_sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
            if subject_sets[left] & subject_sets[right]
        }
        if overlaps:
            raise ValueError(f"Subjects occur in multiple splits: {overlaps}")

        return {
            "data_dir": str(self.data_dir),
            "parameters": self.parameters,
            "interval_sec": self.interval_sec,
            "history_sec": self.history_sec,
            "upstream_opentslm_commit": UPSTREAM_OPENTSLM_COMMIT,
            "splits": {
                split: {
                    "windows": len(obj),
                    "subjects": len(subject_sets[split]),
                    "series_per_sample": 2 * len(self.parameters),
                }
                for split, obj in split_objects.items()
            },
        }


class PreparedVitalDBSplit:
    def __init__(self, corpus: PreparedVitalDBCorpus, split: str):
        self.corpus = corpus
        self.split = split
        path = corpus.data_dir / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared split: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {missing}")
            self.arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
        self._validate_arrays(path)

    def _validate_arrays(self, path: Path) -> None:
        values = self.arrays["x_values"]
        mask = self.arrays["x_mask"]
        if values.ndim != 3:
            raise ValueError(f"{path}: x_values must have shape [windows, time, parameters]")
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError(f"{path}: x_mask must be boolean and match x_values")
        expected_time = self.corpus.history_sec // self.corpus.interval_sec
        if values.shape[1:] != (expected_time, len(self.corpus.parameters)):
            raise ValueError(
                f"{path}: tensor tail {values.shape[1:]} does not match "
                f"({expected_time}, {len(self.corpus.parameters)})"
            )
        window_count = values.shape[0]
        for name in REQUIRED_ARRAYS[2:]:
            if self.arrays[name].shape != (window_count,):
                raise ValueError(f"{path}: {name} must have shape ({window_count},)")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: x_values contains NaN or infinity")
        if not np.all(values[~mask] == 0):
            raise ValueError(f"{path}: missing x_values must be zero where x_mask is false")
        labels = set(self.arrays["y"].astype(int).tolist())
        if not labels.issubset(self.corpus.labels):
            raise ValueError(f"{path}: unknown label indexes {sorted(labels - self.corpus.labels.keys())}")

    def __len__(self) -> int:
        return int(self.arrays["x_values"].shape[0])

    def normalized_and_mask(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.arrays["x_values"][index].copy(), self.arrays["x_mask"][index].copy()

    def raw_values(self, index: int) -> np.ndarray:
        normalized, mask = self.normalized_and_mask(index)
        raw = normalized * self.corpus.scale + self.corpus.median
        raw[~mask] = np.nan
        return raw.astype(np.float32, copy=False)


class VitalDBHypotensionDataset:
    """A torch-style dataset emitting OpenTSLM's prompt dictionary contract.

    The class intentionally has no torch or opentslm import, so dataset QA can
    run before a GPU environment is provisioned.  OpenTSLM's published collator
    converts the one-dimensional NumPy series to tensors and pads length 10 to
    its patch-size multiple at batch time.
    """

    def __init__(
        self,
        split: str,
        EOS_TOKEN: str,
        data_dir: str | Path = Path("data/lstm/top_10"),
    ):
        self.corpus = PreparedVitalDBCorpus(data_dir)
        self.split = self.corpus.load_split(split)
        self.eos_token = EOS_TOKEN

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict:
        normalized, mask = self.split.normalized_and_mask(index)
        target_key = self.corpus.labels[int(self.split.arrays["y"][index])]
        time_series: list[np.ndarray] = []
        time_series_text: list[str] = []
        for column, parameter in enumerate(self.corpus.parameters):
            unit = PARAMETER_UNITS[parameter]
            time_series.append(normalized[:, column].astype(np.float32, copy=True))
            time_series_text.append(
                f"{parameter} history in normalized {unit}; training median "
                f"{self.corpus.median[column]:g}, IQR scale {self.corpus.scale[column]:g}:"
            )
            time_series.append(mask[:, column].astype(np.float32, copy=True))
            time_series_text.append(
                f"{parameter} observation mask for the preceding channel (1 observed, 0 missing):"
            )

        answer = self._answer(index, target_key)
        if self.eos_token and not answer.endswith(self.eos_token):
            answer += self.eos_token
        return {
            "pre_prompt": self._pre_prompt(),
            "time_series_text": time_series_text,
            "time_series": time_series,
            "post_prompt": POST_PROMPT,
            "answer": answer,
            # Provenance is not consumed by OpenTSLM's model inputs.
            "sample_id": self.sample_id(index),
            "source_split": self.split.split,
        }

    def _pre_prompt(self) -> str:
        sample_count = self.corpus.history_sec // self.corpus.interval_sec
        return f"""You are given a {self.corpus.history_sec}-second intraoperative monitoring window
with {sample_count} source samples per channel at {self.corpus.interval_sec}-second intervals.
Interpret only the supplied signal histories and their observation masks. Signal
channels are normalized with training-split statistics. A zero at a source-missing
position is imputation and its paired mask is 0. OpenTSLM's collator may append
additional right-side zeros to both series solely to reach a patch-size multiple.
Do not infer a causal mechanism or recommend a drug or dose. Predict the tightest
supplied horizon containing the onset of sustained MAP below 65 mmHg, or no onset
within 15 minutes."""

    def sample_id(self, index: int) -> str:
        case_id = int(self.split.arrays["case_id"][index])
        cutoff_sec = int(self.split.arrays["cutoff_sec"][index])
        return f"vitaldb-case-{case_id}-cutoff-{cutoff_sec}"

    def _answer(self, index: int, target_key: str) -> str:
        raw = self.split.raw_values(index)
        map_index = self.corpus.parameters.index("MAP")
        map_values = raw[:, map_index]
        observed = map_values[np.isfinite(map_values)]
        if observed.size:
            delta = float(observed[-1] - observed[0])
            interpret = (
                f"MAP moved from {observed[0]:.1f} to {observed[-1]:.1f} mmHg "
                f"across {observed.size}/{map_values.size} observed input samples "
                f"(net change {delta:+.1f} mmHg)."
            )
        else:
            interpret = "No MAP samples were observed in the input window."
        final_answer = ANSWER_TEXT[target_key]
        return (
            f"INTERPRET: {interpret}\n"
            f"ANTICIPATE: The generated research label is {final_answer}.\n"
            "ACT: Verify signal quality and reassess the current hemodynamic state; "
            "this research label does not prescribe treatment.\n"
            f"Answer: {final_answer}"
        )

    def iter_samples(self, limit: int | None = None) -> Iterator[dict]:
        stop = len(self) if limit is None else min(limit, len(self))
        for index in range(stop):
            yield self[index]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/lstm/top_10"))
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--show-sample", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    corpus = PreparedVitalDBCorpus(args.data_dir)
    report = corpus.validate()
    if args.show_sample is not None:
        dataset = VitalDBHypotensionDataset(args.split, EOS_TOKEN="", data_dir=args.data_dir)
        sample = dataset[args.show_sample]
        report["sample"] = {
            "sample_id": sample["sample_id"],
            "series": len(sample["time_series"]),
            "series_length": [len(values) for values in sample["time_series"]],
            "answer": sample["answer"],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
