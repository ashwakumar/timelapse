"""Real VitalDB onset inputs and portable, verifiable evaluation bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.leakage import audit_prepared_splits
from scripts.opentslm_vitaldb_dataset import (
    ANSWER_TEXT, PARAMETER_UNITS, PreparedVitalDBCorpus,
)

TASK = "vitaldb_onset_v2"
DATA_FILES = ("manifest.json", "train.npz", "validation.npz", "test.npz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    return {"schema_version": 1, "files": {name: sha256(root / name) for name in DATA_FILES}}


def validate_corpus(data_dir: str | Path) -> tuple[PreparedVitalDBCorpus, dict]:
    corpus = PreparedVitalDBCorpus(data_dir)
    if corpus.manifest.get("training_contract") != TASK:
        raise ValueError("Requires real VitalDB onset_v2 data. Run scripts/prepare_vitaldb_training.py; synthetic and legacy exploratory corpora are rejected.")
    if "vitaldb" not in str(corpus.manifest.get("source", "")).lower():
        raise ValueError("Dataset source must explicitly identify VitalDB")
    if corpus.manifest.get("source_is_synthetic") is True:
        raise ValueError("Synthetic data cannot be used for real VitalDB training")
    if corpus.labels != {i: key for i, key in enumerate(ANSWER_TEXT)}:
        raise ValueError("Label order must match the five canonical onset classes")
    audit = audit_prepared_splits(corpus)
    for name in ("train", "validation", "test"):
        split = corpus.load_split(name)
        if len(split) == 0:
            raise ValueError(f"Empty {name} split; prepare more cases")
        labels = split.arrays["y"]
        if not (np.any(labels < 4) and np.any(labels == 4)):
            raise ValueError(f"{name} needs both event and no-event windows; prepare more cases")
    if len(np.unique(corpus.load_split("train").arrays["y"])) != 5:
        raise ValueError("Training must contain all five target classes; prepare more cases")
    return corpus, audit


class VitalDBOnsetDataset:
    """One physical series plus one explicit mask per channel; no double masking.

    This is the SurgicalTelemetryModel contract, distinct from the upstream
    adapter's interleaved value/mask channels. Future-derived labels appear only
    in `answer`, which model.generate never incorporates in its input prompt.
    """

    def __init__(self, data_dir: str | Path, split: str, eos_token: str = ""):
        self.corpus = PreparedVitalDBCorpus(data_dir)
        self.split = self.corpus.load_split(split)
        self.eos_token = eos_token

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index: int) -> dict:
        values, masks = self.split.normalized_and_mask(index)
        key = self.corpus.labels[int(self.split.arrays["y"][index])]
        choices = "; ".join(ANSWER_TEXT.values())
        return {
            "sample_id": f"vitaldb-case-{int(self.split.arrays['case_id'][index])}-cutoff-{int(self.split.arrays['cutoff_sec'][index])}",
            "pre_prompt": (
                f"Predict the first NEW sustained MAP below 65 mmHg episode lasting at least 60 seconds. "
                f"The preceding minute was observed and non-low. Input is {self.corpus.history_sec} seconds "
                f"sampled every {self.corpus.interval_sec} seconds, strictly before the prediction cutoff. "
                "Use only supplied histories and masks. Values are normalized with training-only median/IQR. "
                "Horizon categories refer to onset at or after the cutoff, not current MAP or a numerical MAP forecast."
            ),
            "post_prompt": f"Return exactly one of these five answers, with no explanation: {choices}.",
            "time_series_text": [
                f"{p} ({PARAMETER_UNITS[p]}), training median={self.corpus.median[i]:.6g}, IQR={self.corpus.scale[i]:.6g}:"
                for i, p in enumerate(self.corpus.parameters)
            ],
            "time_series": [values[:, i].copy() for i in range(values.shape[1])],
            "signal_valid_mask": masks.T.copy(),
            "signal_length": values.shape[0],
            "answer": ANSWER_TEXT[key] + self.eos_token,
        }


def collate_samples(samples: list[dict]) -> list[dict]:
    """Model handles ragged lengths and right padding itself."""
    return samples


def verify_bundle(bundle: str | Path) -> PreparedVitalDBCorpus:
    root = Path(bundle)
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("task") != TASK:
        raise ValueError("Bundle is not a real VitalDB onset model")
    files = manifest.get("files", {})
    required = set(DATA_FILES) | {"best.pt", "dataset_fingerprint.json", "run_config.json", "metrics.json", "split_audit.json"}
    if not required.issubset(files):
        raise ValueError(f"Incomplete bundle: missing hashes for {sorted(required - set(files))}")
    for name, expected in files.items():
        if Path(name).name != name or sha256(root / name) != expected:
            raise ValueError(f"Bundle checksum mismatch or unsafe name: {name}")
    if json.loads((root / "dataset_fingerprint.json").read_text()) != fingerprint(root):
        raise ValueError("Bundle data does not match the training fingerprint")
    import torch
    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("task_metadata") != {"task": TASK, "dataset_fingerprint": fingerprint(root)}:
        raise ValueError("Bundle checkpoint task/dataset metadata does not match its data")
    corpus, _ = validate_corpus(root)
    return corpus
