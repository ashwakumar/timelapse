"""Load the 3-epoch OpenTSLM checkpoint and generate on held-out fixture samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/run-opentslm-3ep"
MANIFEST = ROOT / "data/surgical_telemetry/manifest.jsonl"

_model = None
_dataset = None
_records_by_id: dict[str, dict[str, Any]] = {}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_runtime() -> None:
    global _model, _dataset, _records_by_id
    if _model is not None:
        return
    from training.data import ChannelNormalizer, TelemetryDataset, load_manifest
    from training.model import load_for_inference

    audit = json.loads((RUN / "split_audit.json").read_text(encoding="utf-8"))
    val_ids = set(audit["val_sample_ids"])
    records = load_manifest(MANIFEST, require_device_id=True, allow_case_only=False)
    held_out = [record for record in records if record["sample_id"] in val_ids]
    if not held_out:
        raise RuntimeError("No held-out samples found in split_audit.json")
    normalizer = ChannelNormalizer.load(RUN / "normalization.json")
    _model = load_for_inference(RUN / "best.pt", _device())
    eos = ""
    if hasattr(_model, "tokenizer") and getattr(_model.tokenizer, "eos_token", None):
        eos = _model.tokenizer.eos_token or ""
    _dataset = TelemetryDataset(
        held_out, normalizer, max_signal_length=300, eos_token=eos
    )
    _records_by_id = {record["sample_id"]: record for record in held_out}


def list_samples() -> list[dict[str, Any]]:
    load_runtime()
    samples = []
    for index, record in enumerate(_dataset.records):
        samples.append(
            {
                "index": index,
                "sample_id": record["sample_id"],
                "patient_id": record["patient_id"],
                "device_id": record["device_id"],
                "prompt": record["prompt"],
                "gold_target": record["target"],
            }
        )
    return samples


def sample_input(index: int) -> dict[str, Any]:
    load_runtime()
    record = _dataset.records[index]
    signal = np.load(record["signal_path"], allow_pickle=False)
    if signal.shape[1] > 300:
        signal = signal[:, -300:]
    channels = ["ABP", "HR", "SpO2", "EtCO2"]
    series = {}
    for i, name in enumerate(channels):
        values = signal[i].astype(np.float32)
        # downsample for the browser chart
        step = max(1, values.size // 120)
        series[name] = [
            None if not np.isfinite(v) else float(v) for v in values[::step].tolist()
        ]
    return {
        "sample_id": record["sample_id"],
        "patient_id": record["patient_id"],
        "case_id": record["case_id"],
        "device_id": record["device_id"],
        "prompt": record["prompt"],
        "window_start": record["window_start"],
        "window_end": record["window_end"],
        "query_time": record["query_time"],
        "channels": channels,
        "series": series,
        "gold_target": record["target"],
        "held_out": True,
    }


def generate(index: int, max_new_tokens: int = 160) -> dict[str, Any]:
    load_runtime()
    item = _dataset[index]
    texts = _model.generate([item], max_new_tokens=max_new_tokens, do_sample=False)
    return {
        "sample_id": item["sample_id"],
        "model_output": texts[0].strip(),
        "gold_target": _dataset.records[index]["target"],
        "checkpoint": str(RUN / "best.pt"),
        "device": _device(),
    }
