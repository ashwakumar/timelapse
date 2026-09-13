"""Load an OpenTSLM best.pt checkpoint and generate on held-out fixture samples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/surgical_telemetry/manifest.jsonl"

_runtimes: dict[str, dict[str, Any]] = {}


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_sections(text: str) -> dict[str, str]:
    parts = {"Observation": "", "Rationale": "", "Recommendation": ""}
    matches = list(
        re.finditer(
            r"(Observation|Rationale|Recommendation)\s*:\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    for i, match in enumerate(matches):
        key = match.group(1).title()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[key] = text[match.end() : end].strip()
    if not any(parts.values()) and text.strip():
        parts["Observation"] = text.strip()
    return parts


def load_runtime(run_id: str | None = None) -> dict[str, Any]:
    from demo.runs import resolve_run
    from training.data import ChannelNormalizer, TelemetryDataset, load_manifest
    from training.model import load_for_inference

    run = resolve_run(run_id, ROOT)
    cache_key = str(run["id"])
    if cache_key in _runtimes:
        return _runtimes[cache_key]

    folder = ROOT / run["folder"]
    checkpoint = folder / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing trained checkpoint: {checkpoint}")

    audit = json.loads((folder / "split_audit.json").read_text(encoding="utf-8"))
    val_ids = set(audit["val_sample_ids"])
    records = load_manifest(MANIFEST, require_device_id=True, allow_case_only=False)
    held_out = [record for record in records if record["sample_id"] in val_ids]
    if not held_out:
        raise RuntimeError("No held-out samples found in split_audit.json")
    normalizer = ChannelNormalizer.load(folder / "normalization.json")
    model = load_for_inference(checkpoint, _device())
    eos = ""
    if hasattr(model, "tokenizer") and getattr(model.tokenizer, "eos_token", None):
        eos = model.tokenizer.eos_token or ""
    dataset = TelemetryDataset(
        held_out, normalizer, max_signal_length=300, eos_token=eos
    )
    runtime = {
        "run": run,
        "model": model,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
    }
    _runtimes[cache_key] = runtime
    return runtime


def list_samples(run_id: str | None = None) -> list[dict[str, Any]]:
    runtime = load_runtime(run_id)
    samples = []
    for index, record in enumerate(runtime["dataset"].records):
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


def sample_input(index: int, run_id: str | None = None) -> dict[str, Any]:
    runtime = load_runtime(run_id)
    record = runtime["dataset"].records[index]
    signal = np.load(record["signal_path"], allow_pickle=False)
    if signal.shape[1] > 300:
        signal = signal[:, -300:]
    channels = ["ABP", "HR", "SpO2", "EtCO2"]
    series = {}
    for i, name in enumerate(channels):
        values = signal[i].astype(np.float32)
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
        "run_id": runtime["run"]["id"],
    }


def generate(
    index: int, run_id: str | None = None, max_new_tokens: int = 160
) -> dict[str, Any]:
    runtime = load_runtime(run_id)
    item = runtime["dataset"][index]
    texts = runtime["model"].generate(
        [item], max_new_tokens=max_new_tokens, do_sample=False
    )
    output = texts[0].strip()
    return {
        "sample_id": item["sample_id"],
        "model_output": output,
        "sections": _parse_sections(output),
        "gold_target": runtime["dataset"].records[index]["target"],
        "checkpoint": runtime["checkpoint"],
        "run_id": runtime["run"]["id"],
        "title": runtime["run"]["title"],
        "subtitle": runtime["run"]["subtitle"],
        "passes": runtime["run"]["passes"],
        "device": _device(),
    }
