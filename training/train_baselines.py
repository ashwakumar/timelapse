"""Fit Chronos/Ridge once; export the frozen text-only baseline locally."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from training.baseline_features import (
    ACTIVE_SENSORS,
    extract_chronos_representations,
    load_splits,
)


def train_baselines(
    data_path="data/processed/windows.jsonl",
    baseline_dir="models/baselines",
    chronos_model="amazon/chronos-t5-tiny",
    text_model="HuggingFaceTB/SmolLM-135M-Instruct",
    device="cpu",
    batch_size=16,
    force=False,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(baseline_dir)
    manifest = root / "manifest.json"
    if manifest.exists() and not force:
        print(f"Baselines already prepared at {root}. Use --force to retrain.")
        return
    train, _ = load_splits(data_path)
    from chronos import ChronosPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root.mkdir(parents=True, exist_ok=True)
    # A manifest marks a complete export; invalidate it before replacing any weights.
    manifest.unlink(missing_ok=True)
    print("Saving frozen Chronos backbone and fitting its RUL head...", flush=True)
    pipeline = ChronosPipeline.from_pretrained(chronos_model, device_map=device)
    pipeline.model.eval()
    features, targets = extract_chronos_representations(pipeline, train, batch_size, device)
    regressor = Ridge(alpha=10.0).fit(features, targets)
    np.savez(root / "chronos_ridge.npz", coef=regressor.coef_, intercept=regressor.intercept_)
    pipeline.model.model.save_pretrained(root / "chronos")
    del pipeline, features, regressor

    print("Saving frozen text-only model and tokenizer (no fine-tuning)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    if tokenizer is None:
        raise ValueError(f"Could not load tokenizer for {text_model}")
    text = AutoModelForCausalLM.from_pretrained(text_model)
    tokenizer.save_pretrained(root / "text_model")
    text.save_pretrained(root / "text_model")
    metadata = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "data_path": str(data_path),
        "sensors": ACTIVE_SENSORS,
        "train_units": sorted({int(r["unit_number"]) for r in train}),
        "training_windows": len(train),
        "training_mean": float(targets.mean()),
        "chronos_model": chronos_model,
        "text_model": text_model,
        "text_training": "Frozen pretrained model; no task fine-tuning",
    }
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(manifest)
    print(f"Saved baselines to {root}. Benchmark and GUI can now reuse them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/windows.jsonl")
    parser.add_argument("--baseline-dir", default="models/baselines")
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--text-model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing prepared baseline bundle"
    )
    train_baselines(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
