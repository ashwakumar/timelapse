"""Fine-tune real OpenTSLM on prepared VitalDB onset labels, never fixtures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.config import ExperimentConfig
from training.model import load_pretrained_model
from training.provenance import verify_run_artifacts
from training.train import resolve_device, seed_everything, train
from training.vitaldb import TASK, VitalDBOnsetDataset, collate_samples, fingerprint, validate_corpus


def run(config: ExperimentConfig, data_dir: Path, *, balance_classes: bool = True,
        preflight_only: bool = False, model=None) -> dict:
    config.validate()
    corpus, audit = validate_corpus(data_dir)
    if corpus.history_sec // corpus.interval_sec > config.training.max_signal_length:
        raise ValueError("Configured max_signal_length would truncate the prepared history")
    dataset_id = fingerprint(data_dir)
    counts = {s: np.bincount(corpus.load_split(s).arrays["y"], minlength=5).tolist()
              for s in ("train", "validation", "test")}
    summary = {"task": TASK, "data_fingerprint": dataset_id, "class_counts": counts,
               "split_windows": audit["windows_by_split"], "split_patients": audit["subjects_by_split"],
               "parameters": corpus.parameters, "balanced_training_sampler": balance_classes}
    print(json.dumps(summary, indent=2), flush=True)
    if preflight_only:
        return summary
    output = config.training.output_path
    if not config.training.resume_from and output.exists() and any(output.iterdir()):
        raise FileExistsError("New training requires an empty output directory; use --resume-from to resume")
    # Bind resume to data, task and optimization semantics. Locations/device can change.
    critical = config.to_dict()
    for name in ("manifest_path", "output_dir", "resume_from", "device", "precision", "num_workers", "dataloader_pin_memory", "log_every_steps"):
        critical["training"].pop(name, None)
    critical["model"].pop("cache_dir", None)
    identity = {"task": TASK, "config": critical, "balance_classes": balance_classes}
    artifacts = {"dataset_fingerprint.json": dataset_id, "split_audit.json": audit,
                 "training_identity.json": identity}
    if config.training.resume_from:
        verify_run_artifacts(output, artifacts, config.training.resume_from)
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    model = model or load_pretrained_model(config.model, str(device))
    if model.backbone.patch_size != config.training.patch_size:
        raise ValueError("Configured patch size differs from the pretrained encoder")
    # A failed first download must not leave a seemingly occupied run directory.
    if not config.training.resume_from:
        verify_run_artifacts(output, artifacts, None)
    eos = model.tokenizer.eos_token or ""
    development = VitalDBOnsetDataset(data_dir, "train", eos)
    validation = VitalDBOnsetDataset(data_dir, "validation", eos)
    labels = development.split.arrays["y"].astype(int)
    generator = torch.Generator().manual_seed(config.training.seed)
    sampler = None
    if balance_classes:
        class_counts = np.bincount(labels, minlength=5)
        sampler = WeightedRandomSampler(torch.as_tensor(1.0 / class_counts[labels], dtype=torch.double),
                                        len(labels), replacement=True, generator=generator)
    common = {"collate_fn": collate_samples, "num_workers": config.training.num_workers,
              "pin_memory": device.type == "cuda" and config.training.dataloader_pin_memory}
    train_loader = DataLoader(development, batch_size=config.training.train_batch_size,
                              sampler=sampler, shuffle=sampler is None, generator=generator, **common)
    validation_loader = DataLoader(validation, batch_size=config.training.eval_batch_size,
                                   shuffle=False, **common)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    metadata = {"task": TASK, "data_dir": str(data_dir.resolve()), "config": config.to_dict(),
                "balance_classes": balance_classes, "git_commit": revision,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ("torch", "transformers", "peft", "numpy", "opentslm")}}
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    history = train(model, train_loader, validation_loader, config, device,
                    checkpoint_metadata={"task": TASK, "dataset_fingerprint": dataset_id})
    # Training never generates or scores predictions on the reserved test split.
    return {**summary, "epochs_completed": len(history), "best_checkpoint": str(output / "best.pt")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/vitaldb_config.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/vitaldb/onset_v2"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--unbalanced", action="store_true", help="Disable train-only inverse-frequency sampling")
    parser.add_argument("--preflight-only", action="store_true", help="Audit real data without downloading/loading a model")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_dict(json.loads(args.config.read_text()))
    for arg, name in ((args.output_dir, "output_dir"), (args.epochs, "epochs"),
                      (args.device, "device"), (args.resume_from, "resume_from")):
        if arg is not None:
            setattr(config.training, name, str(arg) if isinstance(arg, Path) else arg)
    config.training.manifest_path = str(args.data_dir / "manifest.json")
    run(config, args.data_dir, balance_classes=not args.unbalanced, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
