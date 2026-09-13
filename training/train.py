#!/usr/bin/env python3
"""Train OpenTSLM-SP with LoRA on leakage-audited surgical telemetry."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from training.config import ExperimentConfig
from training.model import (
    SurgicalTelemetryModel,
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)

DEFAULT_CONFIG_PATH = Path("training/example_config.json")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, workers, and deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable accelerator requests."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps":
        raise RuntimeError(
            "OpenTSLM-SP training is unsupported on MPS; use CUDA or CPU"
        )
    return device


def resolve_precision(requested: str, device: torch.device) -> str:
    """Select a supported precision while keeping CPU smoke runs in fp32."""
    if requested == "auto":
        if device.type == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if (
        requested == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bf16 was requested but this CUDA device does not support it")
    if requested in {"bf16", "fp16"} and device.type not in {"cuda", "cpu"}:
        raise ValueError(f"{requested} training is not enabled for {device.type}")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 for a CPU smoke run")
    return requested


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _grad_scaler(device: torch.device, precision: str) -> Any:
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def build_optimizer(
    model: SurgicalTelemetryModel, config: ExperimentConfig
) -> torch.optim.AdamW:
    """Build separate AdamW groups for encoder, projector, and LoRA weights."""
    training = config.training
    encoder = [
        parameter
        for parameter in model.backbone.encoder.parameters()
        if parameter.requires_grad
    ]
    projector = [
        parameter
        for parameter in model.backbone.projector.parameters()
        if parameter.requires_grad
    ]
    lora = [
        parameter for parameter in model.llm.parameters() if parameter.requires_grad
    ]
    if not encoder or not projector or not lora:
        raise RuntimeError(
            "Expected trainable encoder, projector, and LoRA parameter groups"
        )
    groups = [
        {"params": encoder, "lr": training.encoder_learning_rate, "name": "encoder"},
        {
            "params": projector,
            "lr": training.projector_learning_rate,
            "name": "projector",
        },
        {"params": lora, "lr": training.lora_learning_rate, "name": "lora"},
    ]
    return torch.optim.AdamW(groups, weight_decay=training.weight_decay)


def _batch_size(batch: Mapping[str, Any] | list[Mapping[str, Any]]) -> int:
    samples = batch["opentslm_batch"] if isinstance(batch, Mapping) else batch
    return len(samples)


@torch.no_grad()
def evaluate(
    model: SurgicalTelemetryModel,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    precision: str,
) -> float:
    """Compute token-weighted validation loss without constructing gradients."""
    model.eval()
    weighted_loss = 0.0
    tokens_seen = 0
    for batch in loader:
        with _autocast(device, precision):
            loss, token_count = model.loss_and_token_count(batch)
        weighted_loss += float(loss) * token_count
        tokens_seen += token_count
    if tokens_seen == 0:
        raise ValueError("Validation loader is empty")
    return weighted_loss / tokens_seen


def train(
    model: SurgicalTelemetryModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    *,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Run accumulated mixed-precision training and save restartable checkpoints."""
    training = config.training
    precision = resolve_precision(training.precision, device)
    optimizer = build_optimizer(model, config)
    updates_per_epoch = math.ceil(
        len(train_loader) / training.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * training.epochs
    if total_updates < 1:
        raise ValueError("Training loader is empty")
    warmup_steps = int(total_updates * training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = _grad_scaler(device, precision)
    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")
    if training.resume_from:
        restored = load_checkpoint(
            training.resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        if checkpoint_metadata is not None and restored.get("task_metadata") != dict(checkpoint_metadata):
            raise ValueError("Resume checkpoint task/dataset metadata does not match this run")
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_validation_loss = float(
            restored.get("metrics", {}).get("best_validation_loss", float("inf"))
        )

    output_dir = training.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    if training.resume_from:
        previous_metrics = Path(training.resume_from).parent / "metrics.json"
        if previous_metrics.exists():
            history = [
                row
                for row in json.loads(previous_metrics.read_text())
                if row["epoch"] < start_epoch
            ]
    accumulation = training.gradient_accumulation_steps
    for epoch in range(start_epoch, training.epochs + 1):
        model.train()
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(training.seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        token_count = 0
        group_start = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_index == group_start:
                current_group_size = min(accumulation, len(train_loader) - group_start)
                group_tokens = 0
            with _autocast(device, precision):
                loss, supervised_tokens = model.loss_and_token_count(batch)
                scaled_loss = loss * supervised_tokens / current_group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss; refusing to save invalid weights"
                )
            scaler.scale(scaled_loss).backward()
            group_tokens += supervised_tokens
            loss_sum += float(loss.detach()) * supervised_tokens
            token_count += supervised_tokens
            end_of_group = batch_index + 1 == group_start + current_group_size
            if end_of_group:
                scaler.unscale_(optimizer)
                # Normalize across all answer tokens, including uneven final batches.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(current_group_size / group_tokens)
                clip_grad_norm_(model.parameters(), training.max_grad_norm)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_was_applied = scaler.get_scale() >= previous_scale
                if step_was_applied:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
                group_start += current_group_size
                if global_step % training.log_every_steps == 0:
                    print(
                        json.dumps(
                            {"epoch": epoch, "step": global_step, "loss": float(loss)}
                        )
                    )

        if token_count == 0:
            raise ValueError("Training loader is empty")
        train_loss = loss_sum / token_count
        validation_loss = evaluate(model, validation_loader, device, precision)
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss")
        improved = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss, validation_loss)
        metrics = {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
        }
        history.append({"epoch": float(epoch), **metrics})
        print(json.dumps({"epoch": epoch, "step": global_step, **metrics}))
        checkpoint_arguments = dict(
            model=model,
            epoch=epoch,
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            training_config=training.to_dict(),
            task_metadata=checkpoint_metadata,
        )
        if training.save_every_epoch:
            save_checkpoint(
                output_dir / f"checkpoint-epoch-{epoch:03d}.pt", **checkpoint_arguments
            )
        save_checkpoint(output_dir / "last.pt", **checkpoint_arguments)
        if improved:
            save_checkpoint(output_dir / "best.pt", **checkpoint_arguments)
        (output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return history


def build_dataloaders(
    config: ExperimentConfig,
    model: SurgicalTelemetryModel,
    device: torch.device,
    records: list[dict[str, Any]] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create identity-disjoint datasets using normalization fitted on train only."""
    from training.data import (
        ChannelNormalizer,
        TelemetryCollator,
        TelemetryDataset,
        load_manifest,
        split_records,
    )
    from training.provenance import dataset_fingerprint, verify_run_artifacts

    training = config.training
    output_dir = training.output_path
    if training.patch_size != model.backbone.patch_size:
        raise ValueError("patch_size must match the pretrained OpenTSLM encoder")
    position_embeddings = getattr(model.backbone.encoder, "pos_embed", None)
    if (
        position_embeddings is not None
        and math.ceil(training.max_signal_length / training.patch_size)
        > position_embeddings.shape[1]
    ):
        raise ValueError(
            "max_signal_length exceeds the pretrained encoder positional capacity"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = load_manifest(
            training.manifest_path,
            require_device_id=training.require_device_id,
            allow_case_only=training.allow_case_only,
        )
    train_records, validation_records, audit = split_records(
        records,
        val_fraction=training.validation_fraction,
        seed=training.seed,
        audit_path=None,
    )
    normalizer = ChannelNormalizer.fit(
        train_records, max_signal_length=training.max_signal_length
    )
    artifacts = {
        "split_audit.json": audit,
        "normalization.json": normalizer.to_dict(),
        "dataset_fingerprint.json": dataset_fingerprint(
            records, training.manifest_path
        ),
        "environment.json": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "numpy",
                    "scikit-learn",
                    "opentslm",
                )
            },
        },
    }
    verify_run_artifacts(output_dir, artifacts, training.resume_from)
    eos_token = model.tokenizer.eos_token or ""
    train_dataset = TelemetryDataset(
        train_records,
        normalizer,
        max_signal_length=training.max_signal_length,
        eos_token=eos_token,
    )
    validation_dataset = TelemetryDataset(
        validation_records,
        normalizer,
        max_signal_length=training.max_signal_length,
        eos_token=eos_token,
    )
    collator = TelemetryCollator(
        patch_size=training.patch_size,
        max_signal_length=training.max_signal_length,
        tokenizer=model.tokenizer,
        max_text_length=training.max_text_length,
    )
    generator = torch.Generator().manual_seed(training.seed)
    loader_arguments = {
        "num_workers": training.num_workers,
        "collate_fn": collator,
        "pin_memory": training.dataloader_pin_memory and device.type == "cuda",
        "persistent_workers": training.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.train_batch_size,
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.eval_batch_size,
        shuffle=False,
        **loader_arguments,
    )
    return train_loader, validation_loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON ExperimentConfig; built-in defaults are used when the file is absent",
    )
    parser.add_argument("--manifest", help="Override training.manifest_path")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--device", help="Override training.device")
    parser.add_argument("--resume-from", help="Override training.resume_from")
    parser.add_argument("--base-model-id", help="Override model.base_model_id")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    from training.data import load_manifest, split_records

    args = _parse_args()
    if args.config.is_file():
        config = ExperimentConfig.from_dict(
            json.loads(args.config.read_text(encoding="utf-8"))
        )
    elif args.config != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    else:
        config = ExperimentConfig()
    for argument, field in (
        (args.manifest, "manifest_path"),
        (args.output_dir, "output_dir"),
        (args.device, "device"),
        (args.resume_from, "resume_from"),
    ):
        if argument is not None:
            setattr(config.training, field, argument)
    if args.base_model_id is not None:
        config.model.base_model_id = args.base_model_id
    config.validate()
    if config.training.resume_from:
        resume = torch.load(
            config.training.resume_from, map_location="cpu", weights_only=False
        )
        saved_model = dict(resume["model_config"])
        provenance = resume.get("provenance")
        if provenance:
            saved_model["model_revision"] = provenance["model_revision"]
            saved_model["base_model_revision"] = provenance["base_model_revision"]
        config.model = config.model.from_dict(saved_model)
        saved_training = resume.get("training_config", {})
        changeable = {
            "output_dir",
            "resume_from",
            "log_every_steps",
            "save_every_epoch",
        }
        critical = set(config.training.to_dict()) - changeable
        mismatches = [
            key
            for key in critical
            if key in saved_training
            and getattr(config.training, key) != saved_training[key]
        ]
        if mismatches:
            raise RuntimeError(
                f"Resume configuration differs for: {', '.join(sorted(mismatches))}"
            )
    elif config.training.output_path.exists() and any(
        config.training.output_path.iterdir()
    ):
        raise FileExistsError(
            "Use an empty output directory for a new run, or supply --resume-from"
        )
    seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    precision = resolve_precision(config.training.precision, device)
    # Reject malformed data or impossible identity splits before any Hub download.
    records = load_manifest(
        config.training.manifest_path,
        require_device_id=config.training.require_device_id,
        allow_case_only=config.training.allow_case_only,
    )
    split_records(
        records,
        val_fraction=config.training.validation_fraction,
        seed=config.training.seed,
    )
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        precision
    ]
    model = load_pretrained_model(config.model, str(device), dtype=dtype)
    if model.provenance is not None:
        config.model.model_revision = model.provenance.model_revision
        config.model.base_model_revision = model.provenance.base_model_revision
    train_loader, validation_loader = build_dataloaders(config, model, device, records)
    config.training.output_path.mkdir(parents=True, exist_ok=True)
    (config.training.output_path / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    train(model, train_loader, validation_loader, config, device)


if __name__ == "__main__":
    main()
