"""Configuration objects for reproducible OpenTSLM fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """OpenTSLM-SP and LoRA configuration.

    Mutable Hub revision names are resolved to immutable commit SHAs before any
    weights are loaded; those SHAs are written into every training checkpoint.
    """

    model_id: str = "OpenTSLM/llama-3.2-1b-tsqa-sp"
    model_revision: str = "main"
    base_model_revision: str = "main"
    cache_dir: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    gradient_checkpointing: bool = True
    max_context_tokens: int | None = None
    truncate_prompt_to_fit: bool = False

    def validate(self) -> None:
        """Reject configurations that cannot define a LoRA adapter."""
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("lora_rank and lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules cannot be empty")
        if self.max_context_tokens is not None and self.max_context_tokens < 2:
            raise ValueError("max_context_tokens must be at least 2")

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        """Reconstruct a config, accepting JSON lists for tuple fields."""
        values = dict(values)
        if "lora_target_modules" in values:
            values["lora_target_modules"] = tuple(values["lora_target_modules"])
        return cls(**values)


@dataclass(slots=True)
class TrainingConfig:
    """Data, optimization, precision, and checkpoint parameters."""

    manifest_path: str = "data/surgical_telemetry/manifest.jsonl"
    output_dir: str = "artifacts/opentslm_surgical_telemetry"
    seed: int = 42
    validation_fraction: float = 0.20
    require_device_id: bool = True
    allow_case_only: bool = False
    max_signal_length: int = 1200
    max_text_length: int = 2048
    patch_size: int = 4
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 0
    epochs: int = 3
    gradient_accumulation_steps: int = 8
    encoder_learning_rate: float = 2e-4
    projector_learning_rate: float = 1e-4
    lora_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    precision: str = "auto"
    device: str = "auto"
    resume_from: str | None = None
    save_every_epoch: bool = True
    log_every_steps: int = 10
    dataloader_pin_memory: bool = True

    def validate(self) -> None:
        """Validate values before model allocation or data loading."""
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        positive = {
            "max_signal_length": self.max_signal_length,
            "max_text_length": self.max_text_length,
            "patch_size": self.patch_size,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_signal_length % self.patch_size:
            raise ValueError("max_signal_length must be divisible by patch_size")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be auto, bf16, fp16, or fp32")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")
        for name, value in {
            "encoder_learning_rate": self.encoder_learning_rate,
            "projector_learning_rate": self.projector_learning_rate,
            "lora_learning_rate": self.lora_learning_rate,
            "max_grad_norm": self.max_grad_norm,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")

    @property
    def output_path(self) -> Path:
        """Return the configured output directory."""
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation."""
        return asdict(self)


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level serializable experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Validate all nested configuration."""
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a nested checkpoint-safe representation."""
        return {"model": self.model.to_dict(), "training": self.training.to_dict()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        """Reconstruct an experiment configuration from JSON/checkpoint data."""
        return cls(
            model=ModelConfig.from_dict(values.get("model", {})),
            training=TrainingConfig(**values.get("training", {})),
        )
