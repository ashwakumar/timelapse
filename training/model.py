"""OpenTSLM-SP loading, multimodal loss assembly, and adapter checkpoints."""

from __future__ import annotations

import importlib.metadata
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from training.config import ModelConfig


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Immutable weight and implementation identifiers saved with a run."""

    model_id: str
    model_revision: str
    base_model_id: str
    base_model_revision: str
    opentslm_version: str
    opentslm_source_revision: str = "2968f4b891baab4307f7e9d0043e87677b593a30"


def _base_model_id(model_id: str) -> str:
    """Map SP repository names to their published base language models."""
    name = model_id.rsplit("/", 1)[-1].lower()
    if name.startswith("gemma-3-1b") and "-pt" in name:
        return "google/gemma-3-1b-pt"
    if name.startswith("gemma-3-270m") and "-pt" in name:
        return "google/gemma-3-270m-pt"
    mappings = (
        ("llama-3.2-3b", "meta-llama/Llama-3.2-3B"),
        ("llama-3.2-1b", "meta-llama/Llama-3.2-1B"),
        ("gemma-3-1b", "google/gemma-3-1b-pt"),
        ("gemma-3-270m", "google/gemma-3-270m"),
    )
    for prefix, base_id in mappings:
        if name.startswith(prefix):
            return base_id
    raise ValueError(f"Cannot infer the base LLM from OpenTSLM repository {model_id!r}")


def _resolved_hub_revision(repo_id: str, revision: str) -> str:
    """Resolve a Hub branch/tag to the immutable commit SHA it currently names."""
    from huggingface_hub import HfApi

    sha = HfApi().model_info(repo_id=repo_id, revision=revision).sha
    if not sha:
        raise RuntimeError(
            f"Hugging Face did not return a commit SHA for {repo_id}@{revision}"
        )
    return sha


def apply_lora(backbone: nn.Module, config: ModelConfig) -> None:
    """Freeze the LLM and attach a PEFT causal-LM LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model

    for parameter in backbone.llm.parameters():
        parameter.requires_grad_(False)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    backbone.llm = get_peft_model(backbone.llm, peft_config)
    backbone.lora_enabled = True
    backbone.encoder.requires_grad_(True)
    backbone.projector.requires_grad_(True)


def prepare_backbone_for_training(
    backbone: nn.Module, config: ModelConfig, frozen_llm_dtype: torch.dtype
) -> None:
    """Set frozen LLM precision while retaining FP32 trainable weights for scaling."""
    backbone.llm.to(dtype=frozen_llm_dtype)
    backbone.encoder.float()
    backbone.projector.float()
    apply_lora(backbone, config)
    for parameter in backbone.llm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


class SurgicalTelemetryModel(nn.Module):
    """Corrected training wrapper around the native OpenTSLM-SP components.

    OpenTSLM-SP interleaves text embeddings and encoded time-series patches.
    This wrapper retains that architecture while excluding padded patches and
    masking every prompt/padding position from the language-model objective.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: ModelConfig,
        provenance: ModelProvenance | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.backbone = backbone
        self.config = config
        self.provenance = provenance
        self._warned_prompt_truncation = False

    @property
    def llm(self) -> nn.Module:
        """Return the PEFT-wrapped causal language model."""
        return self.backbone.llm

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer used by the base LLM."""
        tokenizer = getattr(self.backbone, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.backbone.text_tokenizer
        return tokenizer

    @property
    def device(self) -> torch.device:
        """Return the device holding the language model."""
        return next(self.llm.parameters()).device

    def _context_limit(self) -> int:
        configured = self.config.max_context_tokens
        native = getattr(self.llm.config, "max_position_embeddings", None)
        limits = [int(limit) for limit in (configured, native) if limit is not None]
        if not limits:
            raise ValueError(
                "Set max_context_tokens: the LLM config has no context limit"
            )
        return min(limits)

    def _token_ids(self, text: str, *, add_special_tokens: bool) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )
        ids = (
            encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        )
        return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _answer_ids(self, answer: str) -> torch.Tensor:
        ids = self._token_ids(" " + answer.strip(), add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (ids.numel() == 0 or ids[-1].item() != eos_id):
            ids = torch.cat([ids, ids.new_tensor([eos_id])])
        if ids.numel() == 0:
            raise ValueError("A training answer must contain at least one token")
        return ids

    def _encode_time_series(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Encode values and observation masks without cross-length padding attention."""
        groups: dict[int, list[tuple[torch.Tensor, int, int, int]]] = {}
        encoded: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None] * len(sample["time_series"]) for sample in samples
        ]
        patch_size = int(getattr(self.backbone, "patch_size", 4))

        for sample_index, sample in enumerate(samples):
            values = sample["time_series"]
            masks = sample.get("signal_valid_mask")
            history_length = int(sample.get("signal_length", len(values[0])))
            for channel_index, raw_series in enumerate(values):
                series = torch.as_tensor(raw_series, dtype=torch.float32)
                if series.ndim != 1:
                    raise ValueError(
                        "Each OpenTSLM time_series channel must be one-dimensional"
                    )
                if not 0 < history_length <= len(series):
                    raise ValueError(
                        "signal_length must identify the non-padding source history"
                    )
                if masks is None:
                    valid = torch.isfinite(series[:history_length])
                else:
                    valid = torch.as_tensor(masks[channel_index], dtype=torch.bool)
                    if valid.shape != series.shape:
                        raise ValueError(
                            "signal_valid_mask must match every time-series channel"
                        )
                    valid = valid[:history_length] & torch.isfinite(
                        series[:history_length]
                    )
                clean = torch.where(valid, series[:history_length], 0.0)
                observation_mask = valid.to(torch.float32)
                padded_length = math.ceil(history_length / patch_size) * patch_size
                groups.setdefault(padded_length, []).extend(
                    (
                        (clean, sample_index, channel_index, 0),
                        (observation_mask, sample_index, channel_index, 1),
                    )
                )

        embedding_dtype = self.llm.get_input_embeddings().weight.dtype
        encoder_dtype = next(self.backbone.encoder.parameters()).dtype
        for padded_length, group in groups.items():
            tensor = torch.stack(
                [
                    torch.nn.functional.pad(series, (0, padded_length - len(series)))
                    for series, *_ in group
                ]
            ).to(self.device, dtype=encoder_dtype, non_blocking=True)
            projected = self.backbone.projector(self.backbone.encoder(tensor)).to(
                dtype=embedding_dtype
            )
            for row, (_, sample_index, channel_index, kind) in enumerate(group):
                pair = encoded[sample_index][channel_index]
                if pair is None:
                    pair = (projected[row], projected[row])
                mutable = list(pair)
                mutable[kind] = projected[row]
                encoded[sample_index][channel_index] = (mutable[0], mutable[1])
        if any(pair is None for sample in encoded for pair in sample):
            raise RuntimeError("Failed to encode a telemetry value/mask pair")
        return [[pair for pair in sample if pair is not None] for sample in encoded]

    def _prompt_embeddings(
        self,
        sample: Mapping[str, Any],
        series_embeddings: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        texts = sample["time_series_text"]
        values = sample["time_series"]
        if len(texts) != len(values):
            raise ValueError(
                "time_series_text and time_series must contain the same channels"
            )
        units: list[torch.Tensor] = []
        embedding = self.llm.get_input_embeddings()
        pre_ids = self._token_ids(str(sample["pre_prompt"]), add_special_tokens=True)
        if pre_ids.numel():
            units.append(embedding(pre_ids))
        history_length = int(sample.get("signal_length", len(values[0])))
        for description, (value_embeddings, mask_embeddings) in zip(
            texts, series_embeddings
        ):
            text_ids = self._token_ids(" " + str(description), add_special_tokens=False)
            mask_ids = self._token_ids(
                f" Observation mask for this channel (1=observed, 0=missing; history length={history_length}):",
                add_special_tokens=False,
            )
            chunks = []
            if text_ids.numel():
                chunks.append(embedding(text_ids))
            chunks.extend((value_embeddings, embedding(mask_ids), mask_embeddings))
            units.append(torch.cat(chunks, dim=0))
        post_ids = self._token_ids(
            " " + str(sample.get("post_prompt", "")), add_special_tokens=False
        )
        if post_ids.numel():
            units.append(embedding(post_ids))
        if not units:
            raise ValueError("A prompt must contain text or a valid time series")
        return units

    def prepare_batch(
        self,
        batch: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        include_answers: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Create right-padded LLM inputs and answer-only causal-LM labels."""
        samples = batch["opentslm_batch"] if isinstance(batch, Mapping) else batch
        if not samples:
            raise ValueError("Cannot prepare an empty batch")
        encoded_series = self._encode_time_series(samples)
        limit = self._context_limit()
        sequences: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        prompt_counts: list[int] = []
        answer_counts: list[int] = []

        for sample, signal_chunks in zip(samples, encoded_series):
            prompt_units = self._prompt_embeddings(sample, signal_chunks)
            answer_ids = (
                self._answer_ids(str(sample["answer"])) if include_answers else None
            )
            reserve = int(answer_ids.numel()) if answer_ids is not None else 0
            if reserve >= limit:
                sample_id = sample.get("sample_id", "<unknown>")
                raise ValueError(
                    f"Target for sample {sample_id} has {reserve} tokens, exceeding the {limit}-token context; "
                    "increase max_context_tokens or shorten the target explicitly"
                )
            prompt_budget = limit - reserve
            prompt_length = sum(unit.size(0) for unit in prompt_units)
            if prompt_length > prompt_budget:
                if not self.config.truncate_prompt_to_fit:
                    raise ValueError(
                        f"Prompt plus target needs {prompt_length + reserve} tokens, context limit is {limit}"
                    )
                while (
                    len(prompt_units) > 1
                    and sum(unit.size(0) for unit in prompt_units) > prompt_budget
                ):
                    prompt_units.pop(0)
                if sum(unit.size(0) for unit in prompt_units) > prompt_budget:
                    raise ValueError(
                        "No complete prompt unit fits after reserving the full target"
                    )
                if not self._warned_prompt_truncation:
                    warnings.warn(
                        "Oldest complete prompt units were removed; target tokens and retained channel units are intact.",
                        stacklevel=2,
                    )
                    self._warned_prompt_truncation = True
            prompt = torch.cat(prompt_units, dim=0)
            prompt_counts.append(prompt.size(0))
            if answer_ids is None:
                sequences.append(prompt)
                answer_counts.append(0)
                continue
            answer_embeddings = self.llm.get_input_embeddings()(answer_ids)
            sequences.append(torch.cat([prompt, answer_embeddings], dim=0))
            sample_labels = answer_ids.new_full((prompt.size(0) + reserve,), -100)
            sample_labels[prompt.size(0) :] = answer_ids
            labels.append(sample_labels)
            answer_counts.append(reserve)

        inputs_embeds = pad_sequence(sequences, batch_first=True)
        attention_mask = pad_sequence(
            [
                torch.ones(len(sequence), dtype=torch.long, device=self.device)
                for sequence in sequences
            ],
            batch_first=True,
        )
        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prompt_token_counts": torch.tensor(prompt_counts, device=self.device),
            "answer_token_counts": torch.tensor(answer_counts, device=self.device),
        }
        if include_answers:
            result["labels"] = pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
        return result

    def forward(self, batch: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Any:
        """Run an answer-supervised multimodal forward pass."""
        prepared = self.prepare_batch(batch, include_answers=True)
        return self.llm(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            labels=prepared["labels"],
            return_dict=True,
        )

    def loss_and_token_count(
        self, batch: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> tuple[torch.Tensor, int]:
        """Return answer-only loss and its exact number of supervised tokens."""
        prepared = self.prepare_batch(batch, include_answers=True)
        output = self.llm(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            labels=prepared["labels"],
            return_dict=True,
        )
        return output.loss, int((prepared["labels"] != -100).sum())

    def compute_loss(
        self, batch: Mapping[str, Any] | Sequence[Mapping[str, Any]]
    ) -> torch.Tensor:
        """Return mean causal-LM loss over answer tokens only."""
        return self.loss_and_token_count(batch)[0]

    @torch.inference_mode()
    def generate(
        self,
        batch: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = 256,
        **generation_kwargs: Any,
    ) -> list[str]:
        """Generate one rationale per sample without batch padding ambiguity."""
        samples = batch["opentslm_batch"] if isinstance(batch, Mapping) else batch
        outputs: list[str] = []
        for sample in samples:
            prepared = self.prepare_batch([sample], include_answers=False)
            prompt_length = int(prepared["attention_mask"].sum())
            if prompt_length + max_new_tokens > self._context_limit():
                raise ValueError(
                    "prompt length plus max_new_tokens exceeds the model context"
                )
            generated = self.llm.generate(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs,
            )
            new_ids = generated[0, -max_new_tokens:]
            outputs.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return outputs


def load_pretrained_model(
    config: ModelConfig,
    device: str,
    *,
    dtype: torch.dtype | None = None,
) -> SurgicalTelemetryModel:
    """Load a commit-pinned OpenTSLM-SP checkpoint and attach LoRA."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    config.validate()
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("Pretrained OpenTSLM training supports CPU and CUDA")
    if not config.model_id.endswith("-sp"):
        raise ValueError(
            "This training path requires an OpenTSLM sequence-projection (-sp) checkpoint"
        )
    model_sha = _resolved_hub_revision(config.model_id, config.model_revision)
    base_id = config.base_model_id or _base_model_id(config.model_id)
    base_sha = _resolved_hub_revision(base_id, config.base_model_revision)
    checkpoint_path = hf_hub_download(
        repo_id=config.model_id,
        filename="model_checkpoint.pt",
        revision=model_sha,
        cache_dir=config.cache_dir,
    )
    base_path = snapshot_download(
        repo_id=base_id,
        revision=base_sha,
        cache_dir=config.cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"],
    )
    backbone = OpenTSLMSP(llm_id=base_path, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    projector_weight = checkpoint["projector_state"].get("projector.1.weight")
    if (
        projector_weight is not None
        and projector_weight.shape[0] != backbone.llm.config.hidden_size
    ):
        raise RuntimeError(
            f"{config.model_id} projects time series into {projector_weight.shape[0]} "
            f"dimensions, but base model {base_id} has hidden size "
            f"{backbone.llm.config.hidden_size}; supply a base_model_id whose "
            "architecture matches the checkpoint"
        )
    backbone.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    backbone.projector.load_state_dict(checkpoint["projector_state"], strict=True)
    if dtype is None:
        dtype = (
            torch.float32
            if torch.device(device).type == "cpu"
            else torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    prepare_backbone_for_training(backbone, config, dtype)
    if checkpoint.get("lora_enabled"):
        saved_lora = checkpoint.get("lora_state")
        if not saved_lora:
            raise RuntimeError(
                "OpenTSLM checkpoint declares LoRA but contains no adapter weights"
            )
        named = dict(backbone.llm.named_parameters())
        missing = sorted(key for key in saved_lora if key not in named)
        if missing:
            raise RuntimeError(
                f"OpenTSLM adapter keys do not match configured LoRA: {missing[:3]}"
            )
        expected = {key for key, parameter in named.items() if parameter.requires_grad}
        if set(saved_lora) != expected:
            raise RuntimeError(
                "OpenTSLM checkpoint does not contain every configured LoRA weight"
            )
        for key, value in saved_lora.items():
            if named[key].shape != value.shape:
                raise RuntimeError(f"OpenTSLM adapter shape mismatch for {key}")
            named[key].data.copy_(
                value.to(device=named[key].device, dtype=named[key].dtype)
            )
    if config.gradient_checkpointing:
        backbone.llm.gradient_checkpointing_enable()
        backbone.llm.config.use_cache = False
    backbone.encoder.requires_grad_(True)
    backbone.projector.requires_grad_(True)
    try:
        version = importlib.metadata.version("opentslm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    provenance = ModelProvenance(
        model_id=config.model_id,
        model_revision=model_sha,
        base_model_id=base_id,
        base_model_revision=base_sha,
        opentslm_version=version,
    )
    return SurgicalTelemetryModel(backbone, config, provenance)


def trainable_state_dict(model: SurgicalTelemetryModel) -> dict[str, Any]:
    """Return encoder/projector weights and only the PEFT adapter from the LLM."""
    from peft import get_peft_model_state_dict

    return {
        "encoder": model.backbone.encoder.state_dict(),
        "projector": model.backbone.projector.state_dict(),
        "lora": get_peft_model_state_dict(model.llm),
    }


def load_trainable_state_dict(
    model: SurgicalTelemetryModel, state: Mapping[str, Any]
) -> None:
    """Restore all trainable multimodal and adapter weights strictly."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    model.backbone.encoder.load_state_dict(state["encoder"], strict=True)
    model.backbone.projector.load_state_dict(state["projector"], strict=True)
    expected = get_peft_model_state_dict(model.llm)
    saved = state["lora"]
    if set(expected) != set(saved):
        missing = sorted(set(expected) - set(saved))
        extra = sorted(set(saved) - set(expected))
        raise RuntimeError(
            f"LoRA checkpoint key mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for key in expected:
        if expected[key].shape != saved[key].shape:
            raise RuntimeError(f"LoRA checkpoint shape mismatch for {key}")
    result = set_peft_model_state_dict(model.llm, saved)
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys in checkpoint: {unexpected}")


def save_checkpoint(
    path: str | Path,
    model: SurgicalTelemetryModel,
    *,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    metrics: Mapping[str, float] | None = None,
    training_config: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save adapter, encoder/projector, optimizer, and RNG state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "provenance": asdict(model.provenance) if model.provenance else None,
        "trainable_state": trainable_state_dict(model),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": dict(metrics or {}),
        "training_config": dict(training_config or {}),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: SurgicalTelemetryModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint and return its epoch, step, metrics, and metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint format")
    saved_config = dict(checkpoint["model_config"])
    saved_provenance = checkpoint.get("provenance")
    if saved_provenance:
        saved_config["model_revision"] = saved_provenance["model_revision"]
        saved_config["base_model_revision"] = saved_provenance["base_model_revision"]
    if ModelConfig.from_dict(saved_config).to_dict() != model.config.to_dict():
        raise RuntimeError(
            "Checkpoint model configuration does not match the loaded backbone"
        )
    if saved_provenance and model.provenance:
        actual = asdict(model.provenance)
        if any(actual.get(key) != value for key, value in saved_provenance.items()):
            raise RuntimeError(
                "Checkpoint provenance does not match the loaded immutable revisions"
            )
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint


def load_for_inference(path: str | Path, device: str) -> SurgicalTelemetryModel:
    """Rebuild the exact pinned backbone and restore fine-tuned components."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["model_config"])
    provenance = checkpoint.get("provenance")
    if provenance:
        config_values["model_revision"] = provenance["model_revision"]
        config_values["base_model_revision"] = provenance["base_model_revision"]
    model = load_pretrained_model(ModelConfig.from_dict(config_values), device)
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    model.eval()
    return model
