"""Verify one epoch, checkpoint reload, and generation on CPU or a real CUDA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from training.config import ExperimentConfig, ModelConfig, TrainingConfig
from training.make_example import make_example
from training.model import (
    SurgicalTelemetryModel,
    apply_lora,
    load_checkpoint,
    load_pretrained_model,
)
from training.train import (
    build_dataloaders,
    evaluate,
    resolve_device,
    resolve_precision,
    seed_everything,
    train,
)


def build_smoke_model(
    config: ModelConfig, seed: int, encoder_checkpoint: Path | None
) -> SurgicalTelemetryModel:
    """Use the native OpenTSLM encoder/projector and a small random HF Llama.

    A fixed byte tokenizer represents every input character without fitting on
    validation text. The optional checkpoint initializes only the real encoder;
    the smaller language model and projector are explicitly random.
    """
    from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
    from opentslm.model.projector.MLPProjector import MLPProjector
    from opentslm.model_config import ENCODER_OUTPUT_DIM
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    seed_everything(seed)
    special = ["<unk>", "<bos>", "<eos>", "<pad>"]
    vocabulary = {
        token: index
        for index, token in enumerate(
            special + sorted(pre_tokenizers.ByteLevel.alphabet())
        )
    }
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    backbone = nn.Module()
    backbone.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=special[0],
        bos_token=special[1],
        eos_token=special[2],
        pad_token=special[3],
    )
    backbone.llm = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
    )
    backbone.encoder = TransformerCNNEncoder()
    if encoder_checkpoint:
        state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        backbone.encoder.load_state_dict(state["encoder_state"], strict=True)
    backbone.projector = MLPProjector(ENCODER_OUTPUT_DIM, 64, device="cpu")
    backbone.patch_size = 4
    apply_lora(backbone, config)
    backbone.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backbone.llm.config.use_cache = False
    return SurgicalTelemetryModel(backbone, config)


def parameter_digest(model: nn.Module, *, trainable: bool) -> str:
    """Compare parameter content without retaining another full model in memory."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(
                parameter.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


def run_smoke(
    output: Path, encoder_checkpoint: Path | None = None, *, pretrained: bool = False
) -> dict:
    """Train exactly one epoch, reload all learned weights, and test generation."""
    device = resolve_device("cuda" if pretrained else "cpu")
    precision = resolve_precision("auto", device)
    if pretrained and encoder_checkpoint:
        raise ValueError("--pretrained loads its own encoder checkpoint")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    manifest = make_example(output / "synthetic_data", patients=5 if pretrained else 10)
    config = ExperimentConfig(
        model=ModelConfig(
            model_id="offline-smoke-random-llama",
            model_revision="seed-42",
            base_model_revision="seed-42",
            lora_rank=8,
            lora_alpha=16,
            max_context_tokens=4096,
            gradient_checkpointing=True,
        ),
        training=TrainingConfig(
            manifest_path=str(manifest),
            output_dir=str(output / "run"),
            epochs=1,
            max_signal_length=300,
            max_text_length=4096,
            train_batch_size=1 if pretrained else 3,
            eval_batch_size=1 if pretrained else 2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.5,
            device=str(device),
            precision=precision,
            log_every_steps=1,
        ),
    )
    if pretrained:
        config.model.model_id = "OpenTSLM/llama-3.2-1b-tsqa-sp"
        config.model.model_revision = "1904441f0d87c458d6e9376de04ada5f4c9ab5b7"
        config.model.base_model_revision = "main"
    config.validate()

    def initialize():
        seed_everything(config.training.seed)
        if pretrained:
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            return load_pretrained_model(config.model, str(device), dtype=dtype)
        return build_smoke_model(config.model, config.training.seed, encoder_checkpoint)

    model = initialize()
    if model.provenance:
        config.model.model_revision = model.provenance.model_revision
        config.model.base_model_revision = model.provenance.base_model_revision
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    loaders = build_dataloaders(config, model, device)
    initial_loss = evaluate(model, loaders[1], device, precision)
    frozen_before = parameter_digest(model, trainable=False)
    before = {
        name: parameter_digest(module, trainable=True)
        for name, module in (
            ("encoder", model.backbone.encoder),
            ("projector", model.backbone.projector),
            ("lora", model.llm),
        )
    }
    print(
        json.dumps({"phase": "start", "initial_validation_loss": initial_loss}),
        flush=True,
    )
    metrics = train(model, *loaders, config, device)
    updated = {
        name: parameter_digest(module, trainable=True) != before[name]
        for name, module in (
            ("encoder", model.backbone.encoder),
            ("projector", model.backbone.projector),
            ("lora", model.llm),
        )
    }
    assert all(updated.values()), f"Some trainable components did not update: {updated}"
    assert parameter_digest(model, trainable=False) == frozen_before, (
        "Frozen backbone changed"
    )
    trained_digest = parameter_digest(model, trainable=True)
    del model
    if pretrained:
        torch.cuda.empty_cache()
    restored = initialize()
    checkpoint = load_checkpoint(output / "run" / "last.pt", restored)
    restored_loss = evaluate(restored, loaders[1], device, precision)
    assert checkpoint["epoch"] == 1 and checkpoint["global_step"] == 2
    assert parameter_digest(restored, trainable=True) == trained_digest
    assert math.isclose(
        restored_loss, metrics[-1]["validation_loss"], rel_tol=1e-6, abs_tol=1e-6
    ), "Checkpoint reload changed validation loss"
    predictions = restored.generate(
        next(iter(loaders[1])), max_new_tokens=8, do_sample=False
    )
    assert len(predictions) == config.training.eval_batch_size
    report = {
        "status": "passed",
        "epochs": 1,
        "optimizer_steps": checkpoint["global_step"],
        "train_windows": len(loaders[0].dataset),
        "validation_windows": len(loaders[1].dataset),
        "initial_validation_loss": initial_loss,
        **metrics[-1],
        "checkpoint_reload_validation_loss": restored_loss,
        "updated_components": updated,
        "frozen_backbone_unchanged": True,
        "generation_completed": True,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "data": "synthetic",
        "language_model": config.model.model_id
        if pretrained
        else "random 2-layer Llama, hidden_size=64",
        "encoder_checkpoint": str(encoder_checkpoint.resolve())
        if encoder_checkpoint
        else None,
        "precision": precision,
        "device": str(device),
        "full_pretrained_1b_run": pretrained,
        "limitations": ["No clinical dataset validation"]
        + (
            []
            if pretrained
            else [
                "No CUDA mixed-precision validation",
                "Random language model and smaller random projector",
            ]
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Require CUDA and test the actual pretrained 1B model",
    )
    args = parser.parse_args()
    run_smoke(args.output_dir, args.encoder_checkpoint, pretrained=args.pretrained)


if __name__ == "__main__":
    main()
