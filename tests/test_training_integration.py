"""Exercise the real CLI orchestration offline with a tiny Hugging Face Llama."""

import json
import sys

import torch
from torch import nn

from training.config import ExperimentConfig, ModelConfig, TrainingConfig
from training.make_example import make_example
from training.model import SurgicalTelemetryModel, apply_lora


def test_cli_trains_real_multimodal_lora_and_saves_partial_accumulation(
    tmp_path, monkeypatch
):
    from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from training import train as trainer

    torch.manual_seed(7)
    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    backbone = nn.Module()
    backbone.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    backbone.llm = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
    )
    backbone.encoder = TransformerCNNEncoder(
        output_dim=8,
        transformer_input_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        max_patches=32,
    )
    backbone.projector = nn.Linear(8, 16)
    backbone.patch_size = 4
    model_config = ModelConfig(
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=("q_proj", "v_proj"),
        gradient_checkpointing=False,
        max_context_tokens=1024,
    )
    apply_lora(backbone, model_config)
    model = SurgicalTelemetryModel(backbone, model_config)
    original_encoder = backbone.encoder.patch_embed.weight.detach().clone()
    original_base = backbone.llm.get_input_embeddings().weight.detach().clone()
    manifest = make_example(tmp_path / "data")
    output = tmp_path / "run"
    config = ExperimentConfig(
        model=model_config,
        training=TrainingConfig(
            manifest_path=str(manifest),
            output_dir=str(output),
            epochs=2,
            device="cpu",
            precision="fp32",
            max_signal_length=16,
            max_text_length=1024,
            train_batch_size=3,
            eval_batch_size=3,
            gradient_accumulation_steps=4,
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(trainer, "load_pretrained_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(sys, "argv", ["training.train", "--config", str(config_path)])
    trainer.main()

    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert (
        checkpoint["global_step"] == 4
    )  # Each epoch has six batches: one group of four, then one of two.
    assert checkpoint["epoch"] == 2
    assert (output / "best.pt").is_file()
    assert (output / "normalization.json").is_file()
    assert json.loads((output / "split_audit.json").read_text())[
        "disjoint_verification"
    ]["passed"]
    assert not torch.equal(original_encoder, backbone.encoder.patch_embed.weight)
    torch.testing.assert_close(
        original_base, backbone.llm.get_input_embeddings().weight, rtol=0, atol=0
    )
    assert torch.isfinite(torch.tensor(checkpoint["metrics"]["validation_loss"]))
    uninterrupted = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.train",
            "--config",
            str(config_path),
            "--resume-from",
            str(output / "checkpoint-epoch-001.pt"),
        ],
    )
    trainer.main()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, uninterrupted[name], atol=0, rtol=0)
    assert len(json.loads((output / "metrics.json").read_text())) == 2
