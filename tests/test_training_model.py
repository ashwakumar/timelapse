"""Offline tests for the corrected OpenTSLM-SP multimodal training path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from torch import nn  # noqa: E402

from training.config import ModelConfig  # noqa: E402
from training.model import (  # noqa: E402
    ModelProvenance,
    SurgicalTelemetryModel,
    apply_lora,
    load_checkpoint,
    prepare_backbone_for_training,
    save_checkpoint,
)


class TinyTokenizer:
    """Dependency-free character tokenizer with the Hugging Face call shape."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    eos_token = "<eos>"

    def __call__(self, text, *, add_special_tokens=True, **_):
        ids = [3 + ord(character) % 89 for character in text]
        return {"input_ids": ([self.bos_token_id] + ids) if add_special_tokens else ids}

    def decode(self, ids, **_):
        return "".join(
            chr((int(token) - 3) % 89 + 32) for token in ids if int(token) > 2
        )


class PatchEncoder(nn.Module):
    def __init__(self, width: int, patch_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(1, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, values):
        return self.conv(values.unsqueeze(1)).transpose(1, 2)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        llama_config = transformers.LlamaConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        self.llm = transformers.LlamaForCausalLM(llama_config)
        self.encoder = PatchEncoder(12)
        self.projector = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 32))
        self.tokenizer = TinyTokenizer()
        self.patch_size = 4


def make_model(seed: int = 7) -> SurgicalTelemetryModel:
    torch.manual_seed(seed)
    backbone = TinyBackbone()
    config = ModelConfig(
        model_id="tiny",
        model_revision="abc",
        base_model_revision="def",
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("q_proj", "v_proj"),
        gradient_checkpointing=False,
        max_context_tokens=512,
    )
    apply_lora(backbone, config)
    provenance = ModelProvenance("tiny", "abc", "tiny-base", "def", "test")
    return SurgicalTelemetryModel(backbone, config, provenance)


def sample(answer: str, *, missing_tail: bool = False):
    values = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10
    valid = torch.ones_like(values, dtype=torch.bool)
    if missing_tail:
        valid[0, -3:] = False
        values[0, -3:] = 0.0
    return {
        "pre_prompt": "Q",
        "time_series_text": ["a", "h", "s", "c"],
        "time_series": values,
        "signal_valid_mask": valid,
        "signal_length": 8,
        "post_prompt": "A",
        "answer": answer,
        "sample_id": answer,
    }


def test_multimodal_lora_backward_masks_prompt_and_padding():
    model = make_model()
    batch = {"opentslm_batch": [sample("ok"), sample("longer", missing_tail=True)]}
    prepared = model.prepare_batch(batch)

    for row, (prompt_count, answer_count) in enumerate(
        zip(prepared["prompt_token_counts"], prepared["answer_token_counts"])
    ):
        prompt_count, answer_count = int(prompt_count), int(answer_count)
        assert torch.all(prepared["labels"][row, :prompt_count] == -100)
        assert torch.all(
            prepared["labels"][row, prompt_count : prompt_count + answer_count] != -100
        )
        assert torch.all(prepared["labels"][row, prompt_count + answer_count :] == -100)

    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.backbone.encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.projector.parameters()
    )
    assert any(
        parameter.grad is not None
        for name, parameter in model.llm.named_parameters()
        if "lora_" in name
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.llm.named_parameters()
        if "base_layer" in name
    )


def test_checkpoint_roundtrip_restores_all_trainable_components(tmp_path):
    source = make_model(seed=1)
    source.compute_loss([sample("reason")]).backward()
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.01)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, source, epoch=2, global_step=9, metrics={"validation_loss": 1.5}
    )

    restored = make_model(seed=99)
    metadata = load_checkpoint(path, restored)
    assert metadata["epoch"] == 2
    source_trainable = {
        name: value for name, value in source.named_parameters() if value.requires_grad
    }
    restored_trainable = {
        name: value
        for name, value in restored.named_parameters()
        if value.requires_grad
    }
    assert source_trainable.keys() == restored_trainable.keys()
    for name in source_trainable:
        torch.testing.assert_close(source_trainable[name], restored_trainable[name])


def test_target_is_never_silently_truncated():
    model = make_model()
    too_long = sample("x" * 600)
    with pytest.raises(ValueError, match="Target.*exceeding"):
        model.prepare_batch([too_long])


def test_fp16_is_limited_to_frozen_llm_weights():
    backbone = TinyBackbone()
    config = ModelConfig(
        model_id="tiny",
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=("q_proj", "v_proj"),
        gradient_checkpointing=False,
    )
    prepare_backbone_for_training(backbone, config, torch.float16)
    trainable = [
        parameter for parameter in backbone.parameters() if parameter.requires_grad
    ]
    frozen_llm = [
        parameter
        for parameter in backbone.llm.parameters()
        if not parameter.requires_grad
    ]
    assert trainable and all(
        parameter.dtype == torch.float32 for parameter in trainable
    )
    assert frozen_llm and all(
        parameter.dtype == torch.float16 for parameter in frozen_llm
    )
