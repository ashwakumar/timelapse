"""Regression tests for OpenTSLM time-series padding and missingness semantics."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from training.config import ModelConfig
from training.model import SurgicalTelemetryModel


class _TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.config = SimpleNamespace(max_position_embeddings=128)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.llm = _TinyLanguageModel()
        self.encoder = TransformerCNNEncoder(
            output_dim=8,
            transformer_input_dim=8,
            num_heads=2,
            num_layers=1,
            patch_size=4,
            ff_dim=16,
            max_patches=8,
            dropout=0.0,
        )
        self.projector = nn.Identity()
        self.patch_size = 4
        self.tokenizer = SimpleNamespace()


def _model() -> SurgicalTelemetryModel:
    torch.manual_seed(7)
    model = SurgicalTelemetryModel(
        _TinyBackbone(),
        ModelConfig(max_context_tokens=128, gradient_checkpointing=False),
    )
    model.eval()
    return model


def _sample(values: torch.Tensor, valid: torch.Tensor, *, source_length: int) -> dict:
    return {
        "time_series": [values],
        "signal_valid_mask": valid.unsqueeze(0),
        "signal_length": source_length,
    }


def test_short_channel_embedding_is_invariant_to_longer_batch_peer() -> None:
    """Transformer attention must never see padding introduced only for batching."""

    model = _model()
    short = _sample(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([True, True, True, True, False, False, False, False]),
        source_length=4,
    )
    long = _sample(
        torch.arange(1.0, 9.0),
        torch.ones(8, dtype=torch.bool),
        source_length=8,
    )

    short_alone = model._encode_time_series([short])[0]
    short_batched = model._encode_time_series([short, long])[0]

    for alone_pair, batched_pair in zip(short_alone, short_batched, strict=True):
        for alone, batched in zip(alone_pair, batched_pair, strict=True):
            assert torch.allclose(alone, batched, atol=1e-6, rtol=1e-6)


def test_observation_mask_distinguishes_mean_value_from_missing_value() -> None:
    """Equal zero-filled values need distinct embeddings when observation differs."""

    model = _model()
    normalized_zeros = torch.zeros(4)
    observed = _sample(
        normalized_zeros,
        torch.ones(4, dtype=torch.bool),
        source_length=4,
    )
    missing = _sample(
        normalized_zeros,
        torch.zeros(4, dtype=torch.bool),
        source_length=4,
    )

    observed_value, observed_mask = model._encode_time_series([observed])[0][0]
    missing_value, missing_mask = model._encode_time_series([missing])[0][0]

    assert torch.allclose(observed_value, missing_value, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(observed_mask, missing_mask, atol=1e-6, rtol=1e-6)
