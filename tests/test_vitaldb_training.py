"""Offline integration of real task contracts with a tiny actual Llama + PEFT."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from training.config import ExperimentConfig, ModelConfig, TrainingConfig
from training.model import ModelProvenance, SurgicalTelemetryModel, apply_lora
from training.train_vitaldb import run
from training.vitaldb import TASK, VitalDBOnsetDataset, validate_corpus, verify_bundle
from scripts.export_vitaldb_run import export_run
from scripts.opentslm_vitaldb_dataset import ANSWER_TEXT


def corpus_fixture(root: Path):
    root.mkdir()
    parameters = ["MAP", "HR"]
    manifest = {"training_contract": TASK, "source": "VitalDB (offline test fixture)",
                "parameters_in_tensor_order": parameters,
                "labels": {str(i): v for i, v in enumerate(ANSWER_TEXT)},
                "normalization": {"median": {"MAP": 80, "HR": 70}, "iqr_scale": {"MAP": 10, "HR": 15}},
                "interval_sec": 2, "history_sec": 20}
    (root / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(2)
    for s, name in enumerate(("train", "validation", "test")):
        x = rng.normal(size=(5, 10, 2)).astype(np.float32)
        mask = np.ones_like(x, dtype=bool)
        mask[0, 2, 1] = False
        x[~mask] = 0
        np.savez_compressed(root / f"{name}.npz", x_values=x, x_mask=mask, y=np.arange(5),
                            subject_id=np.full(5, s + 1), case_id=np.full(5, s + 1),
                            cutoff_sec=np.arange(5) * 60 + 60, future_map_coverage=np.ones(5))
    return root


def tiny_model():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tests.test_training_model import PatchEncoder

    torch.manual_seed(7)
    words = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "hypotension", "within", "3", "5", "10", "15", "minutes", "no"]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    backbone = nn.Module()
    backbone.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]")
    backbone.llm = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
        bos_token_id=1, eos_token_id=2, pad_token_id=3))
    backbone.encoder = PatchEncoder(8)
    backbone.projector = nn.Linear(8, 16)
    backbone.patch_size = 4
    config = ModelConfig(model_id="tiny", model_revision="a", base_model_id="tiny", base_model_revision="b",
        lora_rank=2, lora_alpha=4, lora_dropout=0, lora_target_modules=("q_proj", "v_proj"),
        gradient_checkpointing=False, max_context_tokens=2048)
    apply_lora(backbone, config)
    return SurgicalTelemetryModel(backbone, config, ModelProvenance("tiny", "a", "tiny", "b", "test"))


def test_adapter_masks_and_no_future_target_in_inputs(tmp_path):
    root = corpus_fixture(tmp_path / "data")
    ds = VitalDBOnsetDataset(root, "train")
    a = ds[0]
    assert len(a["time_series"]) == 2  # not four interleaved value/mask series
    assert a["signal_valid_mask"].shape == (2, 10)
    assert not a["signal_valid_mask"][1, 2]
    assert a["answer"] == "hypotension within 3 minutes"
    # Future labels must not change any model input field.
    ds.split.arrays["y"][0] = 4
    b = ds[0]
    assert a["pre_prompt"] == b["pre_prompt"] and a["post_prompt"] == b["post_prompt"]
    for x, y in zip(a["time_series"], b["time_series"]):
        np.testing.assert_array_equal(x, y)
    assert a["answer"] != b["answer"]


def test_reject_legacy_before_loading_model(tmp_path):
    root = corpus_fixture(tmp_path / "data")
    p = root / "manifest.json"
    m = json.loads(p.read_text()); m.pop("training_contract"); p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Requires real VitalDB"):
        validate_corpus(root)


def test_train_resume_export_and_checksum(tmp_path):
    torch.set_num_threads(1)
    root = corpus_fixture(tmp_path / "data")
    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    config = ExperimentConfig(model=model.config, training=TrainingConfig(
        output_dir=str(tmp_path / "run"), epochs=2, device="cpu", precision="fp32",
        max_signal_length=12, train_batch_size=2, eval_batch_size=2,
        gradient_accumulation_steps=2, log_every_steps=1))
    run(config, root, model=model)
    for part in ("encoder", "projector", "lora"):
        assert any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if part in name)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    final = {name: p.detach().clone() for name, p in model.named_parameters()}
    config.training.resume_from = str(tmp_path / "run/checkpoint-epoch-001.pt")
    resumed = tiny_model()
    run(config, root, model=resumed)
    for name, p in resumed.named_parameters():
        torch.testing.assert_close(p, final[name], rtol=0, atol=0)
    bundle = export_run(tmp_path / "run", root, tmp_path / "bundle")
    verify_bundle(bundle)
    weights = torch.load(bundle / "best.pt", weights_only=True)
    assert "optimizer" not in weights and "trainable_state" in weights
    checkpoint_path = tmp_path / "run/checkpoint-epoch-001.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["task_metadata"]["task"] = "synthetic"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="Resume checkpoint task/dataset"):
        run(config, root, model=tiny_model())
    with (bundle / "test.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(bundle)
