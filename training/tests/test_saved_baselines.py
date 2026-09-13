"""Saved baselines are reusable and inference never fits weights."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from training.baseline_features import ACTIVE_SENSORS
from training.baseline_inference import BaselinePredictor
from training.train_baselines import train_baselines


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import chronos
    import transformers

    class FakeModel:
        def __init__(self):
            self.model = self

        def eval(self):
            return self

        def save_pretrained(self, path):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / "config.json").write_text("{}")
            (Path(path) / "model.safetensors").touch()
            (Path(path) / "tokenizer_config.json").write_text("{}")

        def embed(self, values):
            return values.unsqueeze(-1), None

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[1, 2]])

    class FakeTokenizer(FakeModel):
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert "999" not in messages[0]["content"]
            return "prompt"

        def __call__(self, *args, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self

            return Inputs(input_ids=torch.tensor([[1]]))

        def decode(self, *args, **kwargs):
            return "12"

    monkeypatch.setattr(chronos.ChronosPipeline, "from_pretrained", lambda *a, **k: FakeModel())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    records = [
        dict(
            unit_number=unit,
            split=split,
            cycle=30,
            rul=rul,
            series={s: [float(unit), float(unit + 1)] for s in ACTIVE_SENSORS},
        )
        for unit, split, rul in [(1, "train", 10), (2, "train", 20), (3, "test", 999)]
    ]
    data = tmp_path / "windows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records))
    root = tmp_path / "baselines"
    train_baselines(str(data), str(root))
    return data, root, records


def test_saved_predictions_and_repeat_preparation_do_not_fit(prepared, monkeypatch):
    data, root, records = prepared

    def no_fit(*args, **kwargs):
        pytest.fail("Inference/repeat preparation must not fit")

    monkeypatch.setattr(Ridge, "fit", no_fit)
    train_baselines(str(data), str(root))
    metadata = json.loads((root / "manifest.json").read_text())
    assert metadata["train_units"] == [1, 2]
    assert metadata["training_mean"] == 15
    record = dict(records[-1])
    record.pop("rul")
    for name in ["Chronos + Ridge", "Text-only LM"]:
        model = BaselinePredictor(name, str(root))
        first = model.predict([record])
        assert np.isfinite(first).all()
        np.testing.assert_allclose(first, BaselinePredictor(name, str(root)).predict([record]))


def test_missing_bundle_has_preparation_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.train_baselines"):
        BaselinePredictor("Chronos + Ridge", str(tmp_path))


def test_benchmark_loads_without_modifying_models(prepared, tmp_path, monkeypatch):
    from training import inference
    from training.evaluate_baselines import run_benchmark

    data, root, _ = prepared

    class FakePredictor:
        def __init__(self, **kwargs):
            pass

        def predict_rul(self, series):
            return 7

    monkeypatch.setattr(inference, "AeroGuardPredictor", FakePredictor)
    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("Benchmark trained"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ["preprocessing.json", "tslm_adapters.pt"]:
        (checkpoint / name).touch()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = run_benchmark(str(data), str(tmp_path / "report.json"), str(checkpoint), str(root))
    assert result["models"]["AeroGuard TSLM"]["RMSE"] == 992
    assert result["models"]["Text-only LM"]["RMSE"] == 987
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = root / "manifest.json"
    metadata = json.loads(manifest.read_text())
    metadata["train_units"].append(3)
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="leakage"):
        run_benchmark(str(data), str(tmp_path / "report.json"), str(checkpoint), str(root))


def test_gui_loads_baselines_and_updates_cycle_results(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from training import baseline_inference, inference

    app_path = Path(__file__).resolve().parents[2] / "demo" / "app.py"
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    records = [
        dict(
            unit_number=84,
            split="test",
            cycle=cycle,
            rul=100,
            series={s: [1.0] * 30 for s in ACTIVE_SENSORS},
        )
        for cycle in [30, 35]
    ]
    (data_dir / "windows.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    monkeypatch.chdir(tmp_path)
    calls = []

    class FakeBaseline:
        def __init__(self, name):
            self.name = name

        def predict(self, records):
            calls.append((self.name, records[0]["cycle"]))
            return np.array([20.0 if records[0]["cycle"] == 30 else 100.0])

    def unavailable(**kwargs):
        raise FileNotFoundError("No test AeroGuard checkpoint")

    monkeypatch.setattr(baseline_inference, "BaselinePredictor", FakeBaseline)
    monkeypatch.setattr(inference, "AeroGuardPredictor", unavailable)
    st.cache_resource.clear()
    st.cache_data.clear()
    try:
        app = AppTest.from_file(str(app_path)).run()
        assert not app.exception
        for label in ["Amazon Chronos", "Text-Only LLM"]:
            next(w for w in app.selectbox if w.label == "Model").select(label).run()
            assert not app.exception
            assert next(w for w in app.metric if w.label == "Predicted RUL").value == "20.0 cycles"
        app.select_slider[0].set_value(35).run()
        assert not app.exception
        assert next(w for w in app.metric if w.label == "Predicted RUL").value == "100.0 cycles"
        assert ("Text-only LM", 35) in calls
    finally:
        st.cache_resource.clear()
        st.cache_data.clear()


def test_chronos_local_export_round_trip(tmp_path):
    """Exercise the real Chronos/Transformers export API without model downloads."""
    from chronos import ChronosPipeline
    from transformers import T5Config, T5ForConditionalGeneration

    from training.baseline_features import extract_chronos_representations

    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
    )
    config.decoder_start_token_id = 0
    config.chronos_config = dict(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=32,
        prediction_length=4,
        n_tokens=32,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        num_samples=1,
        temperature=1.0,
        top_k=10,
        top_p=1.0,
    )
    source = tmp_path / "source"
    T5ForConditionalGeneration(config).save_pretrained(source)
    pipeline = ChronosPipeline.from_pretrained(str(source), local_files_only=True)
    pipeline.model.eval()
    record = {"series": {sensor: [1.0, 2.0, 3.0] for sensor in ACTIVE_SENSORS}}
    features, _ = extract_chronos_representations(pipeline, [record])
    coef = np.full(features.shape[1], 0.1)
    np.savez(tmp_path / "chronos_ridge.npz", coef=coef, intercept=20.0)
    pipeline.model.model.save_pretrained(tmp_path / "chronos")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "sensors": ACTIVE_SENSORS})
    )
    loaded = BaselinePredictor("Chronos + Ridge", str(tmp_path))
    np.testing.assert_allclose(
        loaded.predict([record]), np.clip(features @ coef + 20.0, 0, None), rtol=1e-5
    )


def test_preparation_repairs_missing_downloaded_weights(prepared):
    data, root, _ = prepared
    weights = root / "chronos" / "model.safetensors"
    weights.unlink()
    train_baselines(str(data), str(root))
    assert weights.is_file()
