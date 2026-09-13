"""Regression checks for measured, engine-disjoint benchmark reports."""

import json

import numpy as np
import pytest

from training.evaluate_baselines import (
    ACTIVE_SENSORS,
    load_splits,
    prediction_metrics,
    run_benchmark,
)


def test_rejects_engine_leakage(tmp_path):
    path = tmp_path / "windows.jsonl"
    path.write_text(
        "\n".join(json.dumps({"unit_number": 1, "split": split}) for split in ["train", "test"])
    )
    with pytest.raises(ValueError, match="leakage"):
        load_splits(str(path))


def test_invalid_generations_are_reported():
    result = prediction_metrics(np.array([10.0, 20.0]), np.array([12.0, np.nan]))
    assert result["RMSE"] == 2
    assert result["failed_windows"] == 1
    assert result["coverage"] == 0.5
    assert prediction_metrics(np.array([10.0]), np.array([np.nan]))["RMSE"] is None


def test_report_uses_predictions_and_saves_baselines(tmp_path, monkeypatch):
    import chronos

    from scripts import evaluate_chronos_baseline
    from training import evaluate_baselines, inference

    class FakeChronos:
        model = None

        def __init__(self):
            self.model = self

        def eval(self):
            return self

    class FakePredictor:
        def __init__(self, **kwargs):
            pass

        def predict_rul(self, series):
            return 7.0

    monkeypatch.setattr(chronos.ChronosPipeline, "from_pretrained", lambda *a, **k: FakeChronos())
    monkeypatch.setattr(
        evaluate_chronos_baseline,
        "extract_chronos_representations",
        lambda pipeline, records, *args: (np.ones((len(records), 2)), np.zeros(len(records))),
    )
    monkeypatch.setattr(inference, "AeroGuardPredictor", FakePredictor)
    monkeypatch.setattr(evaluate_baselines, "text_predictions", lambda *a: np.array([np.nan]))
    records = [
        {
            "unit_number": unit,
            "split": split,
            "cycle": 30,
            "rul": rul,
            "series": {sensor: [1.0, 2.0, 3.0] for sensor in ACTIVE_SENSORS},
        }
        for unit, split, rul in [(1, "train", 10), (2, "train", 20), (3, "test", 5)]
    ]
    data = tmp_path / "windows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in records))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ["preprocessing.json", "tslm_adapters.pt"]:
        (checkpoint / name).touch()
    report = tmp_path / "results.json"
    baseline_dir = tmp_path / "baselines"
    result = run_benchmark(str(data), str(report), str(checkpoint), str(baseline_dir))
    assert result["models"]["AeroGuard TSLM"]["RMSE"] == 2
    assert result["models"]["Training mean"]["RMSE"] == 10
    assert result["models"]["Text-only LM"]["coverage"] == 0
    assert json.loads(report.read_text()) == result
    assert (baseline_dir / "xgboost.json").exists()
    assert (baseline_dir / "chronos_ridge.npz").exists()
    saved = json.loads(report.with_suffix(".predictions.jsonl").read_text())
    assert saved["predictions"]["AeroGuard TSLM"] == 7
    assert saved["predictions"]["Text-only LM"] is None
