"""Regression checks for measured, engine-disjoint benchmark reports."""

import json

import numpy as np
import pytest

from training.evaluate_baselines import (
    load_splits,
    prediction_metrics,
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
