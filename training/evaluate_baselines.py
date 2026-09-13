"""Evaluate saved models on engine-disjoint test windows without retraining."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.baseline_features import ACTIVE_SENSORS, load_splits, score_function
from training.baseline_inference import BaselinePredictor


def prediction_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Score finite predictions, explicitly reporting generation failures and coverage."""
    valid = np.isfinite(predictions)
    result: dict[str, Any] = {
        "evaluated_windows": int(valid.sum()),
        "failed_windows": int((~valid).sum()),
        "coverage": float(valid.mean()),
        "RMSE": None,
        "MAE": None,
        "NASA_Score": None,
    }
    if valid.any():
        result.update(
            RMSE=float(np.sqrt(mean_squared_error(targets[valid], predictions[valid]))),
            MAE=float(mean_absolute_error(targets[valid], predictions[valid])),
            NASA_Score=score_function(targets[valid], predictions[valid]),
        )
    return result


def run_benchmark(
    data_path: str = "data/processed/windows.jsonl",
    out_path: str = "artifacts/benchmark_results.json",
    model_dir: str = "models/aeroguard_tslm",
    baseline_dir: str = "models/baselines",
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Evaluate saved models without fitting or changing their artifacts."""
    from training.inference import AeroGuardPredictor

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train, test = load_splits(data_path)
    for filename in ("preprocessing.json", "tslm_adapters.pt"):
        if not (Path(model_dir) / filename).is_file():
            raise FileNotFoundError(f"Train AeroGuard first: missing {Path(model_dir) / filename}")
    print(f"Benchmark: {len(train)} train windows, {len(test)} test windows", flush=True)
    save_dir = Path(baseline_dir)
    manifest_path = save_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run uv run python -m training.train_baselines first")
    metadata = json.loads(manifest_path.read_text())
    test_units = {int(r["unit_number"]) for r in test}
    if test_units & set(metadata["train_units"]):
        raise ValueError("Engine leakage between saved baseline training and evaluation")
    y_test = np.array([float(r["rul"]) for r in test])
    predictions = {"Training mean": np.full(y_test.shape, metadata["training_mean"])}
    for name in ("Chronos + Ridge", "Text-only LM"):
        print(f"Evaluating saved {name}...", flush=True)
        baseline = BaselinePredictor(name, baseline_dir, device)
        predictions[name] = baseline.predict(test, batch_size)
        del baseline

    print("Evaluating saved AeroGuard scalar prediction head...", flush=True)
    predictor = AeroGuardPredictor(model_dir=model_dir, device=device)
    predictions["AeroGuard TSLM"] = np.array([predictor.predict_rul(r["series"]) for r in test])
    del predictor
    config = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "data_path": data_path,
        "model_dir": model_dir,
        "chronos_model": metadata["chronos_model"],
        "text_model": metadata["text_model"],
        "device": device,
        "batch_size": batch_size,
        "sensors": ACTIVE_SENSORS,
        "tabular_statistics": ["mean", "std", "min", "max", "drift"],
        "training_mean": metadata["training_mean"],
        "train_units": metadata["train_units"],
        "test_units": sorted({int(r["unit_number"]) for r in test}),
        "evaluation_unit": "window",
        "text_metric_policy": "Valid numeric responses only; inspect coverage before comparison",
        "aeroguard_prediction": "Saved temporal encoder and scalar RUL head",
    }
    results = {
        "configuration": config,
        "models": {name: prediction_metrics(y_test, preds) for name, preds in predictions.items()},
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = out_file.with_suffix(out_file.suffix + ".tmp")
    temporary_report.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    temporary_report.replace(out_file)
    with out_file.with_suffix(".predictions.jsonl").open("w") as handle:
        for i, record in enumerate(test):
            row = {
                "unit_number": record["unit_number"],
                "cycle": record.get("cycle"),
                "true_rul": float(y_test[i]),
                "predictions": {
                    name: float(values[i]) if np.isfinite(values[i]) else None
                    for name, values in predictions.items()
                },
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    for name, metrics in results["models"].items():
        print(f"{name}: {metrics}")
    print(f"Results: {out_file}; baseline weights: {save_dir}")
    return results


def main() -> None:
    """Run the complete benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/processed/windows.jsonl")
    parser.add_argument("--out-path", default="artifacts/benchmark_results.json")
    parser.add_argument("--model-dir", default="models/aeroguard_tslm")
    parser.add_argument("--baseline-dir", default="models/baselines")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    run_benchmark(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
