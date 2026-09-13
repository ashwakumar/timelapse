"""Train baselines and evaluate actual predictions on engine-disjoint test windows."""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ACTIVE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


def score_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official NASA C-MAPSS scoring metric (penalizes late predictions more severely than early)."""
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def extract_tabular_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract tabular statistical features (mean, std, min, max, drift) from telemetry windows."""
    features = []
    targets = []
    for r in records:
        row = []
        for s in ACTIVE_SENSORS:
            vals = np.array(r["series"][s], dtype=np.float32)
            row.extend(
                [
                    float(np.mean(vals)),
                    float(np.std(vals)),
                    float(np.min(vals)),
                    float(np.max(vals)),
                    float(vals[-1] - vals[0]),  # Drift
                ]
            )
        features.append(row)
        targets.append(float(r["rul"]))
    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def load_splits(data_path: str) -> tuple[list[dict], list[dict]]:
    """Load records and reject empty or overlapping engine splits."""
    with open(data_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    units: dict[str, set[int]] = {}
    for record in records:
        split = record["split"]
        units.setdefault(split, set()).add(int(record["unit_number"]))
    names = list(units)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if units[name] & units[other]:
                raise ValueError(f"Engine leakage between {name} and {other}")
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    if not train or not test:
        raise ValueError("Benchmark requires nonempty train and test splits")
    return train, test


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


def text_predictions(records: list[dict], model_id: str, device: str) -> np.ndarray:
    """Run a frozen text-only LM on sensor summaries without exposing test labels."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
    model: Any = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    predictions = []
    for i, record in enumerate(records):
        features, _ = extract_tabular_features([record])
        rows = features[0].reshape(len(ACTIVE_SENSORS), 5)
        summary = "\n".join(
            f"{sensor}: " + ", ".join(f"{v:.4g}" for v in row)
            for sensor, row in zip(ACTIVE_SENSORS, rows, strict=True)
        )
        prompt = (
            "Estimate turbofan remaining useful life in cycles from these sensor summaries. "
            "Columns: mean, standard deviation, minimum, maximum, final minus initial.\n"
            f"{summary}\nReply with only one nonnegative number."
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if not isinstance(answer, str):
            raise TypeError("Expected one decoded text response")
        answer = answer.strip()
        match = re.fullmatch(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", answer)
        predictions.append(float(answer) if match else float("nan"))
        if (i + 1) % 100 == 0:
            print(f"Text-only LM: {i + 1}/{len(records)} windows", flush=True)
    return np.array(predictions)


def run_benchmark(
    data_path: str = "data/processed/windows.jsonl",
    out_path: str = "artifacts/benchmark_results.json",
    model_dir: str = "models/aeroguard_tslm",
    baseline_dir: str = "models/baselines",
    chronos_model: str = "amazon/chronos-t5-tiny",
    text_model: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    device: str = "cpu",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Fit train-split baselines and score them alongside a saved AeroGuard checkpoint.

    Args:
        data_path: Processed windows with engine-disjoint split labels.
        out_path: Destination JSON report; predictions use a sibling JSONL file.
        model_dir: Existing trained AeroGuard checkpoint (never retrained here).
        baseline_dir: Destination for fitted baseline parameters.
        chronos_model: Frozen Hugging Face Chronos backbone identifier.
        text_model: Frozen text-only instruction model identifier.
        device: Torch device used for inference.
        batch_size: Chronos embedding batch size.

    Returns:
        Metrics, configuration, and split provenance for this run.
    """
    from chronos import ChronosPipeline

    from scripts.evaluate_chronos_baseline import extract_chronos_representations
    from training.inference import AeroGuardPredictor

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train, test = load_splits(data_path)
    for filename in ("preprocessing.json", "tslm_adapters.pt"):
        if not (Path(model_dir) / filename).is_file():
            raise FileNotFoundError(f"Train AeroGuard first: missing {Path(model_dir) / filename}")
    print(f"Benchmark: {len(train)} train windows, {len(test)} test windows", flush=True)
    save_dir = Path(baseline_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = extract_tabular_features(train)
    X_test, y_test = extract_tabular_features(test)
    predictions = {"Training mean": np.full(y_test.shape, float(y_train.mean()))}

    print("Training XGBoost...", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    predictions["XGBoost"] = np.clip(model.predict(X_test), 0, None)
    model.save_model(save_dir / "xgboost.json")

    print("Extracting frozen Chronos features and training Ridge...", flush=True)
    pipeline = ChronosPipeline.from_pretrained(chronos_model, device_map=device)
    pipeline.model.eval()
    train_features, _ = extract_chronos_representations(pipeline, train, batch_size, device)
    regressor = Ridge(alpha=10.0).fit(train_features, y_train)
    del train_features
    test_features, _ = extract_chronos_representations(pipeline, test, batch_size, device)
    predictions["Chronos + Ridge"] = np.clip(regressor.predict(test_features), 0, None)
    np.savez(save_dir / "chronos_ridge.npz", coef=regressor.coef_, intercept=regressor.intercept_)
    del pipeline, test_features

    print("Evaluating saved AeroGuard scalar prediction head...", flush=True)
    predictor = AeroGuardPredictor(model_dir=model_dir, device=device)
    predictions["AeroGuard TSLM"] = np.array([predictor.predict_rul(r["series"]) for r in test])
    del predictor
    print("Evaluating frozen text-only LM (this may take time)...", flush=True)
    predictions["Text-only LM"] = text_predictions(test, text_model, device)

    config = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "data_path": data_path,
        "model_dir": model_dir,
        "chronos_model": chronos_model,
        "text_model": text_model,
        "device": device,
        "batch_size": batch_size,
        "sensors": ACTIVE_SENSORS,
        "tabular_statistics": ["mean", "std", "min", "max", "drift"],
        "training_mean": float(y_train.mean()),
        "train_units": sorted({int(r["unit_number"]) for r in train}),
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
    (save_dir / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
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
    parser.add_argument("--chronos-model", default="amazon/chronos-t5-tiny")
    parser.add_argument("--text-model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    run_benchmark(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
