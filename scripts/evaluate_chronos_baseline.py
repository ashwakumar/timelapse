"""Chronos Pure Time-Series Foundation Baseline for AeroGuard TSLM.

Evaluates Amazon Chronos (amazon/chronos-t5-tiny) foundation time-series representations
for Remaining Useful Life (RUL) regression on held-out engines (81-100) with zero leakage.
Compares against Classical ML (XGBoost) and a Naive Reference predictor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from chronos import ChronosPipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure OpenMP and PyTorch thread limits for macOS stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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
    """Official NASA C-MAPSS asymmetric scoring function.

    Penalizes late predictions (unsafe, risk of in-flight failure) more heavily than early ones.

    Args:
        y_true: Ground-truth RUL array.
        y_pred: Predicted RUL array.

    Returns:
        Cumulative asymmetric error score.
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def extract_tabular_features(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract statistical summary features for Classical ML (XGBoost).

    Args:
        records: Processed window records.

    Returns:
        Feature matrix (N, 70) and target array (N,).
    """
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


def extract_chronos_representations(
    pipeline: ChronosPipeline,
    records: list[dict[str, Any]],
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract foundation temporal representations from Amazon Chronos.

    For each 30-cycle window, extracts Chronos embeddings across all 14 sensor channels,
    mean-pools across time tokens, and flattens into a rich temporal representation vector.

    Args:
        pipeline: Pretrained ChronosPipeline instance.
        records: Processed window records.
        batch_size: Number of window records per forward pass.
        device: PyTorch device ('cpu', 'cuda', 'mps').

    Returns:
        Chronos feature matrix (N, 14 * 256) and target array (N,).
    """
    num_records = len(records)
    all_features = []
    all_targets = []

    for i in range(0, num_records, batch_size):
        batch = records[i : i + batch_size]
        batch_tensors = []
        for r in batch:
            # Shape: [14, 30]
            sensors_window = [r["series"][s] for s in ACTIVE_SENSORS]
            batch_tensors.append(sensors_window)
            all_targets.append(float(r["rul"]))

        # Convert to tensor: [B, 14, 30] -> reshape to [B * 14, 30] for Chronos univariate tokenizer
        tensor_batch = torch.tensor(batch_tensors, dtype=torch.float32, device=device)
        b_size = tensor_batch.shape[0]
        flat_series = tensor_batch.view(b_size * len(ACTIVE_SENSORS), -1)

        with torch.no_grad():
            embs, _ = pipeline.embed(flat_series)
            # embs: [B * 14, tokens, hidden_dim]
            pooled = embs.mean(dim=1)  # [B * 14, hidden_dim]
            # Reshape back to [B, 14 * hidden_dim]
            window_embs = pooled.view(b_size, -1)
            all_features.append(window_embs.cpu().numpy())

    return np.vstack(all_features), np.array(all_targets, dtype=np.float32)


def run_chronos_benchmark(
    data_path: str = "data/processed/windows.jsonl",
    out_path: str = "artifacts/chronos_benchmark_results.json",
    model_name: str = "amazon/chronos-t5-tiny",
) -> dict[str, Any]:
    """Execute evaluation comparing Naive, XGBoost, and Amazon Chronos baselines.

    Args:
        data_path: Path to processed windows JSONL.
        out_path: Output JSON path for benchmark metrics.
        model_name: Chronos HuggingFace repository ID.

    Returns:
        Structured benchmark results dictionary.
    """
    print("=" * 70)
    print("AEROGUARD BENCHMARK: CHRONOS TIME-SERIES FOUNDATION MODEL VS BASELINES")
    print("Zero Leakage: Train Engines 1-70 | Held-Out Test Engines 81-100")
    print("=" * 70)

    # 1. Load data
    with open(data_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]

    print(f"Loaded {len(train_records)} train windows, {len(test_records)} test windows.")

    y_train = np.array([float(r["rul"]) for r in train_records], dtype=np.float32)
    y_test = np.array([float(r["rul"]) for r in test_records], dtype=np.float32)

    # 2. Baseline 1: Naive Mean Lifetime Predictor
    naive_pred = np.full_like(y_test, np.mean(y_train))
    naive_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
    naive_mae = float(mean_absolute_error(y_test, naive_pred))
    naive_score = score_function(y_test, naive_pred)

    # 3. Baseline 2: Pure Time-Series Foundation Model (Amazon Chronos)
    print(f"\nEvaluating Baseline: Amazon Chronos ({model_name})...")
    device = "cpu"
    pipeline = ChronosPipeline.from_pretrained(model_name, device_map=device)

    print("Extracting Chronos foundation representations for train windows...")
    X_train_chronos, _ = extract_chronos_representations(
        pipeline, train_records, batch_size=64, device=device
    )
    print("Extracting Chronos foundation representations for held-out test windows...")
    X_test_chronos, _ = extract_chronos_representations(
        pipeline, test_records, batch_size=64, device=device
    )

    # Fit lightweight calibrated ridge regression on Chronos foundation embeddings
    chronos_regressor = Ridge(alpha=10.0)
    chronos_regressor.fit(X_train_chronos, y_train)
    chronos_preds = np.clip(chronos_regressor.predict(X_test_chronos), 0, None)

    chronos_rmse = float(np.sqrt(mean_squared_error(y_test, chronos_preds)))
    chronos_mae = float(mean_absolute_error(y_test, chronos_preds))
    chronos_score = score_function(y_test, chronos_preds)

    # 4. Baseline 3: Classical ML (XGBoost Regressor)
    print("\nTraining Baseline: Classical ML (XGBoost)...")
    X_train_xgb, _ = extract_tabular_features(train_records)
    X_test_xgb, _ = extract_tabular_features(test_records)

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    xgb_model.fit(X_train_xgb, y_train)
    xgb_preds = np.clip(xgb_model.predict(X_test_xgb), 0, None)

    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
    xgb_mae = float(mean_absolute_error(y_test, xgb_preds))
    xgb_score = score_function(y_test, xgb_preds)

    results = {
        "Baseline 1: Naive (Training Mean)": {
            "Model_Type": "Heuristic Floor",
            "RMSE": round(naive_rmse, 2),
            "MAE": round(naive_mae, 2),
            "NASA_Score": round(naive_score, 1),
            "Explainability": "None (Static Mean)",
            "Actionability": "Zero",
        },
        "Baseline 2: Classical ML (XGBoost)": {
            "Model_Type": "Gradient Boosted Trees (70 Tabular Features)",
            "RMSE": round(xgb_rmse, 2),
            "MAE": round(xgb_mae, 2),
            "NASA_Score": round(xgb_score, 1),
            "Explainability": "None (Black-Box Scalar)",
            "Actionability": "Low (Bare Number)",
        },
        "Baseline 3: Amazon Chronos (Pure Time-Series Foundation)": {
            "Model_Type": f"T5 Time-Series Foundation Model ({model_name})",
            "RMSE": round(chronos_rmse, 2),
            "MAE": round(chronos_mae, 2),
            "NASA_Score": round(chronos_score, 1),
            "Explainability": "None (Pure Numerical Foundation Model)",
            "Actionability": "Low (No Natural Language Reasoning)",
        },
    }

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Model':<40} | {'RMSE':<8} | {'MAE':<8} | {'NASA Score':<12} | {'Explainability'}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<40} | {r['RMSE']:<8.2f} | {r['MAE']:<8.2f} | {r['NASA_Score']:<12.1f} | {r['Explainability']}"
        )
    print("=" * 80)

    # Save to disk
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to: {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="data/processed/windows.jsonl", help="Path to windows JSONL"
    )
    parser.add_argument(
        "--out-path", default="artifacts/chronos_benchmark_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--model-name", default="amazon/chronos-t5-tiny", help="Chronos model identifier"
    )
    args = parser.parse_args()

    run_chronos_benchmark(args.data_path, args.out_path, args.model_name)
