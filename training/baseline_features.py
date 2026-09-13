"""Shared baseline features and split validation."""

import json
from typing import Any

import numpy as np
import torch

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
        targets.append(float(r.get("rul", 0.0)))
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


def extract_chronos_representations(
    pipeline: Any,
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
            all_targets.append(float(r.get("rul", 0.0)))

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
