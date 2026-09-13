"""Feature views for the classical baseline and the patch TSLM."""

from __future__ import annotations

import numpy as np

from scripts.opentslm_vitaldb_dataset import PreparedVitalDBSplit


def baseline_matrix(split: PreparedVitalDBSplit) -> tuple[np.ndarray, np.ndarray]:
    """Hand-engineered window summaries used as the non-TSLM baseline.

    For every channel: last observed value, mean, standard deviation, first-to-last
    change, and missingness. Statistics are computed on the already train-scaled
    NPZ so the held-out split never refits normalization.
    """

    values = split.arrays["x_values"]
    mask = split.arrays["x_mask"].astype(bool)
    windows, _steps, channels = values.shape
    columns: list[np.ndarray] = []
    for channel in range(channels):
        series = values[:, :, channel]
        observed = mask[:, :, channel]
        safe = np.where(observed, series, np.nan)
        last = _last_observed(safe, observed)
        first = _first_observed(safe, observed)
        mean = np.nanmean(safe, axis=1)
        std = np.nanstd(safe, axis=1)
        delta = last - first
        missing = 1.0 - observed.mean(axis=1)
        for vector in (last, mean, std, delta, missing):
            columns.append(np.nan_to_num(vector, nan=0.0).astype(np.float64))
    features = np.stack(columns, axis=1)
    labels = split.arrays["y"].astype(int)
    return features, labels


def tslm_patch_matrix(
    split: PreparedVitalDBSplit, *, patch_size: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """OpenTSLM-style patches over interleaved value and observation-mask series.

    Each physical channel is followed by its mask, matching
    ``VitalDBHypotensionDataset``. Patches of ``patch_size`` source steps are
    mean-pooled. The classifier then maps those temporal tokens to a language
    class; it does not see future MAP.
    """

    values = split.arrays["x_values"].astype(np.float64)
    mask = split.arrays["x_mask"].astype(np.float64)
    windows, steps, channels = values.shape
    remainder = steps % patch_size
    if remainder:
        pad = patch_size - remainder
        values = np.pad(values, ((0, 0), (0, pad), (0, 0)))
        mask = np.pad(mask, ((0, 0), (0, pad), (0, 0)))
        steps = values.shape[1]
    patches = steps // patch_size
    tokens = []
    for channel in range(channels):
        value_view = values[:, :, channel].reshape(windows, patches, patch_size)
        mask_view = mask[:, :, channel].reshape(windows, patches, patch_size)
        tokens.append(value_view.mean(axis=2))
        tokens.append(mask_view.mean(axis=2))
    features = np.concatenate(tokens, axis=1)
    labels = split.arrays["y"].astype(int)
    return features, labels


def _last_observed(safe: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.zeros(safe.shape[0], dtype=np.float64)
    for row, (series, mask) in enumerate(zip(safe, observed, strict=True)):
        hits = np.flatnonzero(mask)
        result[row] = series[hits[-1]] if hits.size else 0.0
    return result


def _first_observed(safe: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.zeros(safe.shape[0], dtype=np.float64)
    for row, (series, mask) in enumerate(zip(safe, observed, strict=True)):
        hits = np.flatnonzero(mask)
        result[row] = series[hits[0]] if hits.size else 0.0
    return result
