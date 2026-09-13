"""Multinomial ridge classifiers used for the baseline and the TSLM head."""

from __future__ import annotations

import numpy as np


class RidgeMultinomial:
    """One-versus-rest ridge regression with a softmax readout.

    Closed form, no sklearn required. Fit only on the training matrix.
    """

    def __init__(self, n_classes: int = 5, l2: float = 1.0) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.n_classes = n_classes
        self.l2 = float(l2)
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RidgeMultinomial":
        labels = labels.astype(int)
        if set(labels.tolist()) - set(range(self.n_classes)):
            raise ValueError("labels contain an index outside n_classes")
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        design = np.concatenate(
            [np.ones((features.shape[0], 1)), (features - self.mean) / self.scale],
            axis=1,
        )
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += self.l2
        inverse = np.linalg.pinv(gram)
        weights = []
        for klass in range(self.n_classes):
            target = (labels == klass).astype(np.float64)
            weights.append(inverse @ (design.T @ target))
        self.weights = np.stack(weights, axis=1)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("RidgeMultinomial must be fit before scoring")
        design = np.concatenate(
            [np.ones((features.shape[0], 1)), (features - self.mean) / self.scale],
            axis=1,
        )
        return design @ self.weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


class MajorityClassifier:
    def __init__(self) -> None:
        self.label = 0

    def fit(self, _features: np.ndarray, labels: np.ndarray) -> "MajorityClassifier":
        values, counts = np.unique(labels.astype(int), return_counts=True)
        self.label = int(values[int(np.argmax(counts))])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.label, dtype=int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probs = np.zeros((features.shape[0], 5), dtype=np.float64)
        probs[:, self.label] = 1.0
        return probs
