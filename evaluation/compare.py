"""Metrics and the train/test comparison of baseline vs TSLM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.features import baseline_matrix, tslm_patch_matrix
from evaluation.language import decode_answer
from evaluation.leakage import audit_prepared_splits
from evaluation.models import MajorityClassifier, RidgeMultinomial
from scripts.opentslm_vitaldb_dataset import PreparedVitalDBCorpus


LABEL_COUNT = 5


@dataclass
class ModelScores:
    name: str
    accuracy: float
    macro_f1: float
    hypotension_auprc: float
    predictions: np.ndarray
    probabilities: np.ndarray

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "hypotension_auprc": self.hypotension_auprc,
        }


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = LABEL_COUNT) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[truth, pred] += 1
    return matrix


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = _confusion(y_true, y_pred)
    scores = []
    for klass in range(matrix.shape[0]):
        tp = matrix[klass, klass]
        fp = matrix[:, klass].sum() - tp
        fn = matrix[klass, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _binary_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for any-onset (classes 0-3) vs none (class 4)."""

    positive = (y_true.astype(int) != 4).astype(int)
    if positive.sum() == 0 or positive.sum() == len(positive):
        return float("nan")
    order = np.argsort(-scores)
    ranked = positive[order]
    tp = np.cumsum(ranked)
    fp = np.cumsum(1 - ranked)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positive.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(precision[1:] * np.diff(recall)))


def score_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> ModelScores:
    hypotension_score = probabilities[:, :4].sum(axis=1)
    return ModelScores(
        name=name,
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=_macro_f1(y_true, y_pred),
        hypotension_auprc=_binary_auprc(y_true, hypotension_score),
        predictions=y_pred,
        probabilities=probabilities,
    )


def compare_models(
    data_dir,
    *,
    l2: float = 1.0,
    patch_size: int = 2,
    n_examples: int = 3,
) -> dict:
    """Train on the patient-disjoint train split; report test, not validation."""

    corpus = PreparedVitalDBCorpus(data_dir)
    leakage = audit_prepared_splits(corpus)
    train = corpus.load_split("train")
    test = corpus.load_split("test")

    baseline_x_train, y_train = baseline_matrix(train)
    baseline_x_test, y_test = baseline_matrix(test)
    tslm_x_train, y_train_tslm = tslm_patch_matrix(train, patch_size=patch_size)
    tslm_x_test, y_test_tslm = tslm_patch_matrix(test, patch_size=patch_size)
    if not np.array_equal(y_train, y_train_tslm) or not np.array_equal(y_test, y_test_tslm):
        raise RuntimeError("Feature extractors saw different label vectors")

    majority = MajorityClassifier().fit(baseline_x_train, y_train)
    baseline = RidgeMultinomial(LABEL_COUNT, l2=l2).fit(baseline_x_train, y_train)
    tslm = RidgeMultinomial(LABEL_COUNT, l2=l2).fit(tslm_x_train, y_train)

    majority_scores = score_model(
        "majority",
        y_test,
        majority.predict(baseline_x_test),
        majority.predict_proba(baseline_x_test),
    )
    baseline_scores = score_model(
        "window_summary_ridge",
        y_test,
        baseline.predict(baseline_x_test),
        baseline.predict_proba(baseline_x_test),
    )
    tslm_pred = tslm.predict(tslm_x_test)
    tslm_scores = score_model(
        "opentslm_patch_tslm",
        y_test,
        tslm_pred,
        tslm.predict_proba(tslm_x_test),
    )

    examples = []
    for index in range(min(n_examples, len(test))):
        examples.append(
            {
                "sample_id": (
                    f"vitaldb-case-{int(test.arrays['case_id'][index])}"
                    f"-cutoff-{int(test.arrays['cutoff_sec'][index])}"
                ),
                "subject_id": int(test.arrays["subject_id"][index]),
                "gold_label": corpus.labels[int(y_test[index])],
                "tslm_label": corpus.labels[int(tslm_pred[index])],
                "tslm_answer": decode_answer(test, index, int(tslm_pred[index])),
            }
        )

    return {
        "data_dir": str(corpus.data_dir),
        "parameters": list(corpus.parameters),
        "held_out_split": "test",
        "validation_unused_for_selection": True,
        "leakage": leakage,
        "models": {
            "majority": majority_scores.to_dict(),
            "baseline": baseline_scores.to_dict(),
            "tslm": tslm_scores.to_dict(),
        },
        "delta_tslm_minus_baseline": {
            "accuracy": tslm_scores.accuracy - baseline_scores.accuracy,
            "macro_f1": tslm_scores.macro_f1 - baseline_scores.macro_f1,
        },
        "language_examples": examples,
        "opentslm_finetune": {
            "status": "not_run_in_compare_models",
            "note": (
                "This comparison trains an OpenTSLM-contract TSLM head on patched "
                "value/mask series. Full SoftPrompt+LoRA fine-tuning of "
                "OpenTSLM/llama-3.2-1b-tsqa-sp is scripts/finetune_opentslm_hypotension.py "
                "in the training environment."
            ),
        },
    }
