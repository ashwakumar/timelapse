"""Evaluate a verified VitalDB OpenTSLM bundle on one prepared split."""

from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.features import baseline_matrix, tslm_patch_matrix
from evaluation.models import MajorityClassifier, RidgeMultinomial
from scripts.opentslm_vitaldb_dataset import ANSWER_TEXT, SPLITS
from training.vitaldb import TASK, VitalDBOnsetDataset, verify_bundle


CLASS_KEYS = tuple(ANSWER_TEXT)
CLASS_ANSWERS = tuple(ANSWER_TEXT[key] for key in CLASS_KEYS)
INVALID_INDEX = len(CLASS_KEYS)
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*(.*?)\s*$")


def _normalized_answer(text: str) -> str:
    return " ".join(text.strip().lower().split()).rstrip(".!?").strip()


def parse_prediction(text: str) -> int | None:
    """Return a complete canonical answer, never a matching narrative substring."""
    if not isinstance(text, str):
        return None
    answer_lines = _ANSWER_LINE.findall(text)
    candidates = answer_lines or [text]
    normalized = [_normalized_answer(candidate) for candidate in candidates]
    indexes = []
    for candidate in normalized:
        if candidate not in CLASS_ANSWERS:
            return None
        indexes.append(CLASS_ANSWERS.index(candidate))
    return indexes[0] if indexes and len(set(indexes)) == 1 else None


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int | None] | np.ndarray,
) -> dict[str, Any]:
    """Five-class metrics with a sixth confusion column for invalid outputs."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(
        [
            int(value)
            if isinstance(value, Integral)
            and not isinstance(value, bool)
            and int(value) in range(INVALID_INDEX)
            else INVALID_INDEX
            for value in y_pred
        ],
        dtype=int,
    )
    if truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("Metrics require equally sized, non-empty one-dimensional labels")
    if not np.isin(truth, np.arange(INVALID_INDEX)).all():
        raise ValueError("Gold labels must be one of the five canonical classes")

    confusion = np.zeros((INVALID_INDEX, INVALID_INDEX + 1), dtype=int)
    for gold, prediction in zip(truth, predicted, strict=True):
        confusion[gold, prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, key in enumerate(CLASS_KEYS):
        true_positive = int(confusion[index, index])
        predicted_count = int(confusion[:, index].sum())
        support = int(confusion[index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted_count,
        }

    gold_event = truth != INVALID_INDEX - 1
    predicted_event = predicted < INVALID_INDEX - 1
    event_count = int(gold_event.sum())
    no_event_count = int((~gold_event).sum())
    return {
        "accuracy": float(np.trace(confusion[:, :INVALID_INDEX]) / len(truth)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion.tolist(),
        "confusion_rows": list(CLASS_KEYS),
        "confusion_columns": [*CLASS_KEYS, "invalid"],
        "per_class": per_class,
        "event_sensitivity": (
            float(np.logical_and(gold_event, predicted_event).sum() / event_count)
            if event_count else None
        ),
        "event_false_positive_rate": (
            float(np.logical_and(~gold_event, predicted_event).sum() / no_event_count)
            if no_event_count else None
        ),
        "class_counts": {
            "gold": {key: int((truth == index).sum()) for index, key in enumerate(CLASS_KEYS)},
            "predicted": {
                **{key: int((predicted == index).sum()) for index, key in enumerate(CLASS_KEYS)},
                "invalid": int((predicted == INVALID_INDEX).sum()),
            },
        },
        "invalid_predictions": int((predicted == INVALID_INDEX).sum()),
        "samples": int(len(truth)),
    }


def _probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Tie-correct event discrimination metrics from scikit-learn."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    finite = np.isfinite(probabilities).all(axis=1)
    gold_event = y_true[finite] != INVALID_INDEX - 1
    event_scores = probabilities[finite, : INVALID_INDEX - 1].sum(axis=1)
    if not len(event_scores) or len(np.unique(gold_event)) != 2:
        average_precision = auroc = None
    else:
        average_precision = float(average_precision_score(gold_event, event_scores))
        auroc = float(roc_auc_score(gold_event, event_scores))
    return {
        "event_average_precision": average_precision,
        "event_auroc": auroc,
        "probability_scored_samples": int(finite.sum()),
    }


def _softmax(values: Sequence[float]) -> list[float]:
    scores = np.asarray(values, dtype=np.float64)
    shifted = scores - scores.max()
    probabilities = np.exp(shifted)
    return (probabilities / probabilities.sum()).tolist()


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return float(value)


def _candidate_log_likelihoods(model: Any, model_input: Mapping[str, Any]) -> list[float]:
    """Score fixed public answers without consulting the sample's future-derived gold."""
    scores = []
    tokenizer = getattr(model, "tokenizer", None)
    eos_token = getattr(tokenizer, "eos_token", "") or ""
    for answer in CLASS_ANSWERS:
        candidate = dict(model_input)
        candidate["answer"] = answer + eos_token
        try:
            import torch

            inference = torch.inference_mode()
        except ModuleNotFoundError:
            inference = nullcontext()
        with inference:
            loss, token_count = model.loss_and_token_count([candidate])
        token_count = int(token_count)
        score = -_as_float(loss) * token_count
        if token_count < 1 or not np.isfinite(score):
            raise ValueError("Candidate scoring returned a non-finite or empty likelihood")
        scores.append(score)
    return scores


def _generate_one(model: Any, model_input: Mapping[str, Any]) -> str:
    generated = model.generate([dict(model_input)], max_new_tokens=32, do_sample=False)
    if isinstance(generated, str):
        return generated
    if not isinstance(generated, Sequence) or len(generated) != 1:
        raise TypeError("model.generate must return one string for one input sample")
    if not isinstance(generated[0], str):
        raise TypeError("model.generate returned a non-string prediction")
    return generated[0]


def _baseline_reports(corpus: Any, split_name: str, indexes: np.ndarray) -> tuple[dict, dict]:
    train = corpus.load_split("train")
    evaluated = corpus.load_split(split_name)
    summary_train, train_labels = baseline_matrix(train)
    summary_evaluated, evaluated_labels = baseline_matrix(evaluated)
    patch_train, patch_labels = tslm_patch_matrix(train)
    patch_evaluated, patch_evaluated_labels = tslm_patch_matrix(evaluated)
    if not np.array_equal(train_labels, patch_labels) or not np.array_equal(
        evaluated_labels, patch_evaluated_labels
    ):
        raise RuntimeError("Baseline feature views produced different labels")

    models = {
        "majority": (
            MajorityClassifier().fit(summary_train, train_labels),
            summary_evaluated,
        ),
        "summary_ridge": (
            RidgeMultinomial(INVALID_INDEX).fit(summary_train, train_labels),
            summary_evaluated,
        ),
        "patch_ridge": (
            RidgeMultinomial(INVALID_INDEX).fit(patch_train, train_labels),
            patch_evaluated,
        ),
    }
    reports: dict[str, Any] = {}
    row_values: dict[str, list[dict[str, Any]]] = {}
    gold = evaluated_labels[indexes]
    for name, (model, features) in models.items():
        probabilities = model.predict_proba(features[indexes])
        predictions = probabilities.argmax(axis=1)
        reports[name] = {
            "fit_split": "train",
            "metrics": {
                **classification_metrics(gold, predictions),
                **_probability_metrics(gold, probabilities),
            },
        }
        row_values[name] = [
            {
                "prediction_index": int(prediction),
                "prediction_label": CLASS_KEYS[int(prediction)],
                "event_probability": float(probability[: INVALID_INDEX - 1].sum()),
            }
            for prediction, probability in zip(predictions, probabilities, strict=True)
        ]
    return reports, row_values


def evaluate_bundle(
    bundle: str | Path,
    output_dir: str | Path | None = None,
    device: str = "auto",
    split: str = "test",
    max_samples: int | None = None,
    score_candidates: bool = False,
    model: Any | None = None,
) -> dict[str, Any]:
    """Verify, load, and evaluate a portable fine-tuned VitalDB bundle."""
    bundle_path = Path(bundle)
    # This must precede model loading: a corrupt bundle must never trigger Hub access.
    corpus = verify_bundle(bundle_path)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, Integral)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be a positive integer")

    dataset = VitalDBOnsetDataset(bundle_path, split, eos_token="")
    count = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    indexes = np.arange(count, dtype=int)
    if model is None:
        from training.model import load_for_inference
        from training.train import resolve_device

        resolved_device = str(resolve_device(device))
        model = load_for_inference(bundle_path / "best.pt", resolved_device)
    else:
        resolved_device = str(getattr(model, "device", device))
    if hasattr(model, "eval"):
        model.eval()

    split_arrays = dataset.split.arrays
    gold = split_arrays["y"][indexes].astype(int)
    predictions: list[int | None] = []
    rows: list[dict[str, Any]] = []
    generation_failures = []
    candidate_probabilities = np.full((count, INVALID_INDEX), np.nan, dtype=np.float64)
    candidate_predictions: list[int | None] = [None] * count
    candidate_failures = []

    for row_index, source_index in enumerate(indexes):
        sample = dataset[int(source_index)]
        # Gold is retained only in local evaluation metadata, never in generation input.
        model_input = {key: value for key, value in sample.items() if key != "answer"}
        failure = None
        try:
            output = _generate_one(model, model_input).strip()
            prediction = parse_prediction(output)
            if prediction is None:
                failure = "InvalidPrediction: output is not exactly one canonical answer"
                generation_failures.append({
                    "sample_id": sample["sample_id"],
                    "error": failure,
                })
        except Exception as exc:  # one bad sample must not erase a long evaluation
            output = ""
            prediction = None
            failure = f"{type(exc).__name__}: {exc}"
            generation_failures.append({"sample_id": sample["sample_id"], "error": failure})
        predictions.append(prediction)
        row = {
            "sample_id": sample["sample_id"],
            "subject_id": int(split_arrays["subject_id"][source_index]),
            "case_id": int(split_arrays["case_id"][source_index]),
            "cutoff_sec": int(split_arrays["cutoff_sec"][source_index]),
            "gold_index": int(gold[row_index]),
            "gold_label": CLASS_KEYS[int(gold[row_index])],
            "prediction_index": prediction,
            "prediction_label": CLASS_KEYS[prediction] if prediction is not None else "invalid",
            "model_output": output,
            "failure": failure,
        }
        if score_candidates:
            try:
                likelihoods = _candidate_log_likelihoods(model, model_input)
                probabilities = _softmax(likelihoods)
                candidate = int(np.argmax(likelihoods))
                candidate_probabilities[row_index] = probabilities
                candidate_predictions[row_index] = candidate
                row["candidate_scoring"] = {
                    "log_likelihoods": dict(zip(CLASS_KEYS, likelihoods, strict=True)),
                    "probabilities": dict(zip(CLASS_KEYS, probabilities, strict=True)),
                    "prediction_index": candidate,
                    "prediction_label": CLASS_KEYS[candidate],
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                row["candidate_scoring"] = {"failure": error}
                candidate_failures.append({"sample_id": sample["sample_id"], "error": error})
        rows.append(row)

    baselines, baseline_rows = _baseline_reports(corpus, split, indexes)
    for row_index, row in enumerate(rows):
        row["baselines"] = {
            name: values[row_index] for name, values in baseline_rows.items()
        }

    metrics = classification_metrics(gold, predictions)
    report: dict[str, Any] = {
        "task": TASK,
        "bundle": str(bundle_path.resolve()),
        "checkpoint": str((bundle_path / "best.pt").resolve()),
        "device": resolved_device,
        "split": split,
        "diagnostic_subset": max_samples is not None,
        "evaluated_samples": count,
        "available_samples": len(dataset),
        "patient_count": int(len(np.unique(split_arrays["subject_id"][indexes]))),
        "case_count": int(len(np.unique(split_arrays["case_id"][indexes]))),
        "metrics": metrics,
        "failures": {
            "generation": generation_failures,
            "generation_count": len(generation_failures),
            "invalid_output_count": metrics["invalid_predictions"],
            "total_count": len(generation_failures),
        },
        "baselines": baselines,
        "predictions": rows,
    }
    if score_candidates:
        candidate_metrics = classification_metrics(gold, candidate_predictions)
        report["candidate_scoring"] = {
            "answers": dict(zip(CLASS_KEYS, CLASS_ANSWERS, strict=True)),
            "metrics": {
                **candidate_metrics,
                **_probability_metrics(gold, candidate_probabilities),
            },
            "failures": candidate_failures,
            "failure_count": len(candidate_failures),
        }

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (destination / "predictions.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, allow_nan=False) + "\n")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--score-candidates",
        action="store_true",
        help="Also score the five public canonical answers by conditional log likelihood",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = evaluate_bundle(
        args.bundle,
        args.output_dir,
        device=args.device,
        split=args.split,
        max_samples=args.max_samples,
        score_candidates=args.score_candidates,
    )
    print(json.dumps({
        "report": str((args.output_dir / "report.json").resolve()),
        "split": report["split"],
        "evaluated_samples": report["evaluated_samples"],
        "diagnostic_subset": report["diagnostic_subset"],
        "accuracy": report["metrics"]["accuracy"],
        "macro_f1": report["metrics"]["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
