"""Offline final-evaluation tests using a verified bundle and stub model."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation.compare import _binary_auprc
from evaluation.evaluate_vitaldb import (
    CLASS_ANSWERS,
    classification_metrics,
    evaluate_bundle,
    parse_prediction,
)
from training.vitaldb import DATA_FILES, TASK, fingerprint, sha256


class StubModel:
    def __init__(self) -> None:
        self.outputs = [
            "Answer: hypotension within 3 minutes",
            "hypotension within 5 minutes",
            "MAP may decline, but no canonical label was returned.",
            "Answer: no hypotension within 15 minutes",
        ]
        self.generation_calls = 0
        self.candidate_calls = 0
        self.device = "cpu"
        self.tokenizer = SimpleNamespace(eos_token="<eos>")

    def eval(self):
        return self

    def generate(self, samples, **_kwargs):
        self.assert_input(samples[0])
        index = self.generation_calls
        self.generation_calls += 1
        if index == 4:
            raise RuntimeError("synthetic generation failure")
        return [self.outputs[index]]

    def loss_and_token_count(self, samples):
        sample = samples[0]
        self.candidate_calls += 1
        expected = tuple(answer + self.tokenizer.eos_token for answer in CLASS_ANSWERS)
        if sample["answer"] not in expected:
            raise AssertionError("candidate scoring used a non-canonical target")
        index = expected.index(sample["answer"])
        return float(index + 1), 1

    @staticmethod
    def assert_input(sample):
        if "answer" in sample:
            raise AssertionError("future-derived answer leaked into generation input")


class VitalDBFinalEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self._write_bundle(self.bundle)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _write_bundle(root: Path) -> None:
        root.mkdir()
        parameters = ["MAP", "HR"]
        manifest = {
            "training_contract": TASK,
            "source": "VitalDB offline final-evaluation fixture",
            "parameters_in_tensor_order": parameters,
            "labels": {
                "0": "within_3",
                "1": "within_5",
                "2": "within_10",
                "3": "within_15",
                "4": "none_within_15",
            },
            "normalization": {
                "median": {"MAP": 80.0, "HR": 70.0},
                "iqr_scale": {"MAP": 10.0, "HR": 15.0},
            },
            "interval_sec": 2,
            "history_sec": 20,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        rng = np.random.default_rng(11)
        for split_index, split in enumerate(("train", "validation", "test")):
            values = rng.normal(size=(5, 10, 2)).astype(np.float32)
            mask = np.ones_like(values, dtype=np.bool_)
            mask[0, 3, 1] = False
            values[~mask] = 0
            np.savez_compressed(
                root / f"{split}.npz",
                x_values=values,
                x_mask=mask,
                y=np.arange(5, dtype=np.int64),
                subject_id=np.full(5, split_index + 10, dtype=np.int64),
                case_id=np.full(5, split_index + 20, dtype=np.int64),
                cutoff_sec=np.arange(5, dtype=np.int64) * 60 + 120,
                future_map_coverage=np.ones(5, dtype=np.float32),
            )
        dataset_fingerprint = fingerprint(root)
        (root / "dataset_fingerprint.json").write_text(json.dumps(dataset_fingerprint))
        torch.save(
            {"task_metadata": {"task": TASK, "dataset_fingerprint": dataset_fingerprint}},
            root / "best.pt",
        )
        for name in ("run_config.json", "metrics.json", "split_audit.json"):
            (root / name).write_bytes(b"stub\n")
        files = {
            path.name: sha256(path)
            for path in root.iterdir()
            if path.is_file()
        }
        self_required = set(DATA_FILES) | {
            "best.pt",
            "dataset_fingerprint.json",
            "run_config.json",
            "metrics.json",
            "split_audit.json",
        }
        assert self_required.issubset(files)
        (root / "bundle.json").write_text(json.dumps({
            "schema_version": 1,
            "task": TASK,
            "files": files,
        }))

    def test_parser_accepts_one_canonical_answer_and_rejects_ambiguity(self) -> None:
        self.assertEqual(parse_prediction("ANSWER: No hypotension within 15 minutes."), 4)
        self.assertEqual(parse_prediction("reason\nAnswer: hypotension within 10 minutes"), 2)
        self.assertIsNone(parse_prediction("hypotension soon"))
        self.assertIsNone(parse_prediction(
            "hypotension within 3 minutes or hypotension within 5 minutes"
        ))
        self.assertIsNone(parse_prediction("not hypotension within 3 minutes"))
        self.assertIsNone(parse_prediction(
            "Answer: hypotension within 3 minutes\nAnswer: hypotension within 5 minutes"
        ))

    def test_metrics_reserve_sixth_confusion_column_for_invalid(self) -> None:
        metrics = classification_metrics([0, 1, 2, 3, 4], [0, 1, None, 4, None])
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (5, 6))
        self.assertEqual(metrics["invalid_predictions"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.4)
        self.assertEqual(metrics["confusion_matrix"][2][5], 1)
        self.assertEqual(metrics["confusion_matrix"][4][5], 1)

    def test_bundle_evaluation_never_generates_with_gold_answer(self) -> None:
        model = StubModel()
        output = self.root / "results"
        report = evaluate_bundle(
            self.bundle,
            output,
            model=model,
            score_candidates=True,
        )
        self.assertEqual(model.generation_calls, 5)
        self.assertEqual(model.candidate_calls, 25)
        self.assertEqual(report["evaluated_samples"], 5)
        self.assertEqual(report["patient_count"], 1)
        self.assertEqual(report["metrics"]["invalid_predictions"], 2)
        self.assertEqual(report["failures"]["generation_count"], 2)
        self.assertEqual(np.asarray(report["metrics"]["confusion_matrix"]).shape, (5, 6))
        self.assertEqual(set(report["baselines"]), {"majority", "summary_ridge", "patch_ridge"})
        self.assertEqual(report["candidate_scoring"]["metrics"]["probability_scored_samples"], 5)
        self.assertTrue((output / "report.json").is_file())
        self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 5)

    def test_average_precision_groups_tied_scores(self) -> None:
        labels = np.asarray([0, 4])
        scores = np.asarray([0.5, 0.5])
        self.assertEqual(_binary_auprc(labels, scores), 0.5)

    def test_checksum_failure_happens_before_model_use(self) -> None:
        model = StubModel()
        with (self.bundle / "test.npz").open("ab") as output:
            output.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_bundle(self.bundle, model=model)
        self.assertEqual(model.generation_calls, 0)

    def test_rehashed_checkpoint_cannot_claim_another_dataset(self) -> None:
        checkpoint_path = self.bundle / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["task_metadata"]["dataset_fingerprint"] = {"wrong": "dataset"}
        torch.save(checkpoint, checkpoint_path)
        bundle_manifest_path = self.bundle / "bundle.json"
        bundle_manifest = json.loads(bundle_manifest_path.read_text())
        bundle_manifest["files"]["best.pt"] = sha256(checkpoint_path)
        bundle_manifest_path.write_text(json.dumps(bundle_manifest))

        model = StubModel()
        with self.assertRaisesRegex(ValueError, "bound|dataset"):
            evaluate_bundle(self.bundle, model=model)
        self.assertEqual(model.generation_calls, 0)


if __name__ == "__main__":
    unittest.main()
