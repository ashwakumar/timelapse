import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.compare import compare_models
from evaluation.language import decode_answer
from evaluation.leakage import audit_prepared_splits
from evaluation.models import MajorityClassifier, RidgeMultinomial
from scripts.opentslm_vitaldb_dataset import PreparedVitalDBCorpus


PARAMETERS = [
    "MAP",
    "RemiEffectSite",
    "RR_CO2",
    "RemiRate",
    "BIS_SQI",
]


class TslmEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name)
        labels = {
            "0": "within_3",
            "1": "within_5",
            "2": "within_10",
            "3": "within_15",
            "4": "none_within_15",
        }
        (self.data_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "parameters_in_tensor_order": PARAMETERS,
                    "labels": labels,
                    "interval_sec": 2,
                    "history_sec": 20,
                    "horizon_sec": 900,
                    "normalization": {
                        "median": {
                            name: float(60 + index) for index, name in enumerate(PARAMETERS)
                        },
                        "iqr_scale": {
                            name: float(index + 1) for index, name in enumerate(PARAMETERS)
                        },
                    },
                    "splits": {
                        "train": {"windows": 8, "subjects": 2},
                        "validation": {"windows": 2, "subjects": 1},
                        "test": {"windows": 4, "subjects": 1},
                    },
                }
            )
        )
        rng = np.random.default_rng(0)
        self._write(
            "train",
            windows=8,
            case_ids=[1, 1, 1, 1, 2, 2, 2, 2],
            subject_ids=[10, 10, 10, 10, 20, 20, 20, 20],
            labels=[0, 0, 1, 4, 2, 2, 4, 4],
            rng=rng,
        )
        self._write(
            "validation",
            windows=2,
            case_ids=[3, 3],
            subject_ids=[30, 30],
            labels=[4, 1],
            rng=rng,
        )
        self._write(
            "test",
            windows=4,
            case_ids=[4, 4, 4, 4],
            subject_ids=[40, 40, 40, 40],
            labels=[0, 2, 4, 4],
            rng=rng,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(
        self,
        split: str,
        *,
        windows: int,
        case_ids: list[int],
        subject_ids: list[int],
        labels: list[int],
        rng: np.random.Generator,
    ) -> None:
        values = rng.normal(size=(windows, 10, len(PARAMETERS))).astype(np.float32)
        mask = np.ones_like(values, dtype=np.bool_)
        mask[:, 2, 0] = False
        values[:, 2, 0] = 0
        np.savez_compressed(
            self.data_dir / f"{split}.npz",
            x_values=values,
            x_mask=mask,
            y=np.asarray(labels, dtype=np.int64),
            case_id=np.asarray(case_ids, dtype=np.int64),
            subject_id=np.asarray(subject_ids, dtype=np.int64),
            cutoff_sec=np.arange(120, 120 + 60 * windows, 60, dtype=np.int64),
            future_map_coverage=np.ones(windows, dtype=np.float32),
        )

    def test_audit_rejects_subject_leakage(self) -> None:
        leaked = Path(self.temp.name) / "leaked"
        leaked.mkdir()
        (leaked / "manifest.json").write_text(
            (self.data_dir / "manifest.json").read_text()
        )
        for split in ("train", "validation", "test"):
            (leaked / f"{split}.npz").write_bytes(
                (self.data_dir / f"{split}.npz").read_bytes()
            )
        with np.load(leaked / "test.npz") as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["subject_id"] = np.asarray([10, 10, 10, 10], dtype=np.int64)
        np.savez_compressed(leaked / "test.npz", **payload)
        with self.assertRaisesRegex(ValueError, "Subject identifiers leak"):
            audit_prepared_splits(PreparedVitalDBCorpus(leaked))

    def test_compare_fits_only_train_and_scores_test(self) -> None:
        report = compare_models(self.data_dir, n_examples=2)
        self.assertTrue(report["leakage"]["ok"])
        self.assertTrue(report["leakage"]["subject_disjoint"])
        self.assertTrue(report["leakage"]["case_disjoint"])
        self.assertFalse(report["leakage"]["device_disjoint"])
        self.assertEqual(report["held_out_split"], "test")
        self.assertTrue(report["validation_unused_for_selection"])
        self.assertEqual(report["leakage"]["windows_by_split"]["test"], 4)
        for name in ("majority", "baseline", "tslm"):
            self.assertIn("accuracy", report["models"][name])
            self.assertIn("macro_f1", report["models"][name])
        self.assertIn("Answer:", report["language_examples"][0]["tslm_answer"])
        self.assertIn("INTERPRET:", report["language_examples"][0]["tslm_answer"])

    def test_language_uses_predicted_class_not_gold(self) -> None:
        corpus = PreparedVitalDBCorpus(self.data_dir)
        test = corpus.load_split("test")
        text = decode_answer(test, 0, predicted_label=4)
        self.assertIn("no hypotension within 15 minutes", text)
        gold = corpus.labels[int(test.arrays["y"][0])]
        self.assertEqual(gold, "within_3")

    def test_ridge_predicts_on_held_out_shape(self) -> None:
        features = np.arange(20, dtype=np.float64).reshape(4, 5)
        labels = np.asarray([0, 1, 0, 1])
        model = RidgeMultinomial(n_classes=5, l2=0.1).fit(features, labels)
        pred = model.predict(features)
        self.assertEqual(pred.shape, (4,))
        majority = MajorityClassifier().fit(features, labels)
        self.assertIn(int(majority.predict(features)[0]), {0, 1})


if __name__ == "__main__":
    unittest.main()
