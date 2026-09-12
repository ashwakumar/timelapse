import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.opentslm_vitaldb_dataset import (
    ANSWER_TEXT,
    PARAMETER_UNITS,
    PreparedVitalDBCorpus,
    VitalDBHypotensionDataset,
)


PARAMETERS = [
    "MAP",
    "RemiEffectSite",
    "RR_CO2",
    "RemiRate",
    "BIS_SQI",
    "InCO2",
    "EtCO2",
    "BIS_EMG",
    "BIS",
    "BIS_SR",
]


class OpenTSLMVitalDBDatasetTest(unittest.TestCase):
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
        manifest = {
            "parameters_in_tensor_order": PARAMETERS,
            "labels": labels,
            "interval_sec": 2,
            "history_sec": 20,
            "horizon_sec": 900,
            "normalization": {
                "median": {name: float(60 + index) for index, name in enumerate(PARAMETERS)},
                "iqr_scale": {name: float(index + 1) for index, name in enumerate(PARAMETERS)},
            },
            "splits": {
                "train": {"windows": 2, "subjects": 1},
                "validation": {"windows": 1, "subjects": 1},
                "test": {"windows": 1, "subjects": 1},
            },
        }
        (self.data_dir / "manifest.json").write_text(json.dumps(manifest))

        base_values = np.arange(100, dtype=np.float32).reshape(1, 10, 10) / 10
        base_mask = np.ones((1, 10, 10), dtype=np.bool_)
        base_mask[0, 3, 0] = False
        base_values[0, 3, 0] = 0
        self._write_split(
            "train",
            np.repeat(base_values, 2, axis=0),
            np.repeat(base_mask, 2, axis=0),
            np.asarray([0, 4]),
            np.asarray([11, 11]),
            np.asarray([101, 101]),
        )
        self._write_split("validation", base_values, base_mask, np.asarray([1]), np.asarray([22]), np.asarray([202]))
        self._write_split("test", base_values, base_mask, np.asarray([2]), np.asarray([33]), np.asarray([303]))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_split(
        self,
        split: str,
        values: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        case_id: np.ndarray,
        subject_id: np.ndarray,
    ) -> None:
        count = len(y)
        np.savez_compressed(
            self.data_dir / f"{split}.npz",
            x_values=values,
            x_mask=mask,
            y=y.astype(np.int64),
            case_id=case_id.astype(np.int64),
            subject_id=subject_id.astype(np.int64),
            cutoff_sec=np.full(count, 120, dtype=np.int64),
            future_map_coverage=np.ones(count, dtype=np.float32),
        )

    def test_validates_patient_disjoint_splits(self) -> None:
        report = PreparedVitalDBCorpus(self.data_dir).validate()
        self.assertEqual(report["splits"]["train"]["windows"], 2)
        self.assertEqual(report["splits"]["test"]["subjects"], 1)
        self.assertEqual(report["splits"]["train"]["series_per_sample"], 20)

    def test_emits_verified_opentslm_contract_and_exact_mask(self) -> None:
        dataset = VitalDBHypotensionDataset("train", EOS_TOKEN="<eos>", data_dir=self.data_dir)
        sample = dataset[0]

        self.assertEqual(
            set(sample),
            {
                "pre_prompt",
                "time_series_text",
                "time_series",
                "post_prompt",
                "answer",
                "sample_id",
                "source_split",
            },
        )
        self.assertEqual(len(sample["time_series"]), 2 * len(PARAMETERS))
        np.testing.assert_array_equal(sample["time_series"][1], dataset.split.arrays["x_mask"][0, :, 0])
        self.assertIn("20-second", sample["pre_prompt"])
        self.assertIn("10 source samples", sample["pre_prompt"])
        self.assertIn("source-missing", sample["pre_prompt"])
        self.assertIn("right-side zeros", sample["pre_prompt"])
        self.assertTrue(sample["answer"].endswith("<eos>"))

    def test_future_label_changes_only_target_answer(self) -> None:
        dataset = VitalDBHypotensionDataset("train", EOS_TOKEN="", data_dir=self.data_dir)
        early = dataset[0]
        no_event = dataset[1]

        for key in ("pre_prompt", "time_series_text", "post_prompt"):
            self.assertEqual(early[key], no_event[key])
        for left, right in zip(early["time_series"], no_event["time_series"], strict=True):
            np.testing.assert_array_equal(left, right)
        self.assertNotEqual(early["answer"], no_event["answer"])
        self.assertIn(ANSWER_TEXT["within_3"], early["answer"])
        self.assertIn(ANSWER_TEXT["none_within_15"], no_event["answer"])

    def test_units_cover_current_top_seventeen_ranking(self) -> None:
        expected = {
            "ST_II": "mm",
            "Temperature": "degC",
            "FeO2": "%",
            "FiO2": "%",
            "BIS_SEF": "Hz",
            "SpO2": "%",
            "HR": "/min",
        }
        self.assertEqual({name: PARAMETER_UNITS[name] for name in expected}, expected)


if __name__ == "__main__":
    unittest.main()
