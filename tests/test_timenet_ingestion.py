import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from connectors.vitaldb.hypotension_windows import VitalDBHypotensionConnector
from scripts.ingest_timenet import audit_dataset
from timenet.client import TimeNet
from timenet.engine import run_pipeline


PARAMETERS = [
    "MAP",
    "RemiEffectSite",
    "RR_CO2",
    "RemiRate",
    "BIS_SQI",
]


class TimeNetIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "prepared"
        self.data_dir.mkdir()
        manifest = {
            "parameters_in_tensor_order": PARAMETERS,
            "labels": {
                "0": "within_3",
                "1": "within_5",
                "2": "within_10",
                "3": "within_15",
                "4": "none_within_15",
            },
            "interval_sec": 2,
            "history_sec": 20,
            "horizon_sec": 900,
            "normalization": {
                "median": {name: float(60 + index) for index, name in enumerate(PARAMETERS)},
                "iqr_scale": {name: float(index + 1) for index, name in enumerate(PARAMETERS)},
            },
            "splits": {
                split: {"windows": 1, "subjects": 1}
                for split in ("train", "validation", "test")
            },
        }
        (self.data_dir / "manifest.json").write_text(json.dumps(manifest))
        for split_index, split in enumerate(("train", "validation", "test")):
            values = np.arange(50, dtype=np.float32).reshape(1, 10, 5) / 10
            mask = np.ones_like(values, dtype=np.bool_)
            mask[0, 3, 0] = False
            values[0, 3, 0] = 0
            np.savez_compressed(
                self.data_dir / f"{split}.npz",
                x_values=values,
                x_mask=mask,
                y=np.asarray([split_index], dtype=np.int64),
                case_id=np.asarray([100 + split_index], dtype=np.int64),
                subject_id=np.asarray([200 + split_index], dtype=np.int64),
                cutoff_sec=np.asarray([120], dtype=np.int64),
                future_map_coverage=np.asarray([1], dtype=np.float32),
            )
        self.previous_data_dir = os.environ.get("VITALDB_PREPARED_DIR")
        os.environ["VITALDB_PREPARED_DIR"] = str(self.data_dir)

    def tearDown(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("VITALDB_PREPARED_DIR", None)
        else:
            os.environ["VITALDB_PREPARED_DIR"] = self.previous_data_dir
        self.temp.cleanup()

    def test_official_sdk_build_and_readback_preserves_masks_and_subject_splits(self) -> None:
        connector = VitalDBHypotensionConnector()
        self.assertEqual(connector.metadata().dataset_id, "hackzurich/vitaldb-hypotension-top-5")

        registry = self.root / "registry"
        cache = self.root / "cache"
        version_dir = run_pipeline(connector, registry, cache_dir=cache)
        self.assertTrue((version_dir / "manifest.json").is_file())

        readback = TimeNet(registry).load("hackzurich/vitaldb-hypotension-top-5@0.1.0")
        report = audit_dataset(
            readback,
            {
                "records": 3,
                "subjects": {"train": 1, "validation": 1, "test": 1},
                "parameters": 5,
            },
        )
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["time_series"], 30)
        self.assertEqual(report["missing_signal_values"], 3)
        self.assertEqual(report["missing_signal_values"], report["mask_zero_values"])
        self.assertEqual(report["subject_overlap"], {})


if __name__ == "__main__":
    unittest.main()
