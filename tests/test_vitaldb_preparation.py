import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_vitaldb_training import (
    HORIZON_SEC,
    INTERVAL_SEC,
    LABEL_TO_INDEX,
    TARGET_VERSION,
    TRAINING_CONTRACT,
    case_windows,
    feature_screen_spec,
    parse_args,
    prepare_corpus,
    split_subjects,
)
from training.vitaldb import validate_corpus


PARAMETERS = ["MAP", "RemiEffectSite", "RR_CO2", "RemiRate", "BIS_SQI"]


def frame_with_boundary_event(*, missing_index: int | None = None) -> pd.DataFrame:
    clean_samples = 60 // INTERVAL_SEC
    horizon_samples = HORIZON_SEC // INTERVAL_SEC
    confirmation_samples = 60 // INTERVAL_SEC
    length = clean_samples + horizon_samples + confirmation_samples
    values = np.full((length, len(PARAMETERS)), 75.0, dtype=np.float32)
    # At the only eligible cutoff, the sustained run begins exactly 900 seconds later.
    values[clean_samples + horizon_samples :, 0] = 60.0
    if missing_index is not None:
        values[clean_samples + missing_index, 0] = np.nan
    return pd.DataFrame(values, columns=PARAMETERS)


def frame_with_all_labels() -> pd.DataFrame:
    values = np.full((2400, len(PARAMETERS)), 75.0, dtype=np.float32)
    # Two isolated observed episodes produce every horizon bucket plus clean no-events.
    values[900:930, 0] = 60.0
    values[1800:1830, 0] = 60.0
    return pd.DataFrame(values, columns=PARAMETERS)


class FakeVitalDB:
    def __init__(self) -> None:
        self.find_calls: list[tuple[str, ...]] = []
        self.clinical_calls = 0

    def find_cases(self, tracks):
        self.find_calls.append(tuple(tracks))
        # The screen required MAP+HR; the training pool only requires MAP.
        return [1, 2] if len(tracks) == 2 else list(range(1, 11))

    def load_clinical_data(self, *, caseids, params):
        self.clinical_calls += 1
        return pd.DataFrame(
            {"caseid": list(caseids), "subjectid": [1000 + int(v) for v in caseids]}
        )


class VitalDBPreparationTest(unittest.TestCase):
    def test_horizon_boundary_event_uses_confirmation_lookahead(self) -> None:
        windows = case_windows(
            frame_with_boundary_event(), PARAMETERS, case_id=7, subject_id=77
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["label"], LABEL_TO_INDEX["within_15"])
        self.assertEqual(windows[0]["onset_seconds"], 900)
        # Cutoff is the first excluded sample, not the last history timestamp.
        self.assertEqual(windows[0]["cutoff_sec"], 60)

    def test_missing_map_anywhere_in_target_lookahead_rejects_window(self) -> None:
        # A missing confirmation sample cannot be treated as evidence of no event.
        frame = frame_with_boundary_event(missing_index=HORIZON_SEC // INTERVAL_SEC + 5)
        self.assertEqual(case_windows(frame, PARAMETERS, 7, 77), [])

    def test_split_is_deterministic_exact_and_subject_disjoint(self) -> None:
        subjects = np.repeat(np.arange(20), 2)
        first = split_subjects(subjects, seed=19)
        second = split_subjects(subjects[::-1], seed=19)

        self.assertEqual(first, second)
        self.assertEqual({name: len(values) for name, values in first.items()}, {
            "train": 14,
            "validation": 3,
            "test": 3,
        })
        self.assertFalse(first["train"] & first["validation"])
        self.assertFalse(first["train"] & first["test"])
        self.assertFalse(first["validation"] & first["test"])
        self.assertEqual(set().union(*first.values()), set(range(20)))

    def test_strict_mode_fails_closed_without_feature_screen_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                feature_screen_spec(missing, missing, missing, exploratory=False)

            spec = feature_screen_spec(missing, missing, missing, exploratory=True)
            self.assertIsNone(spec["report_sha256"])
            self.assertTrue(spec["integrity_issues"])

    def test_offline_fake_api_excludes_screen_subjects_and_writes_strict_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ranking = pd.DataFrame(
                {
                    "parameter": PARAMETERS,
                    "grouped_permutation_auprc_drop_mean": [5, 4, 3, 2, 1],
                }
            )
            ranking_path = root / "ranking.csv"
            ranking.to_csv(ranking_path, index=False)
            report_path = root / "feature-screen.json"
            report_path.write_text(
                json.dumps(
                    {"sampled_cases": 2, "usable_cases": 2, "usable_subjects": 2, "windows": 10}
                )
            )
            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "windows": 10,
                        "case_ids": [1, 2],
                        "subject_ids": [1001, 1002],
                        "source_table_name": "window_features.csv.gz",
                        "source_table_sha256": "offline-fixture",
                        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "ranking_sha256": hashlib.sha256(ranking_path.read_bytes()).hexdigest(),
                    }
                )
            )
            output = root / "onset_v2"
            args = parse_args(
                [
                    "--case-limit",
                    "8",
                    "--ranking-file",
                    str(ranking_path),
                    "--feature-screen-report",
                    str(report_path),
                    "--feature-screen-cohort",
                    str(cohort_path),
                    "--cache-dir",
                    str(root / "cache"),
                    "--output-dir",
                    str(output),
                ]
            )
            api = FakeVitalDB()

            result = prepare_corpus(
                args,
                api=api,
                case_loader=lambda case_id, interval, cache: frame_with_all_labels(),
                show_progress=False,
            )

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result["output_dir"], str(output))
            self.assertEqual(manifest["training_contract"], TRAINING_CONTRACT)
            self.assertEqual(manifest["target_version"], TARGET_VERSION)
            self.assertFalse(manifest["source_is_synthetic"])
            self.assertEqual(
                manifest["feature_screen_exclusion"]["excluded_subject_ids"],
                [1001, 1002],
            )
            self.assertEqual(api.clinical_calls, 1)
            self.assertEqual(len(api.find_calls), 2)
            corpus, audit = validate_corpus(output)
            self.assertEqual(corpus.manifest["training_contract"], TRAINING_CONTRACT)
            self.assertTrue(audit["ok"])

            saved_subjects: set[int] = set()
            for split in ("train", "validation", "test"):
                with np.load(output / f"{split}.npz", allow_pickle=False) as archive:
                    saved_subjects.update(archive["subject_id"].astype(int).tolist())
                    self.assertEqual(archive["x_values"].shape[1:], (10, 5))
            self.assertFalse(saved_subjects & {1001, 1002})


if __name__ == "__main__":
    unittest.main()
