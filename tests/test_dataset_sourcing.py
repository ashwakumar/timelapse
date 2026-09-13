import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError

from sourcing.assess import assess_candidate, rank_assessments
from sourcing.catalog import DatasetCandidate, seed_candidates
from sourcing.pipeline import DataSourcingAgent, run_pipeline
from sourcing.problem import PROBLEM
from sourcing.retrieve import validate_candidate


def _by_id(dataset_id: str) -> DatasetCandidate:
    return next(item for item in seed_candidates() if item.dataset_id == dataset_id)


class ProblemCardTest(unittest.TestCase):
    def test_names_a_user_a_decision_and_search_queries(self) -> None:
        self.assertIn("Anesthesiologist", PROBLEM.target_user.role)
        self.assertIn("operating room", PROBLEM.target_user.setting.lower())
        self.assertIn("hypotension", PROBLEM.problem.lower())
        self.assertGreaterEqual(len(PROBLEM.search_queries), 3)
        self.assertIn("vitaldb", PROBLEM.hub_search_queries)
        self.assertTrue(PROBLEM.constraints.must_include_arterial_pressure)


class AssessmentTest(unittest.TestCase):
    def test_vitaldb_outranks_icu_wearable_and_industrial_alternatives(self) -> None:
        ranked = rank_assessments(
            [assess_candidate(candidate, PROBLEM) for candidate in seed_candidates()]
        )
        self.assertEqual(ranked[0].candidate.dataset_id, "vitaldb")
        self.assertTrue(ranked[0].hard_pass)
        wesad = next(item for item in ranked if item.candidate.dataset_id == "wesad")
        hirid = next(item for item in ranked if item.candidate.dataset_id == "hirid")
        pulsedb = next(item for item in ranked if item.candidate.dataset_id == "pulsedb")
        cmapss = next(item for item in ranked if item.candidate.dataset_id == "cmapss")
        self.assertFalse(wesad.hard_pass)
        self.assertIn("arterial_pressure", wesad.rejection_reasons)
        self.assertFalse(hirid.hard_pass)
        self.assertIn("uncredentialed", hirid.rejection_reasons)
        self.assertFalse(pulsedb.hard_pass)
        self.assertIn("intraoperative_domain", pulsedb.rejection_reasons)
        self.assertFalse(cmapss.hard_pass)


class RetrievalTest(unittest.TestCase):
    def test_vitaldb_validation_reads_case_and_subject_header(self) -> None:
        def fetcher(url: str) -> tuple[int, bytes]:
            if url.endswith("/cases"):
                return 200, b"caseid,subjectid,age,sex\n1,10,60,M\n"
            return 200, b"ok"

        report = validate_candidate(_by_id("vitaldb"), fetcher=fetcher)
        self.assertTrue(report.ok)
        names = [check.name for check in report.checks]
        self.assertIn("vitaldb_cases_table", names)

    def test_unreachable_selected_source_fails_validation(self) -> None:
        def fetcher(_url: str) -> tuple[int, bytes]:
            raise URLError("offline")

        report = validate_candidate(_by_id("vitaldb"), fetcher=fetcher)
        self.assertFalse(report.ok)


class AgentLoopTest(unittest.TestCase):
    def test_excludes_failed_retrieval_and_selects_next_hard_pass(self) -> None:
        fallback = DatasetCandidate(
            dataset_id="or-telemetry-mirror",
            name="OR telemetry mirror",
            source="test",
            landing_url="https://example.test/or-telemetry",
            access_url="https://example.test/or-telemetry",
            license="CC BY-NC-SA 4.0",
            domain="intraoperative",
            description="Second intraoperative source used only to test retry.",
            signals=("MAP", "HR", "SpO2", "EtCO2"),
            has_arterial_pressure=True,
            has_subject_ids=True,
            sampling_hz=1.0,
            programmatic_access=True,
            credentialed=False,
            time_series=True,
        )

        def fetcher(url: str) -> tuple[int, bytes]:
            if "vitaldb" in url:
                raise URLError("vitaldb blocked in test")
            return 200, b"ok"

        report = DataSourcingAgent(
            include_huggingface=False,
            url_fetcher=fetcher,
            candidates=[_by_id("vitaldb"), fallback],
        ).run()
        self.assertEqual(report.selected["dataset_id"], "or-telemetry-mirror")
        self.assertGreaterEqual(len(report.rounds), 2)
        self.assertFalse(report.rounds[0]["validation_ok"])
        self.assertTrue(report.rounds[1]["validation_ok"])

    def test_pipeline_writes_selection_when_vitaldb_validates(self) -> None:
        def json_fetcher(_url: str) -> object:
            return [
                {
                    "id": "demo/wearable-stress",
                    "license": "mit",
                    "tags": ["time-series"],
                    "description": "wrist sensors for stress",
                    "gated": False,
                }
            ]

        def url_fetcher(url: str) -> tuple[int, bytes]:
            if url.endswith("/cases"):
                return 200, b"caseid,subjectid,age\n1,9,40\n"
            return 200, b"ok"

        with TemporaryDirectory() as temp:
            output = Path(temp) / "sourcing"
            report = run_pipeline(
                output,
                include_huggingface=True,
                json_fetcher=json_fetcher,
                url_fetcher=url_fetcher,
            )
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(report.selected["dataset_id"], "vitaldb")
            self.assertEqual(selection["dataset_id"], "vitaldb")
            self.assertTrue(selection["validation_ok"])
            self.assertIn("Anesthesiologist", selection["target_user"]["role"])
            assessed_ids = {item["dataset_id"] for item in report.assessments}
            self.assertIn("hf:demo/wearable-stress", assessed_ids)


if __name__ == "__main__":
    unittest.main()
