import json
import tempfile
import unittest
from pathlib import Path

from connectors.registry import SOURCED_DATASET_CONNECTORS, load_selection
from connectors.vitaldb.hypotension_windows.targets import ANSWER_TEXT, QA_PROMPT


class ChallengeBridgeTest(unittest.TestCase):
    def test_vitaldb_is_registered_for_the_sourced_selection(self) -> None:
        self.assertIn("vitaldb", SOURCED_DATASET_CONNECTORS)
        selection = load_selection(Path("artifacts/data_sourcing/selection.json"))
        self.assertEqual(selection["dataset_id"], "vitaldb")
        self.assertTrue(selection["validation_ok"])
        self.assertIn("Anesthesiologist", selection["target_user"]["role"])

    def test_missing_challenge_1_selection_is_a_clear_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as raised:
            load_selection(Path("/tmp/does-not-exist-selection.json"))
        self.assertIn("source_datasets.py", str(raised.exception))

    def test_unregistered_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            path.write_text(
                json.dumps({"dataset_id": "wesad", "validation_ok": True})
            )
            with self.assertRaises(RuntimeError) as raised:
                load_selection(path)
            self.assertIn("wesad", str(raised.exception))

    def test_task_card_documents_the_same_labels_as_the_connector(self) -> None:
        card = Path("connectors/vitaldb/hypotension_windows/tasks.yaml").read_text()
        for label, answer in ANSWER_TEXT.items():
            self.assertIn(label, card)
            self.assertIn(answer, card)
        self.assertIn("hypotension_onset_horizon_v1", card)
        self.assertIn("sustained MAP below 65", QA_PROMPT)


if __name__ == "__main__":
    unittest.main()
