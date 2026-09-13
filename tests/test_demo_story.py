import unittest
from pathlib import Path

from demo.runs import list_runs
from demo.story import build_story


class DemoStoryTest(unittest.TestCase):
    def test_story_includes_user_sourcing_and_limits(self) -> None:
        story = build_story(Path("."))
        self.assertTrue(story["limitations"])
        self.assertIn("reassess", story["how_it_helps"].lower())
        if story["sourcing"].get("dataset_id"):
            self.assertEqual(story["sourcing"]["dataset_id"], "vitaldb")
        self.assertIn("synthetic", story["opentslm_run"]["note"].lower())
        self.assertIn("models", story["held_out_vitaldb"])
        self.assertTrue(story["pitch"]["one_liner"])


class DemoRunsTest(unittest.TestCase):
    def test_three_pass_run_is_ready(self) -> None:
        from demo.runs import list_runs

        runs = {row["id"]: row for row in list_runs(Path("."))}
        self.assertTrue(runs["3-epochs"]["ready"])
        self.assertTrue(runs["50-epochs"]["ready"])
        self.assertTrue(runs["50-epochs"]["has_model"])
        self.assertGreater(runs["3-epochs"]["passes"], 0)
        self.assertEqual(runs["50-epochs"]["passes"], 50)


if __name__ == "__main__":
    unittest.main()
