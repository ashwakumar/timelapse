"""Fixture-based unit tests for NASA C-MAPSS TimeNet connector."""

import numpy as np
import pytest
from timenet.types import AnswerTask, ScalarPredictionTask

from aeroguard_connectors.cmapss.connector import SENSOR_DEFINITIONS, CMAPSSConnector


@pytest.fixture
def sample_raw_refs() -> list[dict]:
    """Provide deterministic offline sample records matching C-MAPSS schema."""
    channels = list(SENSOR_DEFINITIONS.keys())
    return [
        {
            "record_id": "FD001-U001-C0030",
            "unit_number": 1,
            "cycle": 30,
            "split": "train",
            "rul": 162,
            "status": "NORMAL",
            "window_start": 1,
            "label_provenance": {"rul": "ground_truth"},
            "series": {
                ch: (np.linspace(100.0, 105.0, 30) + i).tolist() for i, ch in enumerate(channels)
            },
            "prompt": "Analyze multi-channel turbofan sensor telemetry for Engine Unit 1.",
            "target": "Status: NORMAL. Projected RUL: 162 cycles. Action: Standard envelope.",
            "rationale": "Sensor drift within nominal bounds. Degradation indicates normal wear.",
        },
        {
            "record_id": "FD001-U085-C0140",
            "unit_number": 85,
            "cycle": 140,
            "split": "test",
            "rul": 25,
            "status": "CRITICAL",
            "window_start": 111,
            "series": {
                ch: (np.linspace(100.0, 115.0, 30) + i).tolist() for i, ch in enumerate(channels)
            },
            "prompt": "Analyze multi-channel turbofan sensor telemetry for Engine Unit 85.",
            "target": "Status: CRITICAL. Projected RUL: 25 cycles. Action: Immediate overhaul.",
            "rationale": "High drift in T50 and Ps30. Indicates severe thermal fatigue.",
        },
    ]


def test_connector_metadata():
    """Verify that dataset metadata card parses cleanly."""
    connector = CMAPSSConnector()
    meta = connector.metadata()
    assert meta.dataset_id == "nasa/cmapss"
    assert meta.dataset_version.major == 1
    assert "turbofan" in meta.tags
    assert str(meta.license) == "CC0-1.0"


def test_connector_convert(sample_raw_refs):
    """Verify offline conversion of raw records to TimeFDataset."""
    connector = CMAPSSConnector()
    dataset = connector.convert(sample_raw_refs)

    # Check dataset structure
    assert dataset.metadata.dataset_id == "nasa/cmapss"
    assert len(dataset.records) == 2

    # Check Record 0
    rec0 = dataset.records[0]
    assert rec0.record_id == "FD001-U001-C0030"
    assert len(rec0.time_series) == 14

    # Check annotations
    anno_dict = {a.key: a.value for a in rec0.annotations}
    assert anno_dict["unit_number"] == "1"
    assert anno_dict["split"] == "train"
    assert anno_dict["cycle"] == "30"
    assert anno_dict["status"] == "NORMAL"
    assert anno_dict["window_size"] == "30"
    assert anno_dict["window_start"] == "1"
    assert "label_provenance" in anno_dict

    # Check tasks on dataset
    tasks = dataset.tasks
    assert len(tasks) == 4  # 2 records x 2 tasks (QA + RUL)

    qa_tasks = [t for t in tasks if isinstance(t, AnswerTask)]
    assert len(qa_tasks) == 2
    assert qa_tasks[0].prompt is not None
    assert "Engine Unit 1" in qa_tasks[0].prompt
    assert qa_tasks[0].rationale is not None

    rul_tasks = [t for t in tasks if isinstance(t, ScalarPredictionTask)]
    assert len(rul_tasks) == 2
    assert rul_tasks[0].target == 162.0
    assert rul_tasks[0].target_name == "RUL"

    # Check Record 1 (held-out test unit)
    rec1 = dataset.records[1]
    assert rec1.record_id == "FD001-U085-C0140"
    anno_dict1 = {a.key: a.value for a in rec1.annotations}
    assert anno_dict1["unit_number"] == "85"
    assert anno_dict1["split"] == "test"
