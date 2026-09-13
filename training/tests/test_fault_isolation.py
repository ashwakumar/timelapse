"""Component observations must not imply verified faults or maintenance orders."""

import pytest

from training.fault_isolation import isolate_component_fault


@pytest.mark.parametrize(
    "rul,urgency",
    [
        (30, "PRIORITY_REVIEW"),
        (30.1, "PLANNING_REVIEW"),
        (75, "PLANNING_REVIEW"),
        (75.1, "MONITOR"),
    ],
)
def test_review_bands_do_not_invent_maintenance_values(rul, urgency):
    result = isolate_component_fault({"sensor_4": [100, 103], "sensor_11": [30, 29]}, rul)
    assert result.replacement_urgency == urgency
    assert result.confidence_score is None
    assert result.replacement_parts == []
    assert result.maintenance_order == result.borescope_inspection_task == ""
    assert result.fault_mode == "Not determined by this rule"
    assert result.sensor_rule_match == "Both thresholds met"
    assert "+3.0000" in result.thermodynamic_evidence[0]


def test_missing_or_invalid_readings_are_not_normal():
    result = isolate_component_fault({"sensor_4": [1, float("nan")]}, 100)
    assert result.sensor_rule_match == "Insufficient data"
    assert "insufficient valid readings" in result.thermodynamic_evidence[0]


def test_flat_signals_do_not_match_rule():
    result = isolate_component_fault({"sensor_4": [1, 1], "sensor_11": [2, 2]}, 20)
    assert result.sensor_rule_match == "Combined thresholds not met"
    assert result.confidence_score is None


@pytest.mark.parametrize("rul", [float("nan"), float("inf"), -1])
def test_invalid_rul_rejected(rul):
    with pytest.raises(ValueError):
        isolate_component_fault({}, rul)
