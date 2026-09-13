"""Descriptive sensor observations and illustrative RUL-based review suggestions.

These rules do not identify failed parts, estimate fault probabilities, or verify
maintenance requirements. HPC is the scenario context, not a diagnosed component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class ReplacementPart:
    """Compatibility structure; no verified replacement catalog is connected."""

    part_name: str
    oem_part_number: str
    subsystem: str
    replacement_status: str
    standard_task_card: str
    estimated_lead_time: str


@dataclass(frozen=True)
class ComponentFaultDiagnosis:
    """Rule observations, with unsupported maintenance fields left empty."""

    station_id: str
    station_name: str
    module_name: str
    fault_mode: str
    degradation_mechanism: str
    confidence_score: float | None
    replacement_urgency: str
    maintenance_order: str
    borescope_inspection_task: str
    action_summary: str
    thermodynamic_evidence: list[str] = field(default_factory=list)
    replacement_parts: list[ReplacementPart] = field(default_factory=list)
    sensor_rule_match: str = "Insufficient data"


def isolate_component_fault(
    raw_series: dict[str, list[float]], predicted_rul: float
) -> ComponentFaultDiagnosis:
    """Describe endpoint changes and select a review suggestion from predicted RUL."""
    if not isfinite(predicted_rul) or predicted_rul < 0:
        raise ValueError("A finite, nonnegative predicted RUL is required")
    drifts = {}
    evidence = []
    for sensor, label, unit in [
        ("sensor_4", "T50 · exhaust temperature", "°R"),
        ("sensor_11", "Ps30 · compressor static pressure", "psia"),
        ("sensor_15", "BPR · bypass ratio", ""),
        ("sensor_9", "Nc · core speed", "rpm"),
        ("sensor_3", "T30 · compressor exit temperature", "°R"),
    ]:
        values = raw_series.get(sensor, [])
        if len(values) < 2 or not all(isfinite(float(v)) for v in values):
            evidence.append(f"{label}: insufficient valid readings")
            continue
        drift = float(values[-1]) - float(values[0])
        drifts[sensor] = drift
        evidence.append(f"{label}: {drift:+.4f} {unit} (last minus first reading)")
    if "sensor_4" not in drifts or "sensor_11" not in drifts:
        match = "Insufficient data"
    elif drifts["sensor_4"] > 2.0 and drifts["sensor_11"] < -0.15:
        match = "Both thresholds met"
    else:
        match = "Combined thresholds not met"

    if predicted_rul <= 30:
        urgency = "PRIORITY_REVIEW"
        action = "Prioritize review of the low RUL estimate and sensor history with a maintenance specialist."
    elif predicted_rul <= 75:
        urgency = "PLANNING_REVIEW"
        action = "Review the RUL estimate and sensor history when planning further inspection."
    else:
        urgency = "MONITOR"
        action = "Continue tracking RUL estimates and sensor changes. This band does not establish that all components are healthy."
    return ComponentFaultDiagnosis(
        station_id="Station 30",
        station_name="Station 30 · High-pressure compressor (scenario reference)",
        module_name="High-pressure compressor (HPC) · scenario context",
        fault_mode="Not determined by this rule",
        degradation_mechanism="The demonstration uses an HPC degradation scenario. Endpoint sensor changes do not establish blade erosion, clearance loss, or the location of a failed part.",
        confidence_score=None,
        replacement_urgency=urgency,
        maintenance_order="",
        borescope_inspection_task="",
        action_summary=action,
        thermodynamic_evidence=evidence,
        sensor_rule_match=match,
    )
