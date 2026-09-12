"""AeroGuard Component Fault Isolation & Maintenance Prescription Engine.

Maps multivariate thermodynamic sensor signatures from NASA C-MAPSS FD001
to physical turbofan engine stations, specific mechanical failure mechanisms,
Line-Replaceable Units (LRUs), and MRO work orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplacementPart:
    """Line-Replaceable Unit (LRU) or overhaul component."""

    part_name: str
    oem_part_number: str
    subsystem: str
    replacement_status: str  # "NOMINAL", "STAGE_KIT", "IMMEDIATE_REPLACE"
    standard_task_card: str
    estimated_lead_time: str


@dataclass(frozen=True)
class ComponentFaultDiagnosis:
    """Physics-informed component isolation and MRO prescription."""

    station_id: str
    station_name: str
    module_name: str
    fault_mode: str
    degradation_mechanism: str
    confidence_score: float
    replacement_urgency: str  # "NOMINAL", "PREVENTIVE_INSPECTION", "AOG_CRITICAL"
    maintenance_order: str
    borescope_inspection_task: str
    action_summary: str
    thermodynamic_evidence: list[str] = field(default_factory=list)
    replacement_parts: list[ReplacementPart] = field(default_factory=list)


def isolate_component_fault(
    raw_series: dict[str, list[float]],
    predicted_rul: float,
) -> ComponentFaultDiagnosis:
    """Isolate degraded turbofan module and prescribe component replacements.

    In NASA C-MAPSS FD001, degradation is governed by High-Pressure Compressor (HPC)
    flow capacity loss and adiabatic efficiency degradation. As blade tip clearances
    widen and aerodynamic profiles erode, the engine demonstrates a distinct thermodynamic
    divergence:
      - Station 50 (T50 / EGT): Rises as combustor burns excess fuel to maintain thrust.
      - Station 30 (Ps30 / P30): Drops due to reduced compression stage pressure rise.
      - Station 15 (BPR): Shifts due to core flow restriction.
      - Station 9 (Nc): Core speed adjusts to balance mechanical shaft load.

    Args:
        raw_series: Mapping of sensor names to lists of historical readings.
        predicted_rul: Estimated Remaining Useful Life in operational flight cycles.

    Returns:
        Structured ComponentFaultDiagnosis with parts replacement bill of materials.
    """
    # Calculate drift metrics (last minus first across 30-cycle window)
    t50_drift = 0.0
    ps30_drift = 0.0
    bpr_drift = 0.0
    nc_drift = 0.0
    t30_drift = 0.0

    if "sensor_4" in raw_series and len(raw_series["sensor_4"]) > 1:
        t50_drift = float(raw_series["sensor_4"][-1] - raw_series["sensor_4"][0])
    if "sensor_11" in raw_series and len(raw_series["sensor_11"]) > 1:
        ps30_drift = float(raw_series["sensor_11"][-1] - raw_series["sensor_11"][0])
    if "sensor_15" in raw_series and len(raw_series["sensor_15"]) > 1:
        bpr_drift = float(raw_series["sensor_15"][-1] - raw_series["sensor_15"][0])
    if "sensor_9" in raw_series and len(raw_series["sensor_9"]) > 1:
        nc_drift = float(raw_series["sensor_9"][-1] - raw_series["sensor_9"][0])
    if "sensor_3" in raw_series and len(raw_series["sensor_3"]) > 1:
        t30_drift = float(raw_series["sensor_3"][-1] - raw_series["sensor_3"][0])

    evidence: list[str] = []
    if t50_drift > 1.0:
        evidence.append(f"Station 50 (LPT EGT): Thermal runaway drift of {t50_drift:+.2f}°R")
    elif t50_drift < -1.0:
        evidence.append(f"Station 50 (LPT EGT): Stable thermal profile ({t50_drift:+.2f}°R drift)")
    else:
        evidence.append(f"Station 50 (LPT EGT): Operating within nominal thermal variance ({t50_drift:+.2f}°R)")

    if ps30_drift < -0.1:
        evidence.append(f"Station 30 (HPC Ps30): Static pressure deficit of {ps30_drift:+.2f} psia")
    else:
        evidence.append(f"Station 30 (HPC Ps30): Compression pressure ratio stable ({ps30_drift:+.2f} psia)")

    if abs(bpr_drift) > 0.01:
        evidence.append(f"Station 15 (BPR): Core flow ratio shift of {bpr_drift:+.4f}")
    if abs(nc_drift) > 5.0:
        evidence.append(f"Station 9 (Nc): Core shaft compensation speed delta {nc_drift:+.1f} rpm")
    if t30_drift > 0.5:
        evidence.append(f"Station 30 (HPC T30): Compressor exit thermal rise of {t30_drift:+.2f}°R")

    # Determine confidence in HPC fault isolation from physical divergence
    # Strong divergence: T50 up AND Ps30 down
    is_diverging = (t50_drift > 2.0) and (ps30_drift < -0.15)
    base_confidence = 0.94 if is_diverging else (0.88 if (t50_drift > 0.5 or ps30_drift < -0.05) else 0.75)

    # 1. CRITICAL WEAR (RUL <= 30 cycles)
    if predicted_rul <= 30.0:
        urgency = "AOG_CRITICAL"
        order_num = f"WO-CFM56-HPC-{max(100, int(predicted_rul * 10))}"
        action_text = (
            "GROUND ENGINE IMMEDIATELY (AOG). High-Pressure Compressor blade erosion exceeds "
            "allowable limits per AMM 72-31-00. Remove engine for shop overhaul and replace "
            "Stage 2-5 rotor blades, variable stator vanes, and interstage honeycomb seals."
        )
        parts = [
            ReplacementPart(
                part_name="HPC Stage 2–5 Rotor Blade Assemblies",
                oem_part_number="CFM56-HPC-RB25",
                subsystem="Station 30 Core Spool",
                replacement_status="IMMEDIATE_REPLACE",
                standard_task_card="AMM 72-31-00 Task 201",
                estimated_lead_time="Stock (AOG Priority Release)",
            ),
            ReplacementPart(
                part_name="Variable Stator Vane (VSV) Bushings & Linkages",
                oem_part_number="CFM56-VSV-KIT",
                subsystem="HPC Stator Casing",
                replacement_status="IMMEDIATE_REPLACE",
                standard_task_card="AMM 72-31-05 Task 104",
                estimated_lead_time="Stock (24h Expedited)",
            ),
            ReplacementPart(
                part_name="Interstage Honeycomb Air Seals (Stage 3–4)",
                oem_part_number="CFM56-HPC-SL30",
                subsystem="HPC Rotor Assembly",
                replacement_status="IMMEDIATE_REPLACE",
                standard_task_card="AMM 72-31-12 Task 302",
                estimated_lead_time="Stock (Immediate)",
            ),
            ReplacementPart(
                part_name="HPC Forward High-Pressure Seal Assembly",
                oem_part_number="CFM56-HPC-FS40",
                subsystem="Station 25/30 Interface",
                replacement_status="STAGE_KIT",
                standard_task_card="AMM 72-31-20 Task 110",
                estimated_lead_time="48 Hours",
            ),
        ]

    # 2. ELEVATED WEAR (30 < RUL <= 75 cycles)
    elif predicted_rul <= 75.0:
        urgency = "PREVENTIVE_INSPECTION"
        order_num = f"WO-INSP-HPC-{max(200, int(predicted_rul * 5))}"
        target_cycle = max(1, int(predicted_rul - 10))
        action_text = (
            f"RESTRICT TO DOMESTIC ROUTES. Schedule Borescope Inspection within {target_cycle} cycles "
            "per AMM 72-31-00. Pre-order HPC Stage 2–5 overhaul kit to primary hub to minimize aircraft downtime."
        )
        parts = [
            ReplacementPart(
                part_name="HPC Stage 2–5 Rotor Blade Assemblies",
                oem_part_number="CFM56-HPC-RB25",
                subsystem="Station 30 Core Spool",
                replacement_status="STAGE_KIT",
                standard_task_card="AMM 72-31-00 Task 201",
                estimated_lead_time="Pre-order / 2–3 Weeks Lead",
            ),
            ReplacementPart(
                part_name="Variable Stator Vane (VSV) Bushings & Linkages",
                oem_part_number="CFM56-VSV-KIT",
                subsystem="HPC Stator Casing",
                replacement_status="STAGE_KIT",
                standard_task_card="AMM 72-31-05 Task 104",
                estimated_lead_time="Standard Spares Stock",
            ),
            ReplacementPart(
                part_name="Interstage Honeycomb Air Seals (Stage 3–4)",
                oem_part_number="CFM56-HPC-SL30",
                subsystem="HPC Rotor Assembly",
                replacement_status="STAGE_KIT",
                standard_task_card="AMM 72-31-12 Task 302",
                estimated_lead_time="Standard Spares Stock",
            ),
            ReplacementPart(
                part_name="HPC Forward High-Pressure Seal Assembly",
                oem_part_number="CFM56-HPC-FS40",
                subsystem="Station 25/30 Interface",
                replacement_status="NOMINAL",
                standard_task_card="AMM 72-31-20 Task 110",
                estimated_lead_time="Routine Spares Pool",
            ),
        ]

    # 3. NOMINAL ENVELOPE (RUL > 75 cycles)
    else:
        urgency = "NOMINAL"
        order_num = "ROUTINE-LINE-05"
        action_text = (
            "ALL COMPONENTS NOMINAL. Aerodynamic clearances and thermal margins within certified "
            "operational limits. Continue scheduled line maintenance per AMM 05-20-00."
        )
        parts = [
            ReplacementPart(
                part_name="HPC Stage 2–5 Rotor Blade Assemblies",
                oem_part_number="CFM56-HPC-RB25",
                subsystem="Station 30 Core Spool",
                replacement_status="NOMINAL",
                standard_task_card="AMM 72-31-00 Task 201",
                estimated_lead_time="Routine Spares Pool",
            ),
            ReplacementPart(
                part_name="Variable Stator Vane (VSV) Bushings & Linkages",
                oem_part_number="CFM56-VSV-KIT",
                subsystem="HPC Stator Casing",
                replacement_status="NOMINAL",
                standard_task_card="AMM 72-31-05 Task 104",
                estimated_lead_time="Routine Spares Pool",
            ),
            ReplacementPart(
                part_name="Interstage Honeycomb Air Seals (Stage 3–4)",
                oem_part_number="CFM56-HPC-SL30",
                subsystem="HPC Rotor Assembly",
                replacement_status="NOMINAL",
                standard_task_card="AMM 72-31-12 Task 302",
                estimated_lead_time="Routine Spares Pool",
            ),
            ReplacementPart(
                part_name="HPC Forward High-Pressure Seal Assembly",
                oem_part_number="CFM56-HPC-FS40",
                subsystem="Station 25/30 Interface",
                replacement_status="NOMINAL",
                standard_task_card="AMM 72-31-20 Task 110",
                estimated_lead_time="Routine Spares Pool",
            ),
        ]

    return ComponentFaultDiagnosis(
        station_id="Station 30",
        station_name="Station 30: High-Pressure Compressor Exit & Core Spool",
        module_name="High-Pressure Compressor (HPC)",
        fault_mode="Aerodynamic Blade Erosion & Tip Clearance Widening",
        degradation_mechanism=(
            "Compressor blade tip clearance widening and airfoil surface erosion reduce "
            "isentropic adiabatic compression efficiency and flow capacity, forcing the "
            "combustor and turbine to operate at elevated temperatures (T50 runaway) to sustain thrust."
        ),
        confidence_score=base_confidence,
        replacement_urgency=urgency,
        maintenance_order=order_num,
        borescope_inspection_task="AMM 72-31-00 (HPC Borescope Inspection Procedure)",
        action_summary=action_text,
        thermodynamic_evidence=evidence,
        replacement_parts=parts,
    )
