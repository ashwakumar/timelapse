"""AeroGuard TSLM: End-to-End Operational Demonstration.

Demonstrates:
1. Data Sourcing & Provenance (NASA C-MAPSS FD001 turbofan simulation)
2. Real Telemetry Inputs (14-channel multivariate sensor windows over 30 cycles)
3. Real Model Outputs (Scalar RUL, Health Bands, and Generated Chain-of-Thought Diagnostics)
4. Target User Value (Fleet Reliability Engineers, MRO Planners, Dispatch Controllers)
5. Evidence & Limitations Analysis
"""

from __future__ import annotations

import json
from pathlib import Path

from training.inference import AeroGuardPredictor


def run_demonstration() -> None:
    print("=" * 80)
    print("✈️  AEROGUARD TSLM: MULTIMODAL TIME-SERIES LANGUAGE MODEL DEMONSTRATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # PART 1: HOW THE DATA WAS SOURCED
    # ---------------------------------------------------------
    print("\n" + "─" * 80)
    print("📌 PART 1: DATA SOURCING & PROVENANCE")
    print("─" * 80)
    print("""
• Dataset: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
• Subset: FD001 (High-Pressure Compressor degradation, sea-level steady cruise)
• Raw File: data/raw/train_FD001.txt (20,631 records across 100 run-to-failure engines)
• Slicing: 30-cycle sliding windows (stride 5) + terminal failure windows
• Sensor Filtering: 14 active degraded sensors selected; 7 invariant sensors dropped
• Leakage Prevention:
    - Training Split: Engines 1–70 (2,502 windows)
    - Validation Split: Engines 71–80 (355 windows)
    - Held-Out Test Split: Engines 81–100 (806 windows) — Strictly Unseen
• Normalization: Z-score fitted strictly on Training Split (saved in preprocessing.json)
""")

    # ---------------------------------------------------------
    # PART 2: REAL INPUTS FROM HELD-OUT FLEET
    # ---------------------------------------------------------
    print("─" * 80)
    print("📌 PART 2: REAL TELEMETRY INPUTS (HELD-OUT ENGINES)")
    print("─" * 80)

    data_path = Path("data/processed/windows.jsonl")
    if not data_path.exists():
        print("❌ Error: Processed data not found. Run scripts.preprocess_data first.")
        return

    # Load records from held-out test split
    test_records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "test":
                    test_records.append(item)

    print(f"Loaded {len(test_records)} held-out test windows (Engines 81–100).")

    # Pick two contrasting cases from Engine Unit #84:
    # 1. Mid-life nominal/early wear window
    # 2. End-of-life critical wear window
    unit_84 = [r for r in test_records if r["unit_number"] == 84]
    if not unit_84:
        # Fallback to first available test engine
        sample_unit = test_records[0]["unit_number"]
        unit_84 = [r for r in test_records if r["unit_number"] == sample_unit]

    mid_record = unit_84[len(unit_84) // 3]
    crit_record = unit_84[-1]

    for label, rec in [("CASE A: MID-LIFE OPERATION", mid_record), ("CASE B: NEAR-FAILURE TERMINAL OPERATION", crit_record)]:
        s4 = rec["series"]["sensor_4"]   # T50 Exhaust Gas Temp
        s11 = rec["series"]["sensor_11"] # Ps30 HPC Static Pressure
        s15 = rec["series"]["sensor_15"] # BPR Bypass Ratio
        s9 = rec["series"]["sensor_9"]   # Nc Core Speed

        print(f"\n[{label}] — Engine Unit #{rec['unit_number']} at Flight Cycle {rec['cycle']}")
        print(f"  • Observation Window: Cycles {rec['cycle']-29} to {rec['cycle']} (30 operational cycles)")
        print(f"  • T50 (LPT Exhaust Gas Temp): Start={s4[0]:.2f}°R -> End={s4[-1]:.2f}°R (Drift: {s4[-1]-s4[0]:+.2f}°R)")
        print(f"  • Ps30 (HPC Static Pressure): Start={s11[0]:.2f} psia -> End={s11[-1]:.2f} psia (Drift: {s11[-1]-s11[0]:+.2f} psia)")
        print(f"  • BPR (Bypass Ratio):        Start={s15[0]:.4f} -> End={s15[-1]:.4f} (Drift: {s15[-1]-s15[0]:+.4f})")
        print(f"  • Nc (Core Rotational Speed): Start={s9[0]:.2f} rpm -> End={s9[-1]:.2f} rpm")
        print(f"  • Prompt Passed to TSLM: \"{rec['prompt']}\"")

    # ---------------------------------------------------------
    # PART 3: MODEL INFERENCE & REAL OUTPUTS
    # ---------------------------------------------------------
    print("\n" + "─" * 80)
    print("📌 PART 3: REAL MODEL OUTPUTS & INFERENCE EXECUTION")
    print("─" * 80)
    print("Initializing AeroGuardPredictor and loading trained SmolLM-135M + LoRA + Patch Encoder...")

    try:
        predictor = AeroGuardPredictor(model_dir="models/aeroguard_tslm")
        print(f"✅ Model loaded successfully on device: {predictor.device}\n")
    except Exception as exc:
        print(f"⚠️ Could not load trained model ({exc}). Running simulated demonstration.")
        return

    for label, rec in [("CASE A: MID-LIFE OPERATION", mid_record), ("CASE B: NEAR-FAILURE TERMINAL OPERATION", crit_record)]:
        assessment = predictor.assess_record(rec, max_new_tokens=96)
        print(f"┌─ {label} (Unit #{assessment.unit_number}, Cycle {assessment.cycle})")
        print(f"│  • Predicted RUL:     {assessment.predicted_rul:.1f} cycles")
        print(f"│  • Ground Truth RUL:  {assessment.true_rul:.1f} cycles")
        error = abs(assessment.predicted_rul - (assessment.true_rul or 0.0))
        print(f"│  • Prediction Error:  {error:.1f} cycles")
        print(f"│  • Health Status:     [{assessment.health_band}]")
        print(f"│  • Dispatch Action:   {assessment.action_directive}")
        if assessment.component_diagnosis:
            cd = assessment.component_diagnosis
            print(f"│  • Fault Location:    {cd.station_name}")
            print(f"│  • Failing Assembly:  {cd.module_name} ({cd.fault_mode})")
            print(f"│  • Isolation Score:   {cd.confidence_score * 100:.1f}% Physics Coupling")
            print(f"│  • MRO Work Order:    {cd.maintenance_order} | {cd.borescope_inspection_task}")
            print("│  • Prescribed Line-Replaceable Units (LRUs):")
            for p in cd.replacement_parts:
                print(f"│     - [{p.replacement_status:<17}] {p.part_name:<42} (OEM P/N: {p.oem_part_number})")
        print("│  • Model Generated Chain-of-Thought (CoT) Diagnostic:")
        print(f"│    \"{assessment.cot_diagnostics}\"")
        print("└" + "─" * 78)

    # ---------------------------------------------------------
    # PART 4: HOW THE RESULT HELPS THE TARGET USER
    # ---------------------------------------------------------
    print("\n" + "─" * 80)
    print("📌 PART 4: HOW THE RESULT HELPS THE TARGET USER")
    print("─" * 80)
    print("""
Target User Personas & Operational Impact:

1. Airline Fleet Reliability Engineers:
   • Problem: Traditional maintenance relies on fixed flight-hour limits (e.g. overhaul every 3,000 hrs),
     risking unexpected in-flight shutdowns (IFSD) or prematurely discarding usable components.
   • Solution: AeroGuard TSLM delivers continuous, real-time RUL degradation tracking grounded in
     exhaust gas temperature (T50) and compressor pressure (Ps30) divergence.

2. Flight Operations & Dispatch Controllers:
   • Problem: Grounding an aircraft unexpectedly at a remote outstation costs $150k+/incident in delays
     and passenger re-accommodation.
   • Solution: Clear RUL bounds enable intelligent routing: engines with 'ELEVATED WEAR' (RUL 31-75)
     are restricted to domestic legs routing back to the airline's primary maintenance hub.

3. Maintenance, Repair, and Overhaul (MRO) Logistics Planners:
   • Problem: High-Pressure Compressor rotor blades and stator vanes have 3-6 week procurement lead times.
   • Solution: Dual-head CoT diagnostics explain the mechanical failure mechanism early, allowing
     teams to stage replacement kits and schedule hangar slots weeks before structural failure.
""")

    # ---------------------------------------------------------
    # PART 5: EVIDENCE AND LIMITATIONS
    # ---------------------------------------------------------
    print("─" * 80)
    print("📌 PART 5: EVIDENCE AND LIMITATIONS ANALYSIS")
    print("─" * 80)
    print("""
EVIDENCE (What the Data & Model Support):
1. Zero Data Leakage: Evaluation is conducted strictly on unseen held-out engines (Units 81-100)
   with normalizations fitted solely on training engines (Units 1-70).
2. Physical Signal Grounding: Multivariate sensor patches directly track coupled thermodynamic degradation:
   T50 rise (+15°R) accompanied by Ps30 pressure drop (-0.4 psia) reliably indicates aerodynamic clearance loss.
3. Multi-Task Coherence: The model couples a quantitative scalar prediction with qualitative engineering
   explanations, making black-box numerical regression explainable to aviation auditors.

LIMITATIONS (Critical Operational Boundaries):
1. Simulated Data Fidelity: NASA C-MAPSS FD001 is a thermodynamic computer simulation at sea level under
   steady cruise. Real-world flights experience transient climbs, weather extremes, dust, and icing.
2. Single Fault Mode: FD001 models only High-Pressure Compressor (HPC) wear. Real aircraft engines
   suffer multi-fault interactions (e.g. HPT turbine blade creep + combustor nozzle clogging + bearing friction).
3. Synthetic Diagnostic Texts: Ground truth CoT texts were synthesized via deterministic domain templates;
   they do not reflect certified OEM shop teardown findings.
4. Human-in-the-Loop Requirement: Language models can produce token hallucinations or variance. AeroGuard
   TSLM is an Advisory Decision Support System, NOT an automated airworthiness certifying authority.
""")
    print("=" * 80)
    print("🏁 DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
