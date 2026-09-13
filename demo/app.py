"""AeroGuard engine diagnostics, component details, route simulation, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from training.evaluate_baselines import extract_tabular_features
from training.fault_isolation import ComponentFaultDiagnosis, isolate_component_fault
from training.inference import AeroGuardPredictor

st.set_page_config(
    page_title="AeroGuard | Engine diagnostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Shared graphite palette; native widgets use .streamlit/config.toml.
st.markdown(
    """
<style>
    .stApp { background: #111418; color: #E6E9ED; }
    .block-container { max-width: 1480px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { letter-spacing: -0.04em; }
    h2, h3, h4 { letter-spacing: -0.02em; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 10px; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    .stButton > button { border-radius: 7px; font-weight: 600; box-shadow: none; }
    .stButton > button[kind="primary"] { background: #5AB8AC; color: #111418; border: 0; }
    .stButton > button[kind="primary"]:hover { background: #80CABE; color: #111418; }
    .status-badge-critical, .status-badge-warning, .status-badge-normal,
    .part-pill-critical, .part-pill-warning, .part-pill-nominal {
        display: inline-block; padding: 5px 10px; border-radius: 6px;
        font-size: 0.85rem; font-weight: 600; border: 1px solid;
    }
    .status-badge-critical, .part-pill-critical {
        background: #322126; color: #F0A4AA; border-color: #71424A;
    }
    .status-badge-warning, .part-pill-warning {
        background: #30291E; color: #E7C281; border-color: #695535;
    }
    .status-badge-normal, .part-pill-nominal {
        background: #1D302A; color: #93C9AA; border-color: #3B6050;
    }
    .limitation-card, .evidence-card, .benchmark-card, .fault-card {
        background: #1B2026; border: 1px solid #30373F;
        border-radius: 8px; padding: 18px; margin-bottom: 12px; line-height: 1.6;
    }
    .route-cleared, .route-restricted {
        border-radius: 8px; padding: 18px; line-height: 1.7; border: 1px solid;
    }
    .status-badge-unavailable {
        display: inline-block; padding: 5px 10px; border-radius: 6px;
        background: #242C33; color: #B4BDC7; border: 1px solid #46515D;
    }
    .station-legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
    .station-legend span { padding: 5px 10px; border: 1px solid currentColor; border-radius: 6px; font-size: 0.85rem; }
    .route-cleared { background: #1D302A; color: #93C9AA; border-color: #3B6050; }
    .route-restricted { background: #322126; color: #F0A4AA; border-color: #71424A; }
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
        h1 { font-size: 2rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_processed_windows() -> list[dict[str, Any]]:
    """Load pre-processed C-MAPSS windowed telemetry."""
    data_path = Path("data/processed/windows.jsonl")
    if not data_path.exists():
        st.error("Processed data missing. Run: `uv run python -m scripts.preprocess_data`")
        return []

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@st.cache_resource
def get_predictor() -> AeroGuardPredictor | None:
    """Load and cache the trained AeroGuard TSLM model for inference."""
    try:
        return AeroGuardPredictor(model_dir="models/aeroguard_tslm")
    except Exception:
        return None


@st.cache_resource
def get_xgboost_model(records: list[dict[str, Any]]) -> Any:
    """Train and cache the classical XGBoost baseline on training records."""
    train_records = [r for r in records if r.get("split") == "train"]
    if not train_records:
        return None
    X_train, y_train = extract_tabular_features(train_records)
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        tree_method="hist",
    )
    try:
        model.fit(X_train, y_train)
    except Exception:
        return None
    return model


records = load_processed_windows()
predictor = get_predictor()
xgb_baseline = get_xgboost_model(records)

# Filter to held-out test units (Engines 81-100)
test_records = [r for r in records if r["split"] == "test"]
test_unit_ids = sorted(list(set(r["unit_number"] for r in test_records))) if test_records else [84]

# ================= CONTEXT & CONTROLS =================
if not test_records:
    st.info("No test windows available. Prepare the dataset to inspect an engine.")
    st.stop()

schematic_path = Path("demo/assets/turbofan_schematic.svg")
st.title("AeroGuard")
st.caption("Engine diagnostics · NASA C-MAPSS demonstration")

engine_col, cycle_col, model_col, run_col = st.columns([2, 3, 3, 2], vertical_alignment="bottom")
with engine_col:
    selected_unit = st.selectbox(
        "Engine",
        test_unit_ids,
        format_func=lambda unit: f"Unit {unit:03d}",
        index=min(3, len(test_unit_ids) - 1),
    )
unit_records = [r for r in test_records if r["unit_number"] == selected_unit]
available_cycles = sorted({r["cycle"] for r in unit_records})
with cycle_col:
    if len(available_cycles) > 1:
        selected_cycle = st.select_slider(
            "Inspection cycle",
            options=available_cycles,
            value=available_cycles[max(0, len(available_cycles) - 4)],
            key=f"cycle_{selected_unit}",
        )
    else:
        selected_cycle = available_cycles[0]
        st.metric("Inspection cycle", selected_cycle)
with model_col:
    selected_model_type = st.selectbox(
        "Model",
        [
            "AeroGuard TSLM",
            "Amazon Chronos",
            "Classical ML (XGBoost)",
            "Text-Only LLM",
            "Static Schedule",
        ],
    )
with run_col:
    model_available = (
        (selected_model_type == "AeroGuard TSLM" and predictor is not None)
        or (selected_model_type == "Classical ML (XGBoost)" and xgb_baseline is not None)
        or selected_model_type == "Static Schedule"
    )
    run_btn = st.button(
        "Run assessment", type="primary", width="stretch", disabled=not model_available
    )

active_record = next(r for r in unit_records if r["cycle"] == selected_cycle)
with st.expander("Advanced"):
    settings_col, data_col = st.columns(2)
    with settings_col:
        st.caption("Model resources")
        st.caption("Checkpoint loaded" if predictor is not None else "Model unavailable")
        if st.button("Reload model checkpoint"):
            st.cache_resource.clear()
            st.session_state.pop("assessment_result", None)
            st.rerun()
    with data_col:
        st.caption("Dataset: NASA C-MAPSS FD001 · 14 sensor channels · 30-cycle window")
        st.caption("Evaluation units: 81–100 · Training units: 1–70")
        st.caption("Normalization: preprocessing.json")


def unavailable_assessment(model_choice: str, record: dict[str, Any]) -> dict[str, Any]:
    """Represent missing or failed inference without inventing a prediction."""
    return {
        "estimate_source": "Model unavailable",
        "pred_rul": None,
        "true_rul": float(record.get("rul", 0.0)),
        "abs_error": None,
        "health_band": "Model unavailable",
        "action_directive": "",
        "cot_diagnostics": "Model unavailable. Load a trained model to generate an assessment.",
        "rationale_source": "Model unavailable",
        "component_status": "UNAVAILABLE",
        "component_diagnosis": None,
        "component_unsupported_reason": "Model unavailable. Component diagnosis requires a successful prediction.",
        "arch_title": model_choice,
    }


def compute_assessment(
    model_choice: str,
    record: dict[str, Any],
    pred_engine: AeroGuardPredictor | None,
    xgb_eng: Any,
) -> dict[str, Any]:
    """Use real inference or the explicit schedule rule; never reference-label fallbacks."""
    result = unavailable_assessment(model_choice, record)
    try:
        if model_choice == "AeroGuard TSLM" and pred_engine is not None:
            prediction = float(pred_engine.predict_rul(record["series"]))
            rationale = "Run assessment to generate a diagnostic explanation."
        elif model_choice == "Classical ML (XGBoost)" and xgb_eng is not None:
            features, _ = extract_tabular_features([record])
            prediction = float(xgb_eng.predict(features)[0])
            rationale = "XGBoost estimates remaining life from sensor summary statistics. Component diagnosis is not supported."
        elif model_choice == "Static Schedule":
            prediction = float(max(0, 120 - (int(record["cycle"]) % 120)))
            rationale = "Cycles until the next fixed 120-cycle service interval. This rule does not analyze sensor health."
        else:
            return result
        if not np.isfinite(prediction):
            return result
        prediction = max(0.0, prediction)
        diagnosis = (
            isolate_component_fault(record.get("series", {}), prediction)
            if model_choice == "AeroGuard TSLM"
            else None
        )
    except Exception:
        return result

    result.update(
        pred_rul=round(prediction, 1),
        abs_error=round(abs(prediction - result["true_rul"]), 1),
        estimate_source="Fixed schedule estimate"
        if model_choice == "Static Schedule"
        else "Model prediction",
        cot_diagnostics=rationale,
        rationale_source="Schedule rule" if model_choice == "Static Schedule" else "Assessment",
        component_status="SUPPORTED" if diagnosis else "UNSUPPORTED",
        component_diagnosis=diagnosis,
        component_unsupported_reason="This model provides a scalar estimate without component diagnosis.",
    )
    return result


# Keep all views on the same assessment when route or chart controls rerun the app.
assessment_key = (2, selected_unit, selected_cycle, selected_model_type)
saved = st.session_state.get("assessment_result")
if saved is None or saved["key"] != assessment_key or run_btn:
    active_eval = compute_assessment(selected_model_type, active_record, predictor, xgb_baseline)
    if run_btn and selected_model_type == "AeroGuard TSLM" and active_eval["pred_rul"] is not None:
        with st.spinner("Generating assessment…"):
            try:
                assessment = predictor.assess_record(active_record, max_new_tokens=96)
                if not np.isfinite(assessment.predicted_rul):
                    raise ValueError("Invalid model prediction")
                active_eval.update(
                    pred_rul=round(assessment.predicted_rul, 1),
                    abs_error=round(abs(assessment.predicted_rul - active_eval["true_rul"]), 1),
                    action_directive=assessment.action_directive,
                    cot_diagnostics=assessment.cot_diagnostics,
                    estimate_source="Model prediction",
                    rationale_source="Generated assessment",
                    component_diagnosis=assessment.component_diagnosis
                    or isolate_component_fault(active_record["series"], assessment.predicted_rul),
                )
            except Exception:
                active_eval = unavailable_assessment(selected_model_type, active_record)
    st.session_state["assessment_result"] = {"key": assessment_key, "value": active_eval}
else:
    active_eval = saved["value"]

true_rul = active_eval["true_rul"]
pred_rul = active_eval["pred_rul"]
diagnosis = active_eval.get("component_diagnosis")
status_label = (
    "Model unavailable"
    if pred_rul is None
    else "Critical wear"
    if pred_rul <= 30
    else ("Elevated wear" if pred_rul <= 75 else "Nominal")
)
status_class = (
    "unavailable"
    if pred_rul is None
    else "critical"
    if pred_rul <= 30
    else ("warning" if pred_rul <= 75 else "normal")
)
rul_display = "—" if pred_rul is None else f"{pred_rul:.1f} cycles"
if pred_rul is None:
    st.info("Model unavailable")
active_eval["health_band"] = status_label
st.caption(
    f"Unit {selected_unit:03d} · Cycles {selected_cycle - 29}–{selected_cycle} · "
    f"{active_eval['estimate_source']}"
)
kpi_rul, kpi_health, kpi_component = st.columns(3)
with kpi_rul, st.container(border=True):
    st.metric("Predicted remaining life", rul_display)
with kpi_health, st.container(border=True):
    st.caption("PREDICTED HEALTH")
    st.markdown(
        f"<span class='status-badge-{status_class}'>{status_label}</span>", unsafe_allow_html=True
    )
    st.caption("No prediction available" if pred_rul is None else "Based on the selected estimate")
with kpi_component, st.container(border=True):
    st.caption("SUSPECTED COMPONENT")
    st.subheader(diagnosis.station_id if diagnosis and pred_rul <= 75 else "—")
    st.caption(
        diagnosis.module_name
        if diagnosis and pred_rul <= 75
        else "No component flagged"
        if diagnosis
        else "Unavailable for this model"
    )

tab_main, tab_components, tab_routes, tab_evaluation = st.tabs(
    ["Overview", "Component details", "Route simulator", "Model evaluation"]
)

with tab_main:
    col_plot, col_action = st.columns([3, 2])
    with col_plot, st.container(border=True):
        st.subheader("Sensor trends")
        station_choice = st.selectbox(
            "Measurements",
            ["All measurements", "Temperature", "Pressure", "Speed & flow"],
        )
        series = active_record.get("series", {})
        groups = [
            ("Temperature", "°R", [("sensor_4", "Exhaust temperature · T50")]),
            (
                "Pressure",
                "psia",
                [
                    ("sensor_11", "HPC static pressure · Ps30"),
                    ("sensor_7", "HPC outlet pressure · P30"),
                ],
            ),
            ("Speed", "rpm", [("sensor_9", "Core speed · Nc")]),
            ("Flow", "ratio", [("sensor_15", "Bypass ratio · BPR")]),
        ]
        for group, unit, channels in groups:
            if (
                station_choice != "All measurements"
                and group != station_choice
                and not (station_choice == "Speed & flow" and group in ("Speed", "Flow"))
            ):
                continue
            # Each measurement has its own visible scale and shares the cycle range.
            # Pressure signals also have separate scales to preserve their small drifts.
            for sensor, label in channels:
                values = series.get(sensor, [])
                if not values:
                    continue
                station_color = {
                    "sensor_4": "#E879F9",
                    "sensor_11": "#FBBF24",
                    "sensor_7": "#FBBF24",
                    "sensor_9": "#FBBF24",
                    "sensor_15": "#38BDF8",
                }[sensor]
                fig = go.Figure(
                    go.Scatter(
                        x=np.arange(selected_cycle - len(values) + 1, selected_cycle + 1),
                        y=values,
                        mode="lines",
                        name=label,
                        line=dict(color=station_color, width=2),
                        hovertemplate=f"Cycle %{{x}}<br>%{{y:.3f}} {unit}<extra>{label}</extra>",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=155,
                    title=dict(text=label, font=dict(size=12, color=station_color)),
                    paper_bgcolor="#1B2026",
                    plot_bgcolor="#1B2026",
                    margin=dict(l=55, r=16, t=35, b=28),
                    showlegend=False,
                    font=dict(color="#B4BDC7"),
                    xaxis=dict(
                        range=[selected_cycle - 29, selected_cycle],
                        gridcolor="#2A3139",
                        zeroline=False,
                    ),
                    yaxis=dict(title=unit, gridcolor="#2A3139", zeroline=False, tickformat=".2f"),
                )
                st.plotly_chart(
                    fig, width="stretch", config={"displayModeBar": False}, key=f"trend_{sensor}"
                )

    with col_action:
        with st.container(border=True):
            st.subheader("Assessment summary")
            st.caption(active_eval["rationale_source"])
            st.markdown(active_eval["cot_diagnostics"])
        with st.container(border=True):
            st.subheader("Suggested next action")
            if pred_rul is None:
                st.write("Load a trained model or select XGBoost to generate an estimate.")
            elif pred_rul <= 30:
                st.write("Review the overhaul recommendation in Component details.")
            elif pred_rul <= 75:
                st.write("Review the inspection recommendation and evaluate the proposed route.")
            else:
                st.write(
                    "Continue monitoring sensor trends and review the next inspection interval."
                )
            st.caption("Route decisions use this same remaining-life estimate.")
        with st.container(border=True):
            st.subheader("Supporting evidence")
            if diagnosis:
                for evidence in diagnosis.thermodynamic_evidence:
                    st.write(f"• {evidence}")
            else:
                st.caption("Component evidence is unavailable for this model.")
    with st.container(border=True):
        st.subheader("Engine schematic")
        st.caption("Colors identify engine sections; they do not indicate health status.")
        st.markdown(
            "<div class='station-legend'>"
            "<span style='color:#38BDF8'>Fan / bypass</span>"
            "<span style='color:#818CF8'>LPC</span>"
            "<span style='color:#FBBF24'>HPC / core</span>"
            "<span style='color:#FB7185'>Combustor</span>"
            "<span style='color:#FB923C'>HPT</span>"
            "<span style='color:#E879F9'>LPT</span></div>",
            unsafe_allow_html=True,
        )
        if schematic_path.exists():
            st.image(str(schematic_path), width="stretch")

with tab_components:
    st.subheader("Component diagnosis")
    if active_eval.get("component_status") == "SUPPORTED" and active_eval.get(
        "component_diagnosis"
    ):
        diag: ComponentFaultDiagnosis = active_eval["component_diagnosis"]

        col_f1, col_f2 = st.columns([13, 7])
        with col_f1:
            with st.container(border=True):
                st.markdown(f"#### Component: `{diag.module_name}`")
                st.markdown(f"**Physical Location**: `{diag.station_name}`")
                st.markdown(f"**Identified Failure Mode**: {diag.fault_mode}")
                st.markdown(
                    f"<div style='font-size: 0.9rem; color: #CBD5E1; margin-top: 6px;'>{diag.degradation_mechanism}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Coupled Thermodynamic Evidence (NASA C-MAPSS Telemetry Drift):")
                for ev in diag.thermodynamic_evidence:
                    st.markdown(f"- `{ev}`")

        with col_f2:
            with st.container(border=True):
                st.markdown("#### Maintenance recommendation")
                if diag.replacement_urgency == "AOG_CRITICAL":
                    st.markdown(
                        "<span class='status-badge-critical'> AOG CRITICAL OVERHAUL</span>",
                        unsafe_allow_html=True,
                    )
                elif diag.replacement_urgency == "PREVENTIVE_INSPECTION":
                    st.markdown(
                        "<span class='status-badge-warning'> PREVENTIVE BORESCOPE INSPECTION</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span class='status-badge-normal'> ALL PARTS NOMINAL</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"<br>**MRO Work Order**: `{diag.maintenance_order}`", unsafe_allow_html=True
                )
                st.markdown(f"**Standard Task Card**: `{diag.borescope_inspection_task}`")
                st.markdown(
                    f"**Isolation Confidence**: `{diag.confidence_score * 100:.1f}% Physics Coupling`"
                )
                st.markdown(
                    f"<div style='font-size: 0.88rem; color: #94A3B8; margin-top: 8px;'>{diag.action_summary}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Parts & maintenance tasks")
        with st.container(border=True):
            cols_h = st.columns([5, 3, 3, 3, 3, 3])
            cols_h[0].markdown("**Part Name / Assembly**")
            cols_h[1].markdown("**OEM Part No.**")
            cols_h[2].markdown("**Subsystem Station**")
            cols_h[3].markdown("**Replacement Urgency**")
            cols_h[4].markdown("**Standard Task Card**")
            cols_h[5].markdown("**Spares Availability**")

            st.markdown("---")

            for part in diag.replacement_parts:
                p_cols = st.columns([5, 3, 3, 3, 3, 3])
                p_cols[0].markdown(f"**{part.part_name}**")
                p_cols[1].code(part.oem_part_number)
                p_cols[2].caption(part.subsystem)

                if part.replacement_status == "IMMEDIATE_REPLACE":
                    p_cols[3].markdown(
                        "<span class='part-pill-critical'> IMMEDIATE</span>",
                        unsafe_allow_html=True,
                    )
                elif part.replacement_status == "STAGE_KIT":
                    p_cols[3].markdown(
                        "<span class='part-pill-warning'> STAGE KIT</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    p_cols[3].markdown(
                        "<span class='part-pill-nominal'> NOMINAL</span>",
                        unsafe_allow_html=True,
                    )

                p_cols[4].caption(part.standard_task_card)
                p_cols[5].caption(part.estimated_lead_time)

    else:
        with st.container(border=True):
            st.info("Model unavailable" if pred_rul is None else "Component diagnosis unavailable")
            st.caption(active_eval["component_unsupported_reason"])


with tab_routes:
    st.subheader("Route simulator")
    st.caption("Compare the selected estimate with the demonstration route requirements.")
    r_col1, r_col2 = st.columns([10, 10])

    with r_col1:
        st.markdown("#### Select Proposed Flight Route")
        route = st.selectbox(
            "Target Flight Segment",
            [
                "Route 1: Trans-Atlantic ETOPS (JFK ──► LHR, 7.5 hrs | Min Safety Margin: 100 Cycles)",
                "Route 2: Continental Trunk (ORD ──► LAX, 4.2 hrs | Min Safety Margin: 45 Cycles)",
                "Route 3: Regional Hub Spoke (ORD ──► DTW, 1.1 hrs | Min Safety Margin: 15 Cycles)",
            ],
            index=0,
        )

        required_margin = 100 if "Route 1" in route else (45 if "Route 2" in route else 15)

    with r_col2:
        st.markdown("#### Simulation result")
        st.caption(active_eval["estimate_source"])
        eval_rul = active_eval["pred_rul"]

        if eval_rul is None:
            st.info("Model unavailable")
            st.caption("No route decision can be calculated without an estimate.")
        elif eval_rul >= required_margin:
            st.markdown(
                f"""
            <div class='route-cleared'>
                ROUTE REQUIREMENT MET<br>
                The selected estimate is {eval_rul:.1f} remaining cycles; this route requires at least {required_margin} cycles.
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.caption("The estimated remaining life meets this demonstration threshold.")
        else:
            st.markdown(
                f"""
            <div class='route-restricted'>
                ROUTE REQUIREMENT NOT MET<br>
                The selected estimate is {eval_rul:.1f} remaining cycles; this route requires at least {required_margin} cycles.
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.caption("Review another route or the component maintenance recommendation.")


with tab_evaluation:
    st.subheader("Prediction & reference")
    reference_cols = st.columns(3)
    reference_cols[0].metric("Predicted RUL", rul_display)
    reference_cols[1].metric("Ground truth RUL", f"{true_rul:.1f} cycles")
    reference_cols[2].metric(
        "Absolute error",
        "—" if active_eval["abs_error"] is None else f"{active_eval['abs_error']:.1f} cycles",
    )
    st.caption(active_eval["estimate_source"])
    st.markdown("---")
    st.markdown("### Model benchmarks")
    st.button("Refresh benchmark results")
    report_path = Path("artifacts/benchmark_results.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        benchmark_models = report["models"]
        benchmark_config = report["configuration"]
        if not isinstance(benchmark_models, dict) or not benchmark_models:
            raise ValueError("No measured model results")
    except (OSError, ValueError, KeyError, TypeError):
        st.info("No current benchmark report is available. Run the benchmark after training.")
        st.code("uv run python -m training.evaluate_baselines", language="bash")
    else:
        st.caption(
            f"Saved evaluation across {len(benchmark_config['test_units'])} test engines. "
            "Metrics are measured per sensor window; lower error is better."
        )
        comparison_rows = [
            {
                "Model": name,
                "RMSE (cycles)": metrics["RMSE"],
                "MAE (cycles)": metrics["MAE"],
                "NASA score": metrics["NASA_Score"],
                "Evaluated windows": metrics["evaluated_windows"],
                "Failed windows": metrics["failed_windows"],
                "Coverage (%)": 100 * metrics["coverage"],
            }
            for name, metrics in benchmark_models.items()
        ]
        st.dataframe(comparison_rows, hide_index=True, width="stretch")
        if any(m["failed_windows"] for m in benchmark_models.values()):
            st.caption(
                "Some predictions failed. Their metrics cover only valid responses, "
                "so compare coverage before comparing errors."
            )
        st.caption(
            "AeroGuard uses its saved sensor encoder and RUL head. Chronos uses frozen "
            "embeddings with a Ridge regressor fitted on the training split. "
            "These scores do not evaluate diagnostic text quality."
        )
        with st.expander("Benchmark run details"):
            st.json(benchmark_config)
        st.download_button(
            "Download benchmark results",
            data=json.dumps(report, indent=2),
            file_name="benchmark_results.json",
            mime="application/json",
        )

    # 6. Scientific Evidence & Operational Limitations (Bottom Expander)
    st.markdown("---")
    with st.expander("Evidence & limitations", expanded=False):
        st.markdown(
            "Responsible AI deployment in commercial aviation demands complete transparency regarding "
            "what is empirically supported versus what operational boundaries must be respected."
        )

        ev_col, lim_col = st.columns(2)

        with ev_col:
            st.markdown("### Evidence")

            st.markdown(
                """
            <div class='evidence-card'>
                <strong>1. Strict Zero Data Leakage Protocol</strong><br>
                Evaluation is conducted exclusively on unseen turbofan units (Engines 81–100). All normalization means and standard deviations were fitted strictly on training engines (Units 1–70), preventing data snooping.
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class='evidence-card'>
                <strong>2. Physical Degradation Grounding</strong><br>
                Multi-channel telemetry patches capture coupled physical dynamics: exhaust gas temperature (T50) rises while High-Pressure Compressor static pressure (Ps30) falls. The model directly learns this aerodynamic clearance loss signature.
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class='evidence-card'>
                <strong>3. Dual-Head Multi-Task Alignment</strong><br>
                The architecture combines numerical scalar regression (MSE loss) with autoregressive language generation (Cross-Entropy loss), producing simultaneously fast numerical bounding and human-readable diagnostic explanations.
            </div>
            """,
                unsafe_allow_html=True,
            )

        with lim_col:
            st.markdown("### Limitations")

            st.markdown(
                """
            <div class='limitation-card'>
                <strong>1. Simulation Environment Fidelity</strong><br>
                NASA C-MAPSS FD001 is a thermodynamic computer simulation under steady-state sea-level cruise. Real commercial engines encounter severe environmental hazards (sand ingestion, volcanic ash, ice, bird strikes, compressor stalls) not captured in FD001.
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class='limitation-card'>
                <strong>2. Single Fault Mode Assumption</strong><br>
                FD001 models only High-Pressure Compressor (HPC) wear. Real aircraft engines experience multi-component degradation interactions (e.g. HPT blade creep, combustor nozzle coking, bearing spalling).
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class='limitation-card'>
                <strong>3. Synthetic Diagnostic Texts</strong><br>
                The training Chain-of-Thought rationales were generated via deterministic domain heuristics. They have not been independently validated against physical borescope photographs or teardown inspection logs.
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
            <div class='limitation-card'>
                <strong>4. Regulatory & Human-in-the-Loop Mandate</strong><br>
                Language models are probabilistic and subject to hallucinations. AeroGuard TSLM is strictly an <em>Advisory Decision Support System</em>. It does not replace FAA/EASA Part 145 certified maintenance release signatures.
            </div>
            """,
                unsafe_allow_html=True,
            )

    with st.expander("Sensor reference"):
        st.subheader("Sensor station mapping")
        st.markdown("""
        AeroGuard TSLM ingests **14 continuous sensor channels** tracking aerothermal, mechanical, and flow dynamics across 7 physical turbofan stations:

        | Engine Station | Sensor ID | Parameter Symbol | Engineering Units | Aeromechanical & Thermodynamic Function | Degradation Sensitivity |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **Station 2 (Fan / LP Spool)** | `sensor_8` | $N_f$ | rpm | Physical Low-Pressure Fan Rotational Speed | Mechanical shaft loading |
        | **Station 2 (Fan / LP Spool)** | `sensor_13` | $N_{f,corr}$ | rpm | Corrected Low-Pressure Fan Rotational Speed | Density-adjusted fan workload |
        | **Station 24 (LPC Exit)** | `sensor_2` | $T_{24}$ | °Rankine | Low-Pressure Compressor (LPC) Exit Total Temp | Intermediate compression heat |
        | **Station 25 (Core Spool)** | `sensor_9` | $N_c$ | rpm | Physical High-Pressure Core Spool Speed | Core compensation RPM |
        | **Station 25 (Core Spool)** | `sensor_14` | $N_{c,corr}$ | rpm | Corrected High-Pressure Core Spool Speed | Thermodynamic speed matching |
        | **Station 30 (HPC Exit)** | `sensor_3` | $T_{30}$ | °Rankine | High-Pressure Compressor (HPC) Exit Total Temp | Primary compression heating |
        | **Station 30 (HPC Exit)** | `sensor_7` | $P_{30}$ | psia | HPC Outlet Total Delivery Pressure | Compressor stage pressure rise |
        | **Station 30 (HPC Exit)** | `sensor_11` | $Ps_{30}$ | psia | HPC Outlet Static Pressure | **Primary Indicator**: Flow capacity loss |
        | **Station 30 (HPC Exit)** | `sensor_12` | $\\Phi$ | pps/psia | Ratio of Fuel Flow to HPC Static Pressure | Burner combustion air ratio |
        | **Station 30 (Bleed Port)** | `sensor_17` | $htBleed$ | BTU/lbm | Enthalpy of Air Bled from Station 30 | Bleed energy extraction |
        | **Station 15 (Bypass Duct)** | `sensor_15` | $BPR$ | - | Fan Bypass Ratio (Bypass Flow / Core Flow) | Core-to-bypass redistribution |
        | **Station 41 (HPT Nozzle)** | `sensor_20` | $W_{31}$ | lbm/s | High-Pressure Turbine (HPT) Coolant Bleed Flow | Thermal cooling mass flow |
        | **Station 42 (LPT Nozzle)** | `sensor_21` | $W_{32}$ | lbm/s | Low-Pressure Turbine (LPT) Coolant Bleed Flow | Turbine stage temperature management |
        | **Station 50 (LPT Exit)** | `sensor_4` | $T_{50}$ | °Rankine | Exhaust Gas Temp (EGT) at Turbine Exit | **Primary Indicator**: Thermal runaway |
        """)

        with st.expander("Excluded sensor channels"):
            st.markdown("""
            Raw NASA C-MAPSS outputs 21 sensor channels. However, **7 sensors are dropped** in the FD001 dataset because they exhibit **zero variance** ($\\sigma = 0.0$) under steady-state sea-level cruise:

            | Station | Sensor ID | Parameter | Value in FD001 | Reason for Zero Variance / Dropped |
            | :--- | :--- | :--- | :--- | :--- |
            | **Station 2** | `sensor_1` | $T_2$ (Total Temp at Fan Inlet) | 518.67 °R | Fixed ambient temperature at sea-level cruise |
            | **Station 2** | `sensor_5` | $P_2$ (Total Pressure at Fan Inlet) | 14.696 psia | Fixed standard atmospheric pressure (1 atm) |
            | **Station 15**| `sensor_6` | $P_{15}$ (Bypass Duct Total Pressure) | 21.61 psia | Invariant duct pressure at constant fan speed |
            | **Engine**    | `sensor_10`| $epr$ (Engine Pressure Ratio) | 1.30 | Unvarying pressure ratio across simulation envelope |
            | **Combustor** | `sensor_16`| $far$ (Burner Fuel-Air Ratio) | 0.03 | Constant fuel-air mixture commanded by FADEC |
            | **FADEC**     | `sensor_18`| $N_{f,dmd}$ (Demanded Fan Speed) | 2388 rpm | Static throttle control command |
            | **FADEC**     | `sensor_19`| $PCN_{fR,dmd}$ (Demanded Corrected Fan Speed) | 100% | Static maximum cruise setpoint |

            Dropping these 7 invariant channels eliminates zero-variance noise and prevents singular covariance matrices during neural network training.
            """)
