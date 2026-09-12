# 🏆 AeroGuard TSLM — Official Hackathon Submission
**European Hackathon League | Temporal AI Challenge (Zurich 2026)**  
**Track: Aionic Labs × ETH Agentic Systems Lab**  
**Repository Branch**: `dev_ashwani`

---

## 1. Executive Summary & Problem Formulation
* **Target User**: Aerospace Fleet Reliability Engineers, Flight Operations Dispatchers, MRO Logistics Planners.
* **Problem**: Turbofan engine wear forecasting under harsh cyclic thermal/aerodynamic stresses. Traditional calendar limits prematurely scrap healthy components or miss rapid wear, leading to emergency outstation groundings costing **$150,000+ per incident**.
* **Solution**: **AeroGuard TSLM** — a multimodal Time-Series Language Model fusing 14 continuous telemetry channels with `SmolLM-135M-Instruct` via OpenTSLM patch encoders and LoRA adapters. Predicts Remaining Useful Life (RUL), diagnoses physical component failure locations, and prescribes OEM overhaul kits.

---

## 2. Submission Deliverables Checklist

| Deliverable | Repository Path / Artifact Location | Description |
| :--- | :--- | :--- |
| **1. Working Demo** | [`demo/app.py`](file:///home/ubuntu/Project/timelapse/demo/app.py) | Streamlit Aviation Mission Control Dashboard with Live Telemetry, Turbofan Digital Twin, Component Fault Isolation, and Route Dispatch Simulator. |
| **2. Code & Training Config** | [`training/train_opentslm.py`](file:///home/ubuntu/Project/timelapse/training/train_opentslm.py) | Full PyTorch training loop, OpenTSLM patch encoder, LoRA configuration ($r=16, \alpha=32$), and dual-head loss functions. |
| **3. Model Checkpoints & Adapters** | [`models/aeroguard_tslm/`](file:///home/ubuntu/Project/timelapse/models/aeroguard_tslm) | Contains `tslm_adapters.pt` (Patch Encoder + RUL Head weights), `lora_adapters/` (PEFT SmolLM adapters), and `preprocessing.json`. |
| **4. TimeNet Connector & Docs** | [`packages/aeroguard-connectors/`](file:///home/ubuntu/Project/timelapse/packages/aeroguard-connectors) | Reusable `nasa/cmapss` TimeNet connector, `dataset.yaml` manifest, typed Pint units (`°R`, `psi`, `rpm`), and unit tests (`pytest`). |
| **5. Baseline Evaluation** | [`training/evaluate_baselines.py`](file:///home/ubuntu/Project/timelapse/training/evaluate_baselines.py) | Held-out benchmark comparing AeroGuard TSLM against Classical ML (XGBoost) and Text-Only LLMs with zero data leakage. |
| **6. End-to-End CLI Demo** | [`scripts/run_demonstration.py`](file:///home/ubuntu/Project/timelapse/scripts/run_demonstration.py) | Self-contained terminal demonstration verifying data sourcing, real inputs, model inference, and target user impact. |

---

## 3. Quick Run & Verification Commands

### 1. Launch Interactive Mission Control Dashboard:
```bash
uv run streamlit run demo/app.py
# Opens at http://localhost:8501
```

### 2. Run End-to-End CLI Demonstration:
```bash
uv run python -m scripts.run_demonstration
```

### 3. Verify Reusable TimeNet Connector Tests:
```bash
uv run pytest packages/aeroguard-connectors/src/aeroguard_connectors/cmapss/tests/
```

### 4. Run Agentic Data Sourcing Pipeline:
```bash
uv run python -m scripts.agentic_data_sourcing
```

### 5. Build TimeNet / TimeF Local Registry:
```bash
uv run python -m scripts.build_timef_registry --out artifacts/registry
```

---

## 4. Evaluation & Baseline Comparison Summary

Evaluated strictly on **held-out turbofan units (Engines 81–100)** with zero data leakage (verified via `training/evaluate_baselines.py`):

| Model Architecture | Input Modality | RUL RMSE | RUL MAE | NASA Score | Explainability & Root Cause | Component Fault Localization |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **AeroGuard TSLM (Ours)** | 14 Continuous Sensor Patches + Prompt | **7.94** | **6.31** | **686.2** | **High** (Causal aerothermal CoT) | **Yes** (Station 30 HPC Rotor Blades & Stator Vanes) |
| **Baseline: Text-Only LLM** | Serialized ASCII Number Tables | 21.11 | 16.81 | 15,197.2 | **Unreliable** (Tabular blindness) | **No** (Fabricated / hallucinated parts) |
| **Classical ML (XGBoost)** | 70 Tabular Summary Stats | 45.91 | 31.02 | 5,602,498.5 | **None** (Black-box numerical scalar) | **No** (Cannot identify failing parts) |
| **Amazon Chronos (T5 Foundation)** | 14 Discretized Time-Series Tokens | 53.93 | 40.12 | 17,218,386.0 | **None** (Pure numerical foundation model) | **No** (Univariate tokens only) |
| **Static Schedule (Legacy)** | Flight Cycle Counter Only | 58.14 | 46.20 | 8,912,400.0 | **None** (Blind calendar threshold) | **No** (Ignores all telemetry) |

---

## 5. Physical Station & Sensor Mapping

AeroGuard TSLM ingests **14 continuous sensor channels** across 7 physical turbofan stations:

* **Station 2 (Fan / LP Spool)**: `sensor_8` ($N_f$ physical fan speed), `sensor_13` ($N_{f,corr}$ corrected fan speed).
* **Station 24 (LPC Exit)**: `sensor_2` ($T_{24}$ LPC outlet temperature).
* **Station 25 (Core Spool)**: `sensor_9` ($N_c$ core speed), `sensor_14` ($N_{c,corr}$ corrected core speed).
* **Station 30 (HPC Exit & Bleed Port)**: `sensor_3` ($T_{30}$ exit temp), `sensor_7` ($P_{30}$ delivery pressure), `sensor_11` ($Ps_{30}$ static pressure — primary erosion indicator), `sensor_12` ($\Phi$ fuel flow ratio), `sensor_17` ($htBleed$ bleed enthalpy).
* **Station 15 (Bypass Duct)**: `sensor_15` ($BPR$ bypass ratio).
* **Station 41 (HPT Nozzle)**: `sensor_20` ($W_{31}$ HPT coolant bleed).
* **Station 42 (LPT Nozzle)**: `sensor_21` ($W_{32}$ LPT coolant bleed).
* **Station 50 (LPT Exit / Exhaust)**: `sensor_4` ($T_{50}$ Exhaust Gas Temperature — primary thermal runaway indicator).

*7 ambient invariant channels ($T_2, P_2, P_{15}, epr, far, N_{f,dmd}, PCN_{fR,dmd}$) with zero variance in FD001 steady-state cruise were dropped to prevent singular covariance matrices.*
