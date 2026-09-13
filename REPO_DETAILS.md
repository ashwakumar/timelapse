# ✈️ AeroGuard TSLM: Comprehensive Repository & Technical Architecture Dossier

> **European Hackathon League (EHL Hackathon Zurich 2026)**  
> **Team**: TimeLapse (Ashwani Kumar, Siddhant Tilekar, Vid Tominec, Marlene Moerig)  
> **Track**: Aionic Labs × ETH Agentic Systems Lab  
> **Project**: AeroGuard TSLM (Multimodal Time-Series Language Model for Turbofan Predictive Maintenance)  
> **Dataset**: NASA C-MAPSS FD001 (`nasa/cmapss`)  
> **Primary Users**: Aerospace Reliability Engineers, Flight Operations Dispatchers, Fleet MRO Managers  

---

## 📑 Table of Contents
1. [Executive Summary & Problem Space](#1-executive-summary--problem-space)
2. [NASA C-MAPSS Telemetry & Aerothermal Physics](#2-nasa-c-mapss-telemetry--aerothermal-physics)
3. [Zero-Leakage Data Partitioning & Contracts](#3-zero-leakage-data-partitioning--contracts)
4. [AeroGuard TSLM Model Architecture](#4-aeroguard-tslm-model-architecture)
5. [Aerospace Chain-of-Thought (CoT) Diagnostic Synthesis](#5-aerospace-chain-of-thought-cot-diagnostic-synthesis)
6. [Aionic TimeNet Connector & TimeF Specification](#6-aionic-timenet-connector--timef-specification)
7. [Empirical Benchmarks & Baseline Comparisons](#7-empirical-benchmarks--baseline-comparisons)
8. [Interactive Mission Control Dashboard](#8-interactive-mission-control-dashboard)
9. [Complete Repository Layout & Directory Map](#9-complete-repository-layout--directory-map)
10. [Engineering Standards & Code Hygiene](#10-engineering-standards--code-hygiene)

---

## 1. Executive Summary & Problem Space

### The $150,000 Outstation Grounding Problem
In commercial and defense aviation, unscheduled aircraft engine removals (UER) and outstation groundings—known as **Aircraft on Ground (AOG)**—cost commercial airlines over **$150,000 per incident** in delays, passenger accommodation, emergency logistics, and charter flights.

### The Limitations of Existing Approaches
Modern high-bypass turbofan engines generate rich multi-sensor telemetry during every flight cycle. However, current aviation maintenance operates under extreme compromises:

1. **Static Calendar Limits (Legacy Rule-Based)**:
   - Engines are overhauled at fixed flight cycle intervals (e.g. every 3,000 cycles) regardless of actual wear.
   - *Failure Mode*: Prematurely retires healthy multi-million-dollar turbine components or completely misses rapid thermal wear induced by aggressive climb profiles.
2. **Pure Time-Series Foundation Models (Amazon Chronos T5)**:
   - Uses univariate tokenization into quantized discrete bins.
   - *Failure Mode*: Blind to multivariate thermodynamic cross-coupling (e.g., simultaneous exhaust gas surge and compressor pressure drop), resulting in high error (53.93 RMSE) and zero natural language diagnostic output.
3. **Standard Text-Only LLMs (Prompting with Raw Arrays)**:
   - Prompting LLMs with serialized ASCII numeric tables or rolling averages.
   - *Failure Mode*: Tabular blindness, token inefficiency, and severe hallucinations of nonexistent part numbers and trends.

### The AeroGuard TSLM Solution
**AeroGuard TSLM** is a multimodal Time-Series Language Model. It bridges continuous multivariate sensor waveforms directly into the embedding space of a causal language model (`SmolLM-135M-Instruct`) via **Prefix Token Fusion**. 

In a single forward pass, AeroGuard TSLM:
- Predicts scalar **Remaining Useful Life (RUL)** with state-of-the-art precision (**7.94 RMSE** on held-out engines).
- Generates auditable, physically grounded **Chain-of-Thought (CoT) engineering diagnostics**.
- Isolates sub-component failure down to the specific Line-Replaceable Unit (LRU) part (e.g., Station 30 HPC Rotor Blades `CFM56-HPC-RB25` under Task Card `AMM 72-31-00`).
- Provides automated flight route clearance and ETOPS (Extended-range Twin-engine Operational Performance Standards) dispatch guidance.

---

## 2. NASA C-MAPSS Telemetry & Aerothermal Physics

### Dataset Overview (FD001)
The Commercial Modular Aero-Propulsion System Simulation (**C-MAPSS**) dataset was developed by NASA Glenn Research Center and NASA Ames Research Center. 

The **FD001** sub-dataset models a dual-spool high-bypass turbofan operating under steady-state sea-level cruise:
- **Total Operational Rows**: 20,631 rows.
- **Engine Population**: 100 unique turbofans run from nominal health to failure threshold.
- **Lifetime Distribution**: Minimum = 128 cycles, Maximum = 362 cycles, Mean = 206.31 cycles.
- **Single Dominant Failure Mode**: High-Pressure Compressor (HPC) flow capacity degradation and adiabatic efficiency loss.

### Physical Engine Stations & Sensor Channel Selection
A dual-spool turbofan engine is organized into standardized aerothermal stations:

```text
               ┌─────────┐             ┌─────────┐
               │ Fan/LPC │             │ HPT/LPT │
               │ Spool   │             │ Spool   │
               └────┬────┘             └────┬────┘
                    │                       │
 ┌───────┐      ┌───┴───┐      ┌─────┐      │      ┌────────┐
 │ Air   │─────►│  Fan  │─────►│ LPC │──────┼─────►│ Bypass │ (Station 15)
 │ Inlet │      │       │      │     │      │      └────────┘
 └───┬───┘      └───┬───┘      └──┬──┘      │
  Station 2     Station 2      Station 24   │
                                            ▼
                                       ┌─────────┐
                                       │   HPC   │ (Station 30: Primary Degradation)
                                       └────┬────┘
                                            ▼
                                       ┌─────────┐
                                       │Combustor│
                                       └────┬────┘
                                            ▼
                                       ┌─────────┐
                                       │ HPT/LPT │ (Stations 41 & 42)
                                       └────┬────┘
                                            ▼
                                       ┌─────────┐
                                       │ Exhaust │ (Station 50: EGT Surge)
                                       └─────────┘
```

NASA C-MAPSS provides 21 raw telemetry channels. AeroGuard TSLM **scientifically retains 14 active aerothermal channels** across 7 physical stations and drops 7 invariant channels:

| Sensor ID | Symbol | Physical Meaning | Physical Unit | Engine Station | Degradation Role |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `sensor_2` | $T_{24}$ | Total temp at LPC outlet | `ureg.degree_Rankine` | Station 24 | Low-pressure compression thermal baseline |
| `sensor_3` | $T_{30}$ | Total temp at HPC outlet | `ureg.degree_Rankine` | Station 30 | Compressor work thermal increase |
| `sensor_4` | $T_{50}$ | Exhaust Gas Temperature (EGT) | `ureg.degree_Rankine` | Station 50 | **Primary thermal runaway indicator** |
| `sensor_7` | $P_{30}$ | Total pressure at HPC outlet | `ureg.psi` | Station 30 | Core aerodynamic compression capability |
| `sensor_8` | $N_f$ | Physical fan speed | `ureg.rpm` | Station 2 | LP spool mechanical rotation |
| `sensor_9` | $N_c$ | Physical core speed | `ureg.rpm` | Station 25 | HP spool mechanical rotation |
| `sensor_11` | $P_{s30}$ | Static pressure at HPC outlet | `ureg.psi` | Station 30 | **Primary blade tip clearance erosion indicator** |
| `sensor_12` | $\phi$ | Fuel flow to $P_{s30}$ ratio | `ureg.lb / (ureg.s * ureg.psi)` | Station 30 | Fuel scheduling compensation |
| `sensor_13` | $N_{Rf}$ | Corrected fan speed | `ureg.rpm` | Station 2 | Standard-day normalized fan speed |
| `sensor_14` | $N_{Rc}$ | Corrected core speed | `ureg.rpm` | Station 25 | Standard-day normalized core speed |
| `sensor_15` | $BPR$ | Bypass ratio | dimensionless | Station 15 | Mass flow ratio (bypass duct / core flow) |
| `sensor_17` | $htBleed$ | Bleed enthalpy | dimensionless | Station 30 | High-pressure extraction enthalpy |
| `sensor_20` | $W_{31}$ | HPT coolant bleed | `ureg.lb / ureg.s` | Station 41 | High-pressure turbine blade cooling flow |
| `sensor_21` | $W_{32}$ | LPT coolant bleed | `ureg.lb / ureg.s` | Station 42 | Low-pressure turbine nozzle cooling flow |

### Why 7 Ambient Invariant Sensors Were Dropped
Sensors 1 ($T_2$), 5 ($P_2$), 6 ($P_{15}$), 10 ($epr$), 16 ($far$), 18 ($N_{f,dmd}$), and 19 ($PCN_{fR,dmd}$) exhibit **mathematically zero variance** ($\sigma = 0.0$) in FD001 because NASA simulated fixed sea-level cruise without atmospheric variation. 

Retaining zero-variance channels introduces artificial noise and creates singular covariance matrices during neural network optimization. Dropping them preserves pure thermodynamic signal fidelity.

### The Coupled Aerothermal Failure Signature
Turbofan degradation is fundamentally multivariate:
$$\Delta Ps_{30} \downarrow \quad \text{and} \quad \Delta T_{50} \uparrow$$
As compressor blade tips erode and stator clearances open up, compression efficiency drops, causing HPC static pressure ($Ps_{30}$) to decrease. To maintain thrust, the flight control system injects more fuel, which causes Exhaust Gas Temperature ($T_{50}$) to surge. Univariate models fail because they cannot observe this coupled divergence.

---

## 3. Zero-Leakage Data Partitioning & Contracts

### Strict Unit-Partitioning (Zero Data Leakage Invariant)
In time-series prognostics, random cross-validation shuffles data across cycles of the same engine, creating severe data leakage (future cycles leak into training). 

AeroGuard TSLM enforces a **strict physical engine partition**:

| Partition | Engine IDs | Window Count | Policy & Invariant |
| :--- | :---: | :---: | :--- |
| **Training** | Engines `1–70` | 2,502 | Fits all population normalization parameters and LoRA weights. |
| **Validation** | Engines `71–80` | 355 | Hyperparameter tuning and model checkpoint selection. |
| **Test** | Engines `81–100` | 806 | Strictly unseen engines; zero leakage; evaluates true physical generalization. |
| **Total** | Engines `1–100` | 3,663 | Complete fleet coverage across all flight lifetimes. |

### Rolling Window Specification
- **Window Length ($T$)**: 30 operational cycles.
- **Stride ($S$)**: 5 operational cycles.
- **Terminal Window Guarantee**: Each engine's final failure window ($RUL = 0$) is guaranteed to be included exactly once.
- **Ground Truth Target ($RUL$)**: $\text{max\_cycle}_{\text{unit}} - \text{current\_cycle}$.

### Normalization Contract
- Population statistics (mean $\mu_c$ and standard deviation $\sigma_c$) are fitted exclusively on the 2,502 training windows.
- Any channel with $\sigma_c < 10^{-6}$ receives scale 1.0.
- Normalization parameters are exported to `models/aeroguard_tslm/preprocessing.json` and locked during validation, testing, and inference.

---

## 4. AeroGuard TSLM Model Architecture

```text
                         ┌────────────────────────────────────────────────────────┐
14 Sensor Channels       │ TimeSeriesPatchEncoder                                 │
(Shape: [B, 14, 30])     │ 1D Unfold (patch_len=10) -> Linear(140, 2048) -> LN    │
────────────────────────►│ Output: [Batch, 3 Patches, 2048]                       │
                         └──────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼ Concatenate
Prompt Text Tokens       ┌──────────────────────────┴─────────────────────────────┐
(Shape: [B, L])          │ Prefix Token Fusion                                    │
────────────────────────►│ Continuous Patches + Text Embeddings [Batch, 3+L, 2048]│
                         └──────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                         ┌────────────────────────────────────────────────────────┐
                         │ SmolLM-135M-Instruct + LoRA Adapters                   │
                         │ (rank=16, alpha=32, targets: q_proj, v_proj, k_proj, o)│
                         └──────────────────────────┬─────────────────────────────┘
                                                    │
                          ┌─────────────────────────┴─────────────────────────┐
                          ▼                                                   ▼
         ┌─────────────────────────────────┐        ┌─────────────────────────────────┐
         │ Auxiliary Scalar RUL Head (MLP) │        │ Autoregressive Causal LM Head   │
         │ Mean-pool -> Linear(2048, 128)  │        │ Computes cross-entropy on CoT   │
         │ -> GELU -> Linear(128, 1)       │        │ tokens (patch/prompt masked -100│
         │ Output: Scalar RUL (Cycles)     │        │ Output: Aerospace Diagnostic CoT│
         └─────────────────────────────────┘        └─────────────────────────────────┘
```

### Module Specifications

#### 1. `TimeSeriesPatchEncoder`
- **Input**: Normalized sensor tensor of shape $[B, C=14, T=30]$.
- **Patch Unfolding**: 1D non-overlapping window with patch length $P = 10$.
  - Number of patches: $N_p = T / P = 30 / 10 = 3$.
  - Patch dimension: $D_p = C \times P = 14 \times 10 = 140$.
- **Linear Projection**: Linear layer mapping from 140 dimensions to the LLM embedding dimension ($D_{\text{model}} = 2048$).
- **Layer Normalization**: Applied directly to projected patch embeddings.
- **Output**: Patch embedding tensor $[B, 3, 2048]$.

#### 2. Base Language Model & LoRA Configuration
- **Backbone**: `HuggingFaceTB/SmolLM-135M-Instruct` (compact, high-efficiency LLaMA causal transformer architecture).
- **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**:
  - Rank: $r = 16$.
  - Scaling: $\alpha = 32$.
  - Dropout: $0.05$.
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
  - Base model weights remain frozen; only LoRA adapters, the patch encoder, and the RUL MLP head receive gradient updates.

#### 3. Prefix Token Fusion & Label Masking
- Patch tokens $[B, 3, 2048]$ are concatenated in front of prompt text embeddings $[B, L, 2048]$, forming a fused sequence of length $3 + L$.
- In the supervised training label tensor, the 3 temporal patch positions and prompt tokens are filled with `-100`, ensuring the causal language modeling loss is strictly evaluated on diagnostic reasoning tokens.

#### 4. Dual-Head Multi-Task Objective
Training simultaneously optimizes language modeling cross-entropy and scalar RUL regression:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{RUL}}$$

- $\mathcal{L}_{\text{LM}}$: Cross-entropy loss computed over autoregressive diagnostic tokens.
- $\mathcal{L}_{\text{RUL}}$: Mean Squared Error (MSE) computed by the auxiliary MLP head operating on mean-pooled temporal embeddings:
  $$\mathcal{L}_{\text{RUL}} = \frac{1}{B} \sum_{i=1}^{B} (\hat{y}_i - y_i)^2$$

---

## 5. Aerospace Chain-of-Thought (CoT) Diagnostic Synthesis

AeroGuard TSLM synthesizes structured, 4-stage engineering Chain-of-Thought rationales:

```text
[STAGE 1: QUANTITATIVE SENSOR OBSERVATION]
Evaluates rolling drift in critical aerothermal channels:
• Exhaust Gas Temp T50 drift (+18.4°R)
• High-Pressure Compressor static pressure Ps30 drift (-3.2 psi)
• Bypass ratio variation (+0.0042)

[STAGE 2: THERMODYNAMIC ROOT-CAUSE MECHANISM]
Translates physical sensor trends into aerodynamic degradation:
• Stage 2-5 HPC rotor blade tip clearance erosion
• Flow boundary layer throttling and turbulent back-pressure

[STAGE 3: REMAINING USEFUL LIFE (RUL) PROJECTION]
Projects structural endurance threshold:
• Exponential decay model bounding remaining life (e.g. 28 cycles)
• Health status: CRITICAL (0-30), WARNING (31-75), or NORMAL (>75)

[STAGE 4: LINE-REPLACEABLE UNIT (LRU) PRESCRIPTION]
Prescribes operational maintenance actions:
• Aircraft Maintenance Manual (AMM) task card reference: AMM 72-31-00
• Line-Replaceable Unit overhaul kit: CFM56-HPC-RB25 (Rotor Blade Assembly)
• Operational routing: Dispatch rejection for Trans-Atlantic ETOPS; reroute to maintenance hub
```

---

## 6. Aionic TimeNet Connector & TimeF Specification

The repository includes a production-grade, reusable TimeNet connector packaged in [`packages/aeroguard-connectors/`](packages/aeroguard-connectors).

### Reusable Package Architecture
- **Manifest Card (`dataset.yaml`)**:
  - `dataset_id`: `nasa/cmapss`
  - `license`: NASA Open Source Agreement / Public Domain
  - `domains`: `engineering`, `aerospace`, `iot`
  - `tags`: `turbofan`, `predictive-maintenance`, `rul`
- **Signal Contract with Typed Pint Units**:
  - Every sensor is mapped to exact physical units via `ureg`: `degree_Rankine`, `psi`, `rpm`, `cycle`.
- **Standardized Multi-Task Definitions**:
  - `AnswerTask`: Prompt + Diagnostic Rationale + Target Response.
  - `ScalarPredictionTask`: Numerical RUL target in `ureg.cycle`.
- **TimeF Sharded Dataset Format**:
  - Encodes telemetry onto standardized `OrdinalAxis(length=30)`.
  - Stored in sharded Apache Parquet files for high-throughput distributed training.
  - Verified via the TimeNet SDK:
    ```python
    from timenet.client import TimeNet
    ds = TimeNet(registry="artifacts/registry").load("nasa/cmapss")
    print(ds.describe())
    ```

---

## 7. Empirical Benchmarks & Baseline Comparisons

### Held-Out Evaluation Protocol (Engines 81–100)
All models were evaluated strictly on the **806 unseen test windows** of Engines 81–100 with zero data leakage:

| Model Architecture | Input Modality | RUL RMSE ↓ | RUL MAE ↓ | NASA Score ↓ | Explainability | Fault Localization |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **AeroGuard TSLM (Ours)** | 14 Continuous Sensor Patches + Text | **7.94** | **6.31** | **686.2** | **High** (Causal CoT) | **Yes** (Station 30 HPC Blades) |
| **Baseline: Text-Only LLM** | Serialized ASCII Number Tables | 21.11 | 16.81 | 15,197.2 | **Unreliable** | **No** (Hallucinates part numbers) |
| **Amazon Chronos (T5)** | 14 Discretized Time-Series Tokens | 53.93 | 40.12 | 17,218,386.0 | **None** (Foundation Model)| **No** (Univariate only) |
| **Static Calendar Schedule** | Flight Cycle Counter Only | 58.14 | 46.20 | 8,912,400.0 | **None** (Blind Threshold) | **No** (Ignores all sensors) |

### Why AeroGuard Outperforms Amazon Chronos (85% Error Reduction)
Amazon Chronos tokenizes time series univariately into quantized discrete buckets. In jet engine degradation, wear is only observable through **cross-channel aerothermal coupling** (falling pressure $Ps_{30}$ accompanied by rising temperature $T_{50}$). Chronos cannot model cross-channel interactions, and its discrete token buckets discard subtle sub-psi micro-drifts. AeroGuard's continuous patch encoder projects all 14 physical channels into a unified embedding space.

### Official NASA Scoring Function
The NASA C-MAPSS scoring metric heavily penalizes late predictions (which lead to catastrophic in-flight failures) compared to early predictions (which cause minor premature maintenance):

$$d = \hat{y} - y$$
$$S = \sum_{i=1}^{N} \begin{cases} e^{-d_i / 13} - 1 & \text{if } d_i < 0 \text{ (early prediction)} \\ e^{d_i / 10} - 1 & \text{if } d_i \ge 0 \text{ (late prediction)} \end{cases}$$

AeroGuard TSLM achieves a NASA Score of **686.2**, outperforming Chronos and text baselines by multiple orders of magnitude.

---

## 8. Interactive Mission Control Dashboard

The Streamlit operations dashboard ([`demo/app.py`](demo/app.py)) provides a modern dark-themed interface for aviation maintenance engineers:

1. **Held-Out Engine Telemetry Viewer**:
   - Select any test engine (`Engine #81` to `#100`) and flight cycle slider.
   - Interactive Plotly multi-sensor waveforms with anomaly threshold bands.
2. **Turbofan Digital Twin & Cross-Section**:
   - Visual mapping of turbofan stations (Fan, LPC, HPC, Combustor, HPT, LPT, Exhaust).
   - Component status indicators highlighting Station 30 erosion.
3. **Interactive Operational Query Bar**:
   - *"🛠️ Component Fault & Part Prescription"*: Pinpoints Stage 2–5 HPC rotor blades (`CFM56-HPC-RB25`) and Task Card `AMM 72-31-00`.
   - *"✈️ Flight Route & ETOPS Dispatch Clearance"*: Live ETOPS flight clearance checks (e.g. automatically rejects Trans-Atlantic crossing and reroutes to Detroit overhaul facility).
   - *"Custom Engineering Prompt"*: Fully editable prompt input for arbitrary diagnostic queries.
4. **Live Streaming Chain-of-Thought**:
   - Autoregressive text streaming of diagnostic rationales with real-time token generation.
5. **Side-by-Side Model Comparison**:
   - Compares AeroGuard against Chronos and Text-Only LLMs on the exact same telemetry window.

---

## 9. Complete Repository Layout & Directory Map

```text
timelapse/
├── README.md                            # Main project landing page & complete operational guide
├── REPO_DETAILS.md                      # Comprehensive technical architecture & physics dossier
├── run_pipeline.sh                      # Shell wrapper for one-click pipeline execution
├── Makefile                             # Build & test tooling (make pipeline, check, test)
├── pyproject.toml                       # Python project dependencies & workspace config (uv)
├── uv.lock                              # Deterministic locked dependency graph
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt              # Raw NASA C-MAPSS telemetry (20,631 records)
│   │   └── train_FD001.source.json      # Provenance metadata and download checksum
│   └── processed/
│       ├── windows.jsonl                # 3,663 preprocessed 30-cycle telemetry windows
│       └── dataset_manifest.json        # Integrity hashes, split IDs, and channel schema
├── packages/
│   └── aeroguard-connectors/            # Reusable TimeNet dataset connector
│       ├── pyproject.toml
│       └── src/aeroguard_connectors/
│           └── cmapss/
│               ├── __init__.py          # Connector export (CONNECTOR = CMAPSSConnector)
│               ├── connector.py         # TimeNet BaseConnector implementation
│               ├── dataset.yaml         # TimeNet dataset manifest card
│               └── tests/
│                   └── test_connector.py# Unit tests with synthetic fixtures
├── notebooks/
│   ├── 01_raw_telemetry_exploration.ipynb   # Raw C-MAPSS data inspection & sensor drift curves
│   ├── 02_windowing_and_dataset_loader.ipynb# 30-cycle windowing & PyTorch loader pre-check
│   ├── 03_model_inference_precheck.ipynb    # Model architecture, weights & inference pre-check
│   └── 04_generate_slide_plots.ipynb        # Visual presentation plots generator (benchmark, drift)
├── scripts/
│   ├── run_pipeline.py                  # Master 5-stage pipeline orchestrator
│   ├── agentic_data_sourcing.py         # Problem definition & raw data validation
│   ├── preprocess_data.py               # 30-cycle window generation & engine splitting
│   ├── agentic_cot_synthesizer.py       # 4-stage aerothermal diagnostic CoT synthesis
│   ├── download_data.py                 # Raw telemetry download & schema verification
│   ├── build_timef_registry.py          # TimeNet TimeF dataset exporter
│   ├── run_demonstration.py             # Standalone CLI terminal demonstration
│   ├── evaluate_chronos_baseline.py     # Amazon Chronos T5 benchmark evaluator
│   └── generate_benchmark_summary.py   # Benchmark JSON formatting & reporting
├── training/
│   ├── train_opentslm.py                # OpenTSLM training loop, patch encoder, LoRA
│   ├── dataset_loader.py                # Lazy JSONL reader, normalization manager, masking
│   ├── inference.py                     # Inference engine (RUL regression + CoT generation)
│   ├── fault_isolation.py               # Deterministic physical fault localization
│   ├── evaluate_baselines.py            # Multi-model evaluation benchmark on held-out test
│   └── tests/
│       ├── test_dataset_loader.py       # Loader regression & zero-leakage tests
│       └── test_model.py                # Architecture & loss function unit tests
├── demo/
│   ├── app.py                           # Interactive Streamlit mission control dashboard
│   └── assets/                          # Graphic assets, diagrams, and visual banners
├── models/
│   └── aeroguard_tslm/                  # Trained model artifacts & configs
│       ├── preprocessing.json           # Normalization statistics (means & scales)
│       ├── tslm_adapters.pt             # Patch encoder & scalar RUL head weights
│       ├── lora_adapters/               # PEFT LoRA adapter weights for SmolLM-135M
│       └── tokenizer/                   # Tokenizer vocabulary and special tokens
├── artifacts/
│   ├── slide_plots/                     # Exported presentation slide figures (HTML & standalone)
│   ├── dataset_sourcing_dossier.json    # Verified data sourcing dossier
│   ├── dataset_sourcing_dossier.md      # Human-readable sourcing summary
│   ├── benchmark_results.json           # Evaluation metrics on Engines 81-100
│   └── benchmark_results.predictions.jsonl # Window-level model predictions
```

---

## 10. Engineering Standards & Code Hygiene

The codebase adheres to strict engineering and reproducibility standards:

* **Package Management**: Managed exclusively through `uv`.
* **Formatting & Linting**: Ruff (`ruff check`, `ruff format`).
* **Type Safety**: Static type checking via `ty check` with full `py.typed` compliance.
* **Test Suite**: Automated unit and regression tests run via `pytest`:
  ```bash
  make check
  ```
* **Hardware Portability**: Automatic CPU fallback when running without CUDA.

---

## 11. Pipeline CLI Reference & Execution Specifications

The master pipeline script (`scripts/run_pipeline.py`) supports full parametric control:

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--epochs` | `int` | `3` | Number of complete training epochs over the 2,502 training windows. |
| `--batch-size` | `int` | `16` | Per-step batch size. Reduce to 8 or 4 if GPU VRAM is restricted. |
| `--lr` | `float` | `2e-4` | Peak learning rate for the AdamW optimizer (with linear warmup). |
| `--save-dir` | `str` | `models/aeroguard_tslm` | Directory where trained LoRA weights, patch encoder, and tokenizer are saved. |
| `--max-steps` | `int` | `None` | Optional step cap for rapid verification (stops after $N$ steps regardless of epoch count). |
| `--skip-preprocess` | `flag` | `False` | Skips Stages 1–3 and directly trains using existing `data/processed/windows.jsonl`. |
| `--build-registry` | `flag` | `False` | Also exports the processed dataset into Aionic's TimeNet / `TimeF` sharded parquet format. |
| `--registry-dir` | `str` | `artifacts/registry` | Destination directory for the TimeF registry. |
| `--model-id` | `str` | `HuggingFaceTB/SmolLM-135M-Instruct` | Hugging Face model repository identifier for base causal LM. |

---

## 12. Troubleshooting & Hardware Diagnostics

1. **`RuntimeError: Triton Error ... Python.h not found`**:
   - *Root Cause*: Triton JIT requires Python C header files to compile optimized CUDA rotary kernels.
   - *Fix*: Install development headers: `sudo apt-get update && sudo apt-get install -y python3-dev python3.12-dev`.
2. **CUDA Out of Memory (OOM)**:
   - *Root Cause*: High batch size during forward/backward pass with multi-patch fusion.
   - *Fix*: Decrease batch size: pass `--batch-size 8` or `--batch-size 4`.
3. **Premature Training Exit (< 1 Epoch)**:
   - *Root Cause*: `--max-steps` was set in the environment or command line.
   - *Fix*: Ensure `--max-steps` is omitted for full-dataset convergence.

