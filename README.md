# AeroGuard TSLM

> **Multimodal Time-Series Language Model for Turbofan Predictive Maintenance & Aerothermal Prognostics**

[![Team: TimeLapse](https://img.shields.io/badge/Team-TimeLapse-7C3AED.svg?style=for-the-badge)](#)
[![EHL Hackathon Zurich](https://img.shields.io/badge/EHL_Hackathon-Zurich_2026-F97316.svg?style=for-the-badge)](https://github.com/aionic-labs)
[![Track: Aionic Labs x ETH](https://img.shields.io/badge/Track-Aionic%20%C3%97%20ETH%20ASL-06B6D4.svg?style=for-the-badge)](https://github.com/aionic-labs)

### 👥 Team TimeLapse — EHL Hackathon Zurich 2026
* **Ashwani Kumar**
* **Siddhant Tilekar**
* **Vid Tominec**
* **Marlene Moerig**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AeroGuard TSLM bridges continuous multivariate jet engine telemetry directly into the embedding space of a causal language model (`SmolLM-135M-Instruct`) via **Prefix Token Fusion**. In a single forward pass, it simultaneously forecasts numerical **Remaining Useful Life (RUL)** with sub-millisecond bounding and synthesizes actionable, physically grounded **Chain-of-Thought (CoT) engineering diagnostics**.

Developed for the **European Hackathon League (EHL Hackathon Zurich)** (*Give AI a Sense of Time*) using the NASA C-MAPSS FD001 dataset and Aionic's TimeNet connector framework.

---

## ⚡ Quickstart: Run Pipeline in One Command

Execute the complete end-to-end pipeline (data sourcing, window slicing, CoT synthesis, normalization, and OpenTSLM fine-tuning) in a **single command**:

```bash
# Option A: One-click shell script (Default production settings)
./run_pipeline.sh

# Option B: Via Makefile
make pipeline

# Option C: Configurable Python CLI
uv run python -m scripts.run_pipeline \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
```

> 💡 **Need a fast smoke test?** Verify the full pipeline end-to-end in under 30 seconds:
> ```bash
> make smoke-pipeline
> ```

---

## ⚙️ Pipeline Configuration & Common Recipes

The master pipeline orchestrator (`scripts/run_pipeline.py`) accepts convenient operational flags:

| Flag | Type | Default | Description | Example |
| :--- | :---: | :---: | :--- | :--- |
| `--epochs` | `int` | `3` | Number of training epochs over the 2,502 training windows. | `--epochs 10` |
| `--batch-size` | `int` | `16` | Batch size per GPU step. Use `8` or `4` if GPU VRAM is constrained. | `--batch-size 8` |
| `--lr` | `float` | `2e-4` | Learning rate for AdamW optimizer with linear warmup. | `--lr 1e-4` |
| `--save-dir` | `str` | `models/aeroguard_tslm` | Directory where trained checkpoints, adapters, and configs are stored. | `--save-dir models/my_run` |
| `--max-steps` | `int` | `None` | Caps training at $N$ steps regardless of epochs (ideal for smoke tests). | `--max-steps 10` |
| `--skip-preprocess` | `flag` | `False` | Retrains model immediately using existing `data/processed/windows.jsonl`. | `--skip-preprocess` |
| `--build-registry` | `flag` | `False` | Also builds and validates Aionic's TimeNet / `TimeF` sharded parquet registry. | `--build-registry` |

### Useful CLI Recipes

```bash
# ⚡ 30-Second Smoke Run (Cap at 10 optimizer steps)
uv run python -m scripts.run_pipeline --max-steps 10 --save-dir models/smoke_test

# 🔄 Rapid Retraining (Skip preprocessing stages)
uv run python -m scripts.run_pipeline --skip-preprocess --epochs 5 --save-dir models/aeroguard_v2

# 📦 Full Production Run + TimeNet TimeF Registry Export
uv run python -m scripts.run_pipeline --epochs 3 --build-registry
```

---

## 🔄 Pipeline Workflow (5 Core Stages)

The pipeline executes five sequential stages with **strict zero data leakage**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA SOURCING & RAW VALIDATION (scripts/agentic_data_sourcing.py)        │
│    • Checks/downloads data/raw/train_FD001.txt                              │
│    • Verifies SHA-256 (963b5e22...), 20,631 rows, 100 engine units          │
│    • Emits artifacts/dataset_sourcing_dossier.json                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. WINDOW SLICING & SPLIT FORMULATION (scripts/preprocess_data.py)          │
│    • Filters to 14 active degradation channels (drops 7 zero-variance ones) │
│    • Slices into 30-cycle observation windows (stride 5)                    │
│    • Partitions by engine ID (Train: 1–70, Val: 71–80, Test: 81–100)        │
│    • Generates data/processed/windows.jsonl & dataset_manifest.json         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CHAIN-OF-THOUGHT (CoT) SYNTHESIS (scripts/agentic_cot_synthesizer.py)   │
│    • Calculates sensor drifts (T50 EGT surge, Ps30 drop, BPR fluctuation)   │
│    • Synthesizes 4-stage engineering rationales & maintenance directives    │
│    • Assigns Station 30 HPC component fault isolations & CFM56 LRU part IDs │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. NORMALIZATION PRE-COMPUTATION (training/dataset_loader.py)               │
│    • Fits population mean & std strictly on training engines (1–70)         │
│    • Preserves zero leakage into validation (71–80) and test (81–100) sets  │
│    • Saves statistics to models/aeroguard_tslm/preprocessing.json           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. OPENTSLM FINE-TUNING & ADAPTER EXPORT (training/train_opentslm.py)       │
│    • Trains TimeSeriesPatchEncoder (unfold 10 -> Linear 2048 -> LayerNorm)  │
│    • Trains LoRA adapters on SmolLM-135M-Instruct (q, k, v, o projections)  │
│    • Jointly optimizes LM Cross-Entropy + Auxiliary Scalar RUL MSE Loss     │
│    • Exports weights, adapters, and tokenizer to models/aeroguard_tslm/     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Manual Stage-by-Stage Execution
To run any stage individually:
```bash
# Stage 1: Data Sourcing & Schema Validation
uv run python -m scripts.agentic_data_sourcing

# Stage 2: Window Slicing (30 cycles, stride 5)
uv run python -m scripts.preprocess_data --window-size 30 --stride 5

# Stage 3: Chain-of-Thought Synthesis
uv run python -m scripts.agentic_cot_synthesizer

# Stage 4: Precompute Normalization Statistics
uv run python -m training.dataset_loader --save-normalization models/aeroguard_tslm/preprocessing.json

# Stage 5: Fine-Tuning Execution
uv run python -m training.train_opentslm --epochs 3 --batch-size 16 --save-dir models/aeroguard_tslm
```

---

## 🏗️ System Architecture & Multi-Task Objective

```text
                    ┌────────────────────────────┐
14 Sensor Channels  │ TimeSeriesPatchEncoder     │
 (Window Length 30) │ Unfold (patch=10) + Linear │
───────────────────►│ + LayerNorm                │───► [Batch, 3 Patches, 2048] (Patch Tokens)
                    └────────────────────────────┘              │
                                                                ▼ Concatenate
Prompt / Context    ┌────────────────────────────┐          ┌──────────────┐
Text Query          │ Tokenizer + Embedding      │─────────►│ Prefix Token │──► LoRA SmolLM-135M
───────────────────►│ Layer                      │          │ Fusion       │    (r=16, alpha=32)
                    └────────────────────────────┘          └──────────────┘
                                                                │        │
                                                ┌───────────────┘        └───────────────┐
                                                ▼                                        ▼
                                   Auxiliary RUL Head (MLP)                 Autoregressive Causal LM
                                   Mean-pool -> 128 -> GELU -> 1            Predicts diagnostic CoT:
                                   Output: Scalar RUL (Cycles)              Observations, Reasoning,
                                                                            Health & MRO Actions
```

- **`TimeSeriesPatchEncoder`**: Projects 14 continuous channels $[B, 14, 30]$ into patch tokens $[B, 3, 2048]$ matching the LLM hidden size.
- **Prefix Token Fusion**: Continuous patch tokens are prepended to prompt embeddings, with prompt tokens masked (`-100`) so loss is computed strictly on reasoning outputs.
- **Multi-Task Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{RUL}}$$
  Simultaneously optimizes causal language modeling cross-entropy and scalar RUL regression MSE.

---

## 📊 Held-Out Baseline Evaluation (Engines 81–100)

Evaluated strictly on **held-out test engines (Engines 81–100)** with zero data leakage:

| Model Architecture | Input Modality | RUL RMSE ↓ | RUL MAE ↓ | NASA Score ↓ | Explainability & Root Cause | Component Fault Localization |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **AeroGuard TSLM (Ours)** | 14 Continuous Sensor Patches + Prompt | **7.94** | **6.31** | **686.2** | **High** (Causal aerothermal CoT) | **Yes** (Station 30 HPC Rotor Blades & Stator Vanes) |
| **Baseline: Text-Only LLM** | Serialized ASCII Number Tables | 21.11 | 16.81 | 15,197.2 | **Unreliable** (Tabular blindness) | **No** (Fabricated / hallucinated parts) |
| **Amazon Chronos (T5 Foundation)** | 14 Discretized Time-Series Tokens | 53.93 | 40.12 | 17,218,386.0 | **None** (Pure numerical foundation model) | **No** (Univariate tokens only) |
| **Static Schedule (Legacy)** | Flight Cycle Counter Only | 58.14 | 46.20 | 8,912,400.0 | **None** (Blind calendar threshold) | **No** (Ignores all telemetry) |

> 🔍 **Why AeroGuard Beats Chronos (85% Error Reduction)**: Jet engine degradation is fundamentally multivariate. Wear is only detectable through coupled aerothermal drift ($Ps_{30}$ static pressure falling while $T_{50}$ exhaust temperature surges). Amazon Chronos tokenizes each channel univariately into discrete token bins and misses this coupling.

---

## 🚀 Live Demonstration & Benchmarks

### 1. Launch Interactive Mission Control Dashboard
Opens the Streamlit web dashboard with interactive Plotly telemetry, Digital Twin station cross-sections, live streaming Chain-of-Thought, and ETOPS flight route dispatch clearance:
```bash
uv run streamlit run demo/app.py
```
*Access at: `http://localhost:8501`*

### 2. Run Baseline Benchmark Evaluation
Prepare the baseline models once (requires processed windows and downloads pretrained backbones):
```bash
uv run python -m training.train_baselines
```
This trains a Ridge RUL head on frozen Chronos embeddings using only training engines.
The text-only model is saved locally as a frozen pretrained baseline, without fine-tuning.
Artifacts are stored in `models/baselines`; repeat runs skip preparation unless `--force` is supplied.

The benchmark and GUI share `training.baseline_inference.BaselinePredictor` and load local saved
weights without training or downloading. Restart the GUI or use **Advanced → Reload model checkpoint**
after preparing models. Chronos and text-only predictions then appear in the model selector.

For a single saved-model prediction:
```bash
uv run python -m training.baseline_inference --model "Chronos + Ridge" --unit 84 --cycle 30
```
Use an engine/cycle present in your processed dataset. The other model name is `Text-only LM`.

Evaluates all models on held-out test windows using saved weights and writes `artifacts/benchmark_results.json`:
```bash
uv run python -m training.evaluate_baselines --model-dir models/aeroguard_tslm
```

### 3. Run Standalone Terminal Demonstration
Executes real telemetry cases from held-out engines (`Unit #84` nominal vs. near-failure wear):
```bash
uv run python -m scripts.run_demonstration
```

---

## 💾 Output Artifacts & Saved Checkpoints

```text
data/
├── raw/train_FD001.txt                   # Verified raw NASA C-MAPSS telemetry (20,631 rows)
└── processed/
    ├── windows.jsonl                     # 3,663 records (Train: 2502, Val: 355, Test: 806)
    └── dataset_manifest.json             # Hash integrity, split metadata, channel order

models/aeroguard_tslm/
├── preprocessing.json                    # Means, stds, and channel order for 14 sensors
├── tslm_adapters.pt                      # Weights for TimeSeriesPatchEncoder & RUL MLP Head
├── lora_adapters/
│   ├── adapter_config.json               # PEFT LoRA parameters (r=16, alpha=32)
│   └── adapter_model.safetensors         # Trained LoRA adapters for SmolLM-135M
└── tokenizer/
    ├── tokenizer.json                    # SmolLM vocabulary & token mappings
    └── tokenizer_config.json

artifacts/
├── dataset_sourcing_dossier.json         # Raw schema and validation dossier
├── benchmark_results.json                # Benchmark metrics on held-out test split
└── benchmark_results.predictions.jsonl   # Per-window predictions across all models
```

---

## 🛠️ Code Hygiene, Testing & Troubleshooting

```bash
# Run test suite
uv run python -m pytest -q

# Full code hygiene check (lockfile, formatting, linting, typing, tests)
make check
```

### Common Troubleshooting
* **`Triton Error: Python.h not found`**: Install Python C headers: `sudo apt-get update && sudo apt-get install -y python3-dev python3.12-dev`.
* **CUDA Out of Memory (OOM)**: Reduce batch size: `./run_pipeline.sh --batch-size 8`.
* **CPU Execution**: Automatically supported; runs cleanly on CPU if CUDA is unavailable.

---

## 📖 Deep Dive Technical Documentation

For the comprehensive scientific, physical, and architectural breakdown, see **[`REPO_DETAILS.md`](REPO_DETAILS.md)**:
* ✈️ **Thermodynamic Engine Physics**: Station-by-station aerothermal breakdown across 7 physical stations.
* 🔬 **14 Active Sensors Derivation**: Why 7 ambient invariant channels were dropped to prevent singular covariance matrices.
* 📐 **Prefix Token Fusion Mathematics**: Unfolding mechanics, linear projection tensor math, and loss weighting.
* 📦 **TimeNet Connector & TimeF**: Reusable package architecture in `packages/aeroguard-connectors/` with typed Pint units (`°R`, `psi`, `rpm`).
* 🎯 **NASA Scoring Metric**: Mathematical proof and analysis of asymmetric late-prediction penalties.

---

## 📁 Repository Directory Structure

```text
├── run_pipeline.sh                      # One-click executable bash pipeline wrapper
├── README.md                            # This comprehensive operational guide & landing page
├── REPO_DETAILS.md                      # Exhaustive technical architecture & physics dossier
├── Makefile                             # Build & test tooling (make pipeline, check, test)
├── pyproject.toml                       # Workspace configuration and dependencies (uv)
├── uv.lock                              # Deterministic locked dependency graph
├── data/
│   ├── raw/train_FD001.txt              # Raw NASA C-MAPSS telemetry (20,631 records)
│   └── processed/windows.jsonl          # 3,663 preprocessed 30-cycle telemetry windows
├── packages/
│   └── aeroguard-connectors/            # Reusable TimeNet dataset connector for C-MAPSS
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
│   └── evaluate_chronos_baseline.py     # Amazon Chronos T5 benchmark evaluator
├── training/
│   ├── train_opentslm.py                # OpenTSLM training loop, patch encoder, LoRA
│   ├── dataset_loader.py                # Lazy JSONL reader, normalization manager, masking
│   ├── inference.py                     # Inference engine (RUL regression + CoT generation)
│   ├── fault_isolation.py               # Deterministic physical fault localization
│   └── evaluate_baselines.py            # Multi-model evaluation benchmark on held-out test
├── demo/
│   ├── app.py                           # Interactive Streamlit mission control dashboard
│   └── assets/                          # Graphic assets, diagrams, and visual banners
├── models/
│   └── aeroguard_tslm/                  # Trained model artifacts & configs
└── artifacts/
    ├── slide_plots/                     # Exported presentation slide figures (HTML & standalone)
    ├── dataset_sourcing_dossier.json    # Verified data sourcing dossier
    └── benchmark_results.json           # Evaluation metrics on Engines 81-100
```
