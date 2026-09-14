# ✈️ AeroGuard TSLM

> **Multimodal Time-Series Language Model for Turbofan Predictive Maintenance & Aerothermal Prognostics**

[![Team: TimeLapse](https://img.shields.io/badge/Team-TimeLapse-7C3AED.svg?style=for-the-badge)](#team--acknowledgments)
[![EHL Hackathon Zurich](https://img.shields.io/badge/EHL_Hackathon-Zurich_2026-F97316.svg?style=for-the-badge)](https://github.com/aionic-labs)
[![Track: Aionic Labs x ETH](https://img.shields.io/badge/Track-Aionic%20%C3%97%20ETH%20ASL-06B6D4.svg?style=for-the-badge)](https://github.com/aionic-labs)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AeroGuard TSLM bridges continuous multivariate jet engine telemetry directly into the embedding space of a causal language model (`SmolLM-135M-Instruct`) via **Prefix Token Fusion**. In a single forward pass, it simultaneously forecasts numerical **Remaining Useful Life (RUL)** with sub-millisecond bounding and synthesizes actionable, physically grounded **Chain-of-Thought (CoT) engineering diagnostics**.

Developed for the **European Hackathon League (EHL Hackathon Zurich 2026)** (*"Give AI a Sense of Time"*) using the NASA C-MAPSS FD001 dataset and Aionic's TimeNet connector framework.

---

## 🌟 Key Highlights

- **🎯 State-of-the-Art Prognostics**: Achieves **7.94 RMSE** on held-out test engines—an **85% error reduction** compared to Amazon Chronos T5 (53.93 RMSE).
- **🧠 Physically Grounded Chain-of-Thought**: Generates auditable aerothermal reasoning across 4 diagnostic stages: Sensor Drift Observation, Aerothermal Root Cause Analysis, Fleet Health & Criticality Bounding, and Line-Replaceable Unit (LRU) MRO Action Plan.
- **⚙️ Multimodal Prefix Token Fusion**: Continuous sensor waveforms are projected via a 1D convolutional patch encoder directly into LLM token embeddings, avoiding discrete quantization errors and table-serialization limits.
- **🛡️ Strict Zero-Leakage Data Partitioning**: Engines are partitioned by physical unit ID (Train: 1–70, Val: 71–80, Test: 81–100); normalization statistics are computed strictly on training engines.
- **🖥️ Mission Control Web Dashboard**: Interactive Streamlit GUI featuring 3D turbofan station cross-sections, synchronized Plotly telemetry inspection, and automated ETOPS flight clearance dispatch.

---

## 🏗️ System Architecture

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

- **Prefix Token Fusion**: Continuous telemetry patches are prepended to prompt token embeddings. Prompt tokens are loss-masked (`-100`) so gradients optimize purely for reasoning and diagnostics.
- **Multi-Task Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{RUL}}$$
  Jointly optimizes autoregressive language modeling cross-entropy with auxiliary scalar RUL regression MSE.

---

## 📊 Held-Out Benchmark Results (Engines 81–100)

Evaluated strictly on **held-out test engines (Engines 81–100)** with zero data leakage:

| Model Architecture | Input Modality | RUL RMSE ↓ | RUL MAE ↓ | NASA Score ↓ | Diagnostic Explainability | Component Fault Localization |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **AeroGuard TSLM (Ours)** | 14 Continuous Sensor Patches + Prompt | **7.94** | **6.31** | **686.2** | **High** (Causal aerothermal CoT) | **Yes** (Station 30 HPC Rotor Blades & Stator Vanes) |
| **Baseline: Text-Only LLM** | Serialized ASCII Number Tables | 21.11 | 16.81 | 15,197.2 | **Unreliable** (Tabular blindness) | **No** (Hallucinates nonexistent components) |
| **Amazon Chronos (T5 Foundation)** | 14 Discretized Time-Series Tokens | 53.93 | 40.12 | 17,218,386.0 | **None** (Pure numerical foundation model) | **No** (Univariate tokens only) |
| **Static Schedule (Legacy)** | Flight Cycle Counter Only | 58.14 | 46.20 | 8,912,400.0 | **None** (Blind calendar threshold) | **No** (Ignores all telemetry) |

> 🔍 **Multivariate Thermodynamic Coupling**: Jet engine degradation is coupled: compressor efficiency drops ($Ps_{30}$ static pressure falls) while the combustor compensates by burning richer ($T_{50}$ exhaust temperature surges). Univariate foundation models tokenize channels in isolation and miss these cross-sensor signatures, whereas AeroGuard's continuous patch encoder captures inter-channel correlations directly.

---

## ⚡ Quickstart & Installation

### 1. Clone and Install Dependencies

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
git clone https://github.com/ashwakumar/timelapse.git
cd timelapse

# Install dependencies via uv
uv sync
```

*(Alternatively, standard pip: `pip install -r requirements.txt`)*

### 2. Launch Interactive Mission Control Dashboard

Run the Streamlit web dashboard for live telemetry exploration, digital twin cross-sections, and real-time inference:

```bash
uv run streamlit run demo/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Standalone CLI Demonstration

Run inference on real held-out engine test cases (`Unit #84` nominal vs. near-failure wear) directly in your terminal:

```bash
uv run python -m scripts.run_demonstration
```

---

## 🚀 End-to-End Pipeline & Training

Run the entire pipeline (data sourcing, window slicing, CoT generation, normalization, and OpenTSLM fine-tuning) with a single command:

```bash
# Option A: One-click shell script
./run_pipeline.sh

# Option B: Via Makefile
make pipeline

# Option C: Configurable CLI
uv run python -m scripts.run_pipeline \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
```

> 💡 **Fast Smoke Test**: Verify the pipeline end-to-end in under 30 seconds:
> ```bash
> make smoke-pipeline
> ```

### Key Pipeline Configuration Flags

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--epochs` | `3` | Number of training epochs over preprocessed windows. |
| `--batch-size` | `16` | Per-GPU batch size (use `8` or `4` for smaller VRAM). |
| `--lr` | `2e-4` | Learning rate for AdamW optimizer with linear warmup. |
| `--save-dir` | `models/aeroguard_tslm` | Directory where trained checkpoints and adapters are stored. |
| `--max-steps` | `None` | Caps training at $N$ steps (ideal for rapid testing). |
| `--skip-preprocess` | `False` | Retrains model immediately using existing `data/processed/windows.jsonl`. |
| `--build-registry` | `False` | Builds and validates Aionic's TimeNet / `TimeF` sharded parquet registry. |

### Running Baseline Benchmarks

To train and evaluate baseline comparisons (Chronos + Ridge, Text-only LM, and Static Schedule):

```bash
# Prepare baseline models
uv run python -m training.train_baselines

# Run evaluation on held-out test engines (Engines 81-100)
uv run python -m training.evaluate_baselines --model-dir models/aeroguard_tslm
```

---

## 📁 Repository Structure

```text
├── run_pipeline.sh                      # One-click executable pipeline script
├── README.md                            # Primary documentation & landing page
├── REPO_DETAILS.md                      # Detailed technical architecture & physics dossier
├── Makefile                             # Build, test, and pipeline automation
├── pyproject.toml                       # Python project configuration and dependencies
├── data/
│   ├── raw/train_FD001.txt              # NASA C-MAPSS telemetry dataset (20,631 records)
│   └── processed/windows.jsonl          # 3,663 preprocessed 30-cycle telemetry windows
├── packages/
│   └── aeroguard-connectors/            # Reusable TimeNet dataset connector for C-MAPSS
├── notebooks/
│   ├── 01_raw_telemetry_exploration.ipynb   # Raw telemetry & sensor drift exploration
│   ├── 02_windowing_and_dataset_loader.ipynb# Windowing & PyTorch dataset loader verification
│   ├── 03_model_inference_precheck.ipynb    # Model architecture & inference sanity checks
│   └── 04_generate_slide_plots.ipynb        # Generation of benchmark & diagnostic plots
├── scripts/
│   ├── run_pipeline.py                  # Master pipeline orchestrator
│   ├── preprocess_data.py               # 30-cycle windowing & engine unit partitioning
│   ├── agentic_cot_synthesizer.py       # Aerothermal diagnostic CoT generation
│   ├── run_demonstration.py             # Standalone CLI terminal demonstration
│   ├── build_timef_registry.py          # TimeNet TimeF dataset exporter
│   └── evaluate_chronos_baseline.py     # Amazon Chronos benchmark evaluator
├── training/
│   ├── train_opentslm.py                # OpenTSLM training loop, patch encoder & LoRA
│   ├── dataset_loader.py                # Lazy JSONL loader & normalization manager
│   ├── inference.py                     # AeroGuard inference engine (RUL + CoT generation)
│   ├── fault_isolation.py               # Deterministic aerothermal fault localization
│   ├── train_baselines.py               # Baseline training (Chronos + Ridge, Text-only LM)
│   └── evaluate_baselines.py            # Held-out test evaluation across all models
├── demo/
│   ├── app.py                           # Interactive Streamlit mission control dashboard
│   └── assets/                          # Turbofan schematic and visual assets
├── models/
│   └── aeroguard_tslm/                  # Trained model checkpoints, LoRA adapters & tokenizer
├── presentation/
│   ├── AeroGuard_TSLM_Hackathon_Deck.pptx # Hackathon presentation slide deck
│   └── AeroGuard_TSLM_TimeLapse.pdf     # Presentation slide deck (PDF export)
└── artifacts/
    ├── slide_plots/                     # Standalone interactive Plotly HTML figures
    └── benchmark_results.json           # Evaluation metrics on Engines 81-100
```

---

## 🔬 In-Depth Documentation

For an exhaustive technical and physical analysis, see **[`REPO_DETAILS.md`](REPO_DETAILS.md)**:
- ✈️ **Thermodynamic Engine Physics**: Station-by-station aerothermal breakdown across 7 physical stations.
- 🔬 **Active Sensor Selection**: Why 7 ambient invariant channels were dropped to prevent singular covariance matrices.
- 📐 **Prefix Token Fusion Formulation**: Unfolding mechanics, linear projection tensor math, and loss weighting.
- 📦 **TimeNet Connector & TimeF**: Reusable package architecture in `packages/aeroguard-connectors/` with typed Pint units.
- 🎯 **NASA Scoring Metric**: Mathematical proof and analysis of asymmetric late-prediction penalties.

---

## 🧪 Testing & Code Quality

```bash
# Run test suite
uv run python -m pytest -q

# Run full code hygiene check (formatting, linting, type-checking, tests)
make check
```

---

## 👥 Team & Acknowledgments

**Team TimeLapse — EHL Hackathon Zurich 2026**
- **Ashwani Kumar**
- **Siddhant Tilekar**
- **Vid Tominec**
- **Marlene Moerig**

Developed for the **European Hackathon League (EHL Hackathon Zurich 2026)** in collaboration with **Aionic Labs** and **ETH Zurich Agentic Systems Lab (ASL)**.
