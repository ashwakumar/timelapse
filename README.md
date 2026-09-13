# AeroGuard TSLM

> **Multimodal Time-Series Language Model for Turbofan Predictive Maintenance & Aerothermal Prognostics**

AeroGuard TSLM bridges continuous multivariate sensor waveforms and causal language models (`SmolLM-135M-Instruct`). Using **Prefix Token Fusion**, it embeds 30-cycle turbofan telemetry directly into the LLM token embedding space to simultaneously forecast numerical **Remaining Useful Life (RUL)** and generate actionable **Chain-of-Thought (CoT) engineering diagnostics**.

Developed for the European Hackathon League (*Give AI a Sense of Time*) using the NASA C-MAPSS FD001 dataset and the TimeNet dataset connector framework.

---

## ⚡ Quickstart: End-to-End Pipeline

Run all prerequisite data sourcing, preprocessing, CoT synthesis, normalization, and training in a **single command**:

```bash
# Option A: Run via Python CLI (Configurable)
uv run python -m scripts.run_pipeline \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm

# Option B: Run via Shell Wrapper
./run_pipeline.sh

# Option C: Run via Make
make pipeline
```

### Useful Pipeline Flags

| Flag | Description | Example |
| :--- | :--- | :--- |
| `--epochs` | Number of training epochs (default: `3`) | `--epochs 5` |
| `--batch-size` | Batch size per GPU step (default: `16`) | `--batch-size 16` |
| `--lr` | AdamW learning rate (default: `2e-4`) | `--lr 1e-4` |
| `--save-dir` | Checkpoint and adapter destination | `--save-dir models/aeroguard_tslm` |
| `--max-steps` | Optional step cap for rapid verification | `--max-steps 10` |
| `--skip-preprocess` | Retrain model using existing preprocessed data | `--skip-preprocess` |
| `--build-registry` | Also build and verify the TimeNet / TimeF registry | `--build-registry` |

> 💡 **Quick Smoke Test**: Test the full pipeline end-to-end in under 30 seconds:
> ```bash
> make smoke-pipeline
> ```

---

## 🔄 Pipeline Workflow (Step-by-Step)

If you prefer executing individual pipeline stages manually:

```bash
# 1. Agentic Data Sourcing & Raw Telemetry Validation
# Validates train_FD001.txt and publishes artifacts/dataset_sourcing_dossier.json
uv run python -m scripts.agentic_data_sourcing

# 2. Window Slicing & Engine Partitioning
# Generates 30-cycle windows with stride 5 into data/processed/windows.jsonl
uv run python -m scripts.preprocess_data

# 3. Agentic Chain-of-Thought (CoT) Diagnostic Synthesis
# Enriches records with thermodynamic reasoning and maintenance directives
uv run python -m scripts.agentic_cot_synthesizer

# 4. Normalization Pre-computation & Split Verification
# Fits training population statistics and writes models/aeroguard_tslm/preprocessing.json
uv run python -m training.dataset_loader --save-normalization models/aeroguard_tslm/preprocessing.json

# 5. OpenTSLM Fine-Tuning & Adapter Export
uv run python -m training.train_opentslm \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
```

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

### Key Modules:
- **`TimeSeriesPatchEncoder`**: Converts 14 continuous channels $[B, 14, 30]$ into patch tokens $[B, 3, 2048]$ matching the LLM hidden size.
- **Prefix Token Fusion**: Continuous patch tokens are prepended to prompt embeddings, with prompt tokens masked (`-100`) so loss is computed strictly on reasoning outputs.
- **Multi-Task Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{RUL}}$$
  Simultaneously optimizes causal language modeling cross-entropy and scalar RUL regression MSE.

---

## 📊 Dataset & Zero-Leakage Splits

NASA C-MAPSS **FD001** (High-Pressure Compressor degradation under sea-level steady cruise):

| Split | Physical Engine IDs | 30-Cycle Windows | Zero-Leakage Policy |
| :--- | :---: | :---: | :--- |
| **Train** | Engines `1–70` | 2,502 | Fits all normalization statistics and LoRA weights |
| **Validation** | Engines `71–80` | 355 | Fixed checkpoint selection and hyperparameter tuning |
| **Test** | Engines `81–100` | 806 | Strictly unseen engines; ground-truth RUL withheld |

### 14 Selected Aerothermal Channels:
- **Temperatures**: LPC outlet ($T_{24}$), HPC outlet ($T_{30}$), LPT exhaust gas temp ($T_{50}$)
- **Pressures**: HPC delivery pressure ($P_{30}$), HPC static pressure ($P_{s30}$)
- **Rotor Speeds**: Physical fan speed ($N_f$), Core speed ($N_c$), Corrected speeds ($NR_f, NR_c$)
- **Thermodynamic Ratios & Bleeds**: Fuel-flow ratio ($\phi$), Bypass ratio ($BPR$), Enthalpy, HPT/LPT coolant bleeds ($W_{31}, W_{32}$)

---

## 🚀 Live Demonstration & Evaluation

### Run End-to-End Terminal Demonstration
Evaluates real telemetry cases from held-out engines (`Unit #84` nominal vs. near-failure wear):
```bash
uv run python -m scripts.run_demonstration
```

### Launch Interactive Streamlit Dashboard
Visualizes multi-channel sensor waveforms, live CoT streaming, and fleet health staging:
```bash
uv run streamlit run demo/app.py
```

### Baseline Benchmarking
After training your main model, run all benchmarks with one command:
```bash
uv run python -m training.evaluate_baselines --model-dir models/aeroguard_tslm
```
This fits XGBoost and a Ridge regressor on frozen Chronos embeddings using only the
training split. It evaluates those models, the training-mean predictor, a frozen
text-only SmolLM, and your saved AeroGuard scalar prediction head on the same test
windows. AeroGuard is not retrained. Overlapping engine splits are rejected.
Pretrained models download into the Hugging Face cache if missing.

Results go to `artifacts/benchmark_results.json`, individual predictions to
`artifacts/benchmark_results.predictions.jsonl`, and fitted baseline parameters to
`models/baselines/`. The report includes text-model failure counts and coverage;
text metrics only cover valid numeric responses. These are window-level RUL metrics,
not a measurement of generated diagnostic quality. The dashboard comparison reads this
report on each rerun; click **Refresh benchmark results** after evaluation finishes.

Use `--device cuda` for GPU inference, `--batch-size 16` to control Chronos memory,
and `--out-path` / `--baseline-dir` to preserve separate runs. Text-only generation
runs on every test window and may be slow on CPU.

---

## 🛠️ Code Hygiene & Testing

Run the automated test suite and formatting checks:

```bash
# Run complete test suite (unit tests for pipeline, connector, and CoT)
uv run python -m pytest -q

# Full code hygiene check (lockfile, formatting, linting, typing, tests)
make check
```

---

## 📁 Repository Layout

```text
├── run_pipeline.sh                     # One-click executable bash pipeline wrapper
├── Makefile                            # Tooling: make pipeline, check, test, format
├── pyproject.toml                      # Workspace configuration and dependencies (uv)
├── data/
│   ├── raw/train_FD001.txt             # Validated raw NASA telemetry
│   └── processed/windows.jsonl         # 30-cycle observation windows with CoT rationales
├── scripts/
│   ├── run_pipeline.py                 # End-to-end Python pipeline orchestrator
│   ├── agentic_data_sourcing.py        # Problem definition & candidate dataset sourcing
│   ├── preprocess_data.py              # Offline windowing & zero-leakage splits
│   ├── agentic_cot_synthesizer.py      # 4-stage aerothermal diagnostic CoT synthesis
│   ├── build_timef_registry.py         # TimeNet / TimeF dataset exporter
│   └── run_demonstration.py            # Operational fleet demonstration script
├── training/
│   ├── train_opentslm.py               # OpenTSLM multi-task training script
│   ├── dataset_loader.py               # Lazy JSONL reader & normalization manager
│   ├── inference.py                    # Inference engine with RUL + CoT generation
│   └── evaluate_baselines.py           # Baseline benchmarks (XGBoost vs TSLM)
├── packages/
│   └── aeroguard-connectors/           # Reusable TimeNet dataset connector for C-MAPSS
├── demo/
│   └── app.py                          # Interactive Streamlit operations dashboard
└── models/
    └── aeroguard_tslm/                 # Exported LoRA weights, patch encoder & tokenizer
```
