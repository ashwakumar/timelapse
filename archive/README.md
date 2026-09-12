# Archived Components

This directory preserves implementations that are downstream of the initial data sourcing and TimeNet ingestion steps. They were moved here to allow focused, step-by-step development according to the hackathon milestones:

1. **Step 1: Find a problem & source the data** (Active in root: `data/`, `scripts/`)
2. **Step 2: Bring the data into TimeNet** (Active in root: `packages/aeroguard-connectors/`, `scripts/build_timef_registry.py`)
3. **Step 3: Temporal / Language Model Training** (Active in root: `training/`)
4. **Step 4: Held-out Evaluation & Baselines** (Active in root: `scripts/evaluate_chronos_baseline.py`)
5. **Step 5: Interactive Dashboard & Demo** (Archived in `archive/demo/`)

---

## Directory Contents

### `archive/training/`
- **`dataset_loader.py`**: PyTorch Dataset loading C-MAPSS JSONL windows with lazy byte offsets, training-only normalization statistics, and token masking.
- **`train_opentslm.py`**: Custom fine-tuning script targeting `SmolLM-135M-Instruct` with LoRA attention adapters and a patch temporal encoder.
- **`evaluate_baselines.py`**: Evaluates baseline regressors (XGBoost) and comparison models.
- **`tests/test_dataset_loader.py`**: Unit tests verifying dataset loading, normalization persistence, and data leakage protection.

### `archive/models/`
- **`aeroguard_tslm/`**: Checkpoints containing LoRA adapters, temporal encoder weights, and tokenizer files.
- **`preprocessing.json`**: Normalization parameters (means, standard deviations, channel order) for the 14 selected sensor channels.

### `archive/demo/`
- **`app.py`**: Streamlit application with interactive Plotly telemetry graphs, RUL cards, and maintenance diagnostic explanations.
- **`assets/`**: Architecture diagrams and visual assets for the demo interface.

---

## How to Reactivate

When ready to proceed to model training or the demo UI:
1. Move the corresponding folder back to the project root:
   ```bash
   mv archive/training ./
   mv archive/models ./
   mv archive/demo ./
   ```
2. Update `pyproject.toml` and `Makefile` to include the reinstated paths in linting, type-checking, and test discovery.
