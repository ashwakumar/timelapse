# AeroGuard TSLM

AeroGuard is a Python prototype for turbofan predictive maintenance. It aims to estimate **Remaining Useful Life (RUL)** from engine sensor windows and generate an assessment in natural language.

The repository contains C-MAPSS data preparation, a TimeNet connector, PyTorch/LoRA model training, baseline evaluation, and a Streamlit dashboard. It was developed for the European Hackathon League's Give AI a Sense of Time project.

**Status:** data preparation is validated. Model evaluation and dashboard inference remain unfinished; the code does not yet establish predictive accuracy or validate maintenance diagnoses.

Follow [step.md](step.md) for the full project roadmap, completion criteria, and the next task: reproducible development setup.

## Workflow (Step-by-Step)

1. **Step 1: Agentic Sourcing & Data Preparation**
   - Autonomous Sourcing Agent: `scripts/agentic_data_sourcing.py` defines the target user persona, evaluates candidate open-source datasets (C-MAPSS vs N-CMAPSS vs PHM08), and publishes the verified integrity dossier `artifacts/dataset_sourcing_dossier.json`.
   - Download & validation: `scripts/download_data.py` validates or downloads raw NASA C-MAPSS FD001 telemetry.
   - Preprocessing & windowing: `scripts/preprocess_data.py` parses records, creates 30-cycle observation windows, enforces strict engine splits (Train: 1-70, Val: 71-80, Test: 81-100), and formulates targets.
   - Agentic CoT Diagnostic Synthesis: `scripts/agentic_cot_synthesizer.py` enriches records with multi-sensor thermodynamic reasoning, severity staging (NORMAL, WARNING, CRITICAL), and actionable shop-level maintenance directives.
   - Validation tests: `scripts/tests/test_data_pipeline.py` and `scripts/tests/test_agentic_pipeline.py`.
2. **Step 2: Bring Data into TimeNet**
   - Connector implementation: `packages/aeroguard-connectors/src/aeroguard_connectors/cmapss/connector.py` maps sensor signals into `TimeSeries` with Pint units, `OrdinalAxis`, and tasks (`AnswerTask`, `ScalarPredictionTask`).
   - TimeF registry generation: `scripts/build_timef_registry.py` runs the pipeline and verifies the dataset using TimeNet SDK.
   - Connector verification tests: `packages/aeroguard-connectors/src/aeroguard_connectors/cmapss/tests/test_connector.py`.
3. **Subsequent Steps (Archived)**
   - Model training, baseline evaluation, and demo dashboard are preserved in `archive/` for activation in later stages. See [archive/README.md](archive/README.md).

## Minimal Data Layout

```text
data/
├── README.md
├── raw/
│   └── train_FD001.txt
└── processed/
    ├── windows.jsonl
    └── dataset_manifest.json
```

See [data/README.md](data/README.md) for target definitions, provenance, and data schemas.

| Split | Engine IDs | Windows |
| --- | --- | ---: |
| Train | 1–70 | 2,502 |
| Validation | 71–80 | 355 |
| Test | 81–100 | 806 |

## Active Commands

Run commands from the repository root using `uv`:

```bash
# 1. Run Agentic Data Sourcing to evaluate datasets and generate dossier
uv run python -m scripts.agentic_data_sourcing

# 2. Preprocess data and generate 30-cycle windows
uv run python -m scripts.preprocess_data

# 3. Synthesize aerospace engineering Chain-of-Thought (CoT) diagnostic targets
uv run python -m scripts.agentic_cot_synthesizer

# 4. Run automated test suite (36 tests)
uv run python -m pytest -q

# 5. Build and verify the TimeNet / TimeF dataset registry
uv run python -m scripts.build_timef_registry --out artifacts/registry

# 6. Complete code hygiene and quality checks
make check
```

## Repository Map

- `data/`: Raw and canonical processed dataset with schema documentation.
- `scripts/agentic_data_sourcing.py`: Autonomous candidate evaluation and sourcing dossier generator.
- `scripts/agentic_cot_synthesizer.py`: 4-stage aerothermal diagnostic CoT reasoning and maintenance directive synthesizer.
- `scripts/download_data.py`: Raw data download, validation, and SHA256 caching.
- `scripts/preprocess_data.py`: Offline windowing, non-leaking splits, and manifest creation.
- `scripts/build_timef_registry.py`: TimeNet / TimeF registry builder and verification script.
- `scripts/tests/`: Automated unit tests for data pipeline, connector, and agentic CoT synthesis.
- `packages/aeroguard-connectors/`: TimeNet dataset connector for NASA C-MAPSS with 7 annotations.
- `archive/`: Preserved training (`archive/training/`), checkpoints (`archive/models/`), and dashboard (`archive/demo/`).
- `Makefile`: Tooling commands for lock-check, formatting, linting, and tests.


# 1. Install uv (if not already installed on the server)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Install all dependencies natively in seconds
uv sync --locked

# 3. Launch Full Cloud GPU Training!
# (The code automatically detects NVIDIA CUDA!)
uv run python -m training.train_opentslm \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
