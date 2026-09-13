# timelapse

## Real VitalDB training on Nebius

Use [NEBIUS_TRAINING.md](NEBIUS_TRAINING.md) for the complete GPU handoff:
`make vitaldb-setup`, `make vitaldb-data`, `make vitaldb-check`,
`make vitaldb-train EPOCHS=5`, then `make vitaldb-export` to return the tuned
weights and exact evaluation data through a GitHub release. Run
`make vitaldb-eval DEVICE=cpu` on the returned bundle for the final local test.

This new path trains actual OpenTSLM encoder/projector/LoRA weights against
real VitalDB hypotension-onset targets. The older `training.train` example
configuration and committed `run-opentslm-*` checkpoints use synthetic software
fixtures. `scripts/finetune_opentslm_hypotension.py` is a small MLP substitute,
not the real OpenTSLM training entry point. Do not use either for the new run.

The target is an onset category, not numerical future MAP. The strict new
preparation excludes the feature-screen cohort and reserves a patient-disjoint
15% test split. Software tests passing does not establish predictive usefulness.

Exploratory VitalDB pipeline for predicting the onset of sustained
intraoperative hypotension from a 20-second observation window.

The repository holds two independent pipelines:

| Directory | Purpose | Environment |
| --- | --- | --- |
| `scripts/`, `connectors/` | Screen VitalDB parameters, prepare windowed tensors, and publish them through the OpenTSLM adapter and the TimeNet connector | `requirements.txt` |
| `training/` | Fine-tune an OpenTSLM SoftPrompt encoder and LoRA adapter on authored surgical telemetry explanations | `requirements-training.txt` |

The two dependency sets pin conflicting versions of NumPy and scikit-learn, so
keep them in separate virtual environments. This file documents the first
pipeline; see [`training/README.md`](training/README.md) for the second.

## Problem and data sourcing

**Target user:** an anesthesiologist or anesthesia trainee at the operating-room
monitor.

**Problem:** current alarms fire when MAP is already low. The useful task is to
read a short multivariate window and say whether a *new* sustained MAP < 65 mmHg
episode is about to start in 3, 5, 10, or 15 minutes, then prompt a reassessment
rather than a drug dose.

[`sourcing/`](sourcing/) is the agentic search → assess → select → retrieve loop
for that card. It does not start from VitalDB as a given: a seed catalog of OR,
ICU, wearable, and industrial time-series sets is merged with a Hugging Face Hub
search, scored against hard constraints (MAP/ABP, subject IDs, uncredentialed
programmatic access, research license), and the top hard-pass is fetched far
enough to prove the access URL still answers. Failures are excluded and the
agent retries.

```bash
python scripts/source_datasets.py
python scripts/source_datasets.py --offline
```

The run writes `artifacts/data_sourcing/report.json` and `selection.json`.
VitalDB is the expected selection for this problem; nearby datasets (HiRID,
MIMIC waveforms, PulseDB, WESAD, C-MAPSS) remain in the report with rejection
reasons. Keep this step: TimeNet ingestion refuses to run without that
selection file.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Every command below assumes this environment is active and that you are in the
repository root, so that `scripts` and `connectors` resolve as packages.

## Feature screen

```bash
python scripts/feature_screen.py --max-cases 40
```

The command downloads and caches selected public VitalDB tracks, creates
patient-disjoint development and validation splits, derives a deterministic
target (MAP below 65 mmHg for at least 60 seconds within 15 minutes), and
writes to `artifacts/feature_screen/`:

- `parameter_ranking.csv` and `reliable_parameter_ranking.csv`
- `feature_ranking.csv`
- `group_ablation.csv`
- `top20_feature_heatmap.png`
- `report.json`
- `window_features.csv.gz` (untracked)

The heatmap shows Spearman relationships and redundancy. Parameter selection
uses grouped permutation loss in validation AUPRC, and the reliable ranking
keeps only parameters observed in at least half of the validation windows.
This is an exploratory screen; the validation patients used for selection are
not a final test set.

A ranking is already committed under `artifacts/feature_screen/`, so this step
is optional. Rerun it only to regenerate that ranking with different sampling
or window parameters.

## LSTM dataset

The positional argument always selects the best `N` dynamic parameters from the
saved reliable ranking. It does not accept an arbitrary feature list, and the
selection must include MAP, which defines input quality.

```bash
python scripts/prepare_lstm_dataset.py 5 --max-cases 40
python scripts/prepare_lstm_dataset.py 10 --max-cases 40
```

Each run writes patient-disjoint `train.npz`, `validation.npz`, and `test.npz`
files under `data/lstm/top_N/`, plus a `manifest.json` containing the exact
parameter order, labels, split counts, availability, and train-only scaling.
Every NPZ contains normalized `x_values`, a separate `x_mask`, multiclass `y`,
and case, subject, cutoff, and future-coverage metadata. Pass `--ranking-file`
to read a different ranking and `--output-root` to write elsewhere.

The target is generated with code: first onset of MAP below 65 mmHg for at
least 60 contiguous observed seconds, bucketed into 3, 5, 10, 15, or no event
within 15 minutes. A fully observed preceding minute at or above 65 is required
so the task predicts a new onset rather than continuation of an existing event.

These are generated research labels, not clinically adjudicated outcomes. The
label builder inspects only the available 900-second future slice, so an episode
starting near the end of that slice may not have enough follow-up to confirm
60 seconds of low MAP and can be mislabeled `none_within_15`. Missing future MAP
samples can also hide a qualifying event. `cutoff_sec` is the timestamp of the
first excluded sample; the final input is one `interval_sec` before it.

## OpenTSLM adapter for Nebius

[`scripts/opentslm_vitaldb_dataset.py`](scripts/opentslm_vitaldb_dataset.py)
implements the sample dictionary used by
[OpenTSLM](https://github.com/OpenTSLM/OpenTSLM) at commit
`2968f4b891baab4307f7e9d0043e87677b593a30`. It emits
`pre_prompt`, `time_series_text`, `time_series`, `post_prompt`, and `answer`.
Each normalized signal is followed by its exact observation mask. The
future-derived label occurs only in `answer`.

Validate the prepared corpus locally before copying it to Nebius:

```bash
python scripts/opentslm_vitaldb_dataset.py \
  --data-dir data/lstm/top_10 --split train --show-sample 0
```

Install the pinned OpenTSLM checkout into the Nebius environment, then run the
training code from this repository root so `scripts` and `data` resolve:

```bash
git clone https://github.com/OpenTSLM/OpenTSLM.git ../OpenTSLM
git -C ../OpenTSLM checkout 2968f4b891baab4307f7e9d0043e87677b593a30
python -m pip install -e ../OpenTSLM
```

Use OpenTSLM's published collator:

```python
from torch.utils.data import DataLoader
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from scripts.opentslm_vitaldb_dataset import VitalDBHypotensionDataset

train = VitalDBHypotensionDataset(
    "train", EOS_TOKEN=model.get_eos_token(), data_dir="data/lstm/top_10"
)
loader = DataLoader(
    train,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=PATCH_SIZE, normalize=False
    ),
)
```

Keep `normalize=False`: the NPZ is already scaled with training-only medians and
IQRs, and each channel description carries those physical-scale statistics.
OpenTSLM pads the 10 source samples on the right to its patch-size multiple; the
prompt distinguishes that padding from source-missing values marked by mask 0.

## Bring the data into TimeNet

This step starts from the challenge-1 selection, not from a hardcoded path.

[`connectors/registry.py`](connectors/registry.py) maps a sourced `dataset_id`
to a reusable TimeNet `BaseConnector`. VitalDB is implemented as
[`connectors/vitaldb/hypotension_windows`](connectors/vitaldb/hypotension_windows):
it standardises numeric signals with units, paired observation-mask channels,
patient/case annotations, and two learning targets documented in
[`tasks.yaml`](connectors/vitaldb/hypotension_windows/tasks.yaml):

- **classification** `hypotension_onset_horizon_v1`: `within_3`, `within_5`,
  `within_10`, `within_15`, `none_within_15`
- **question answering**: the 20-second window plus masks in, a language
  horizon answer out, with an INTERPRET / ANTICIPATE / ACT rationale that does
  not prescribe treatment

```bash
python scripts/source_datasets.py
python scripts/bring_into_timenet.py --num-parameters 5 --max-cases 16
```

`bring_into_timenet.py` will prepare `data/lstm/top_5/` if needed, publish a
local TimeF registry through TimeNet 0.1.0 `run_pipeline`, read it back, audit
masks and patient-disjoint splits, and write
`artifacts/timenet_ingestion/report.json`. Use `--skip-prepare` when the NPZs
already exist, and `--convert-only` to validate the schema without writing a
registry.

The connector is pinned to `timenet[build]==0.1.0` (inspected revision
`c39ca32b64ad0c89ea54093dbcb285c1a93eb006`). Dataset ids are collision-free:
`hackzurich/vitaldb-hypotension-top-5` or
`hackzurich/vitaldb-hypotension-top-10`. TimeNet 0.1.0 publishes to a local
registry; hosted backends are documented as planned.

Lower-level commands remain available:

```bash
python scripts/prepare_lstm_dataset.py 5 --max-cases 16
python scripts/ingest_timenet.py --data-dir data/lstm/top_5 --convert-only
python scripts/ingest_timenet.py \
  --data-dir data/lstm/top_5 \
  --out data/timenet_registry
```

## Tests

The `tests/` directory covers both pipelines, so run each group in its own
environment. With `.venv` active:

```bash
python -m unittest tests.test_dataset_sourcing tests.test_timenet_bridge tests.test_timenet_ingestion tests.test_opentslm_vitaldb_dataset tests.test_tslm_evaluation -v
```

The `tests/test_training_*.py` files import `torch` and the `training` package
and therefore belong to the training environment described in
[`training/README.md`](training/README.md):

```bash
python -m pytest -q tests/test_training_*.py
```

## Relationship to `training/`

The fine-tuning pipeline in `training/` is a different task: it reads a JSONL
manifest of four synchronized channels (ABP, HR, SpO2, EtCO2) with authored
Observation/Rationale/Recommendation targets and complete physical-device
provenance. The corpus prepared here carries generated hypotension-onset labels,
a different set of parameters, and no device identities, so it does not satisfy
that contract and is not consumed by `training/train.py`. See
[`training/README.md`](training/README.md) for what an export would require.

## Train and evaluate a TSLM

Challenge 3 uses the **already prepared**, patient- and case-disjoint NPZ
splits. It does not re-source data or rebuild the TimeNet connector.

[`evaluation/`](evaluation/) trains two models on `train` only and scores
`test` (validation is unused for the reported numbers):

- **Baseline:** ridge classifier on last/mean/std/delta/missingness summaries.
- **TSLM:** ridge head over OpenTSLM-style value+mask patches (`patch_size=2`),
  then decoded to INTERPRET / ANTICIPATE / ACT / Answer text. Inputs never
  include future MAP.

```bash
python scripts/train_evaluate_tslm.py --data-dir data/lstm/top_5
```

The report at `artifacts/tslm_evaluation/report.json` includes the leakage
audit (subject/case disjoint; device IDs are not available in VitalDB so
device-disjoint evaluation is not claimed). Optional neural fine-tune in the
training environment:

```bash
python scripts/finetune_opentslm_hypotension.py --data-dir data/lstm/top_5
```

## Data license

The source files were acquired through the
[VitalDB dataset API](https://vitaldb.net/dataset/) and retain its
[CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The connector records that license as `other` because TimeNet 0.1.0 has no
combined CC BY-NC-SA enum.

## Demonstration

The demo is an OR-style monitor at http://127.0.0.1:8765. Press **Play window**
to record the 20-second vitals strip, then Interpret / Anticipate / Act.
**Evaluation** has a **Before** folder (3 training passes) and an **After**
folder (50 passes, once that run finishes). Train the longer run into its own
directory so the page can show both:

```bash
python -m training.train --config training/config_50_epochs.json \
  --manifest data/surgical_telemetry/manifest.jsonl \
  --output-dir artifacts/run-opentslm-50ep
```

On another machine, pull branch `sid` (it already contains the ~93MB trained
`best.pt` files). Hugging Face base weights stay off GitHub because they are
several gigabytes; download them once with a token:

```bash
git clone -b sid https://github.com/ashwakumar/timelapse.git
cd timelapse
make setup
# put a Hugging Face read token in .env as HF_TOKEN
make hf
make check
make demo
```

Or without Make:

```bash
set -a; source .env; set +a
python scripts/run_demo.py --preload --port 8765
```

Open http://127.0.0.1:8765 — pick **After · 50 training passes**, press
**Play window**, then **Show trained best.pt**. That is the held-out readout
from the trained checkpoint. Teammates do not retrain.
