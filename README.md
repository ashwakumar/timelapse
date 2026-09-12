# timelapse

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

## TimeNet connector

[`connectors/vitaldb/hypotension_windows`](connectors/vitaldb/hypotension_windows)
implements the released [TimeNet](https://github.com/OpenTSLM/TimeNet)
`BaseConnector` contract for prepared top-N corpora (1–17 current dynamic signals;
top-5 and top-10 verified with real data). The dependency
is pinned to `timenet[build]==0.1.0`; the inspected upstream revision was
`c39ca32b64ad0c89ea54093dbcb285c1a93eb006`.

Check the schema without writing anything, then build a local TimeF registry and
read it back through the official client:

```bash
python scripts/ingest_timenet.py --data-dir data/lstm/top_10 --convert-only
python scripts/ingest_timenet.py \
  --data-dir data/lstm/top_10 \
  --out data/timenet_registry
```

Add `--force` to replace an existing registry. Each configuration has a
collision-free ID: `hackzurich/vitaldb-hypotension-top-5` or
`hackzurich/vitaldb-hypotension-top-10`. The readback audit checks record and
task counts, exact agreement between nullable signal values and masks, and
patient-disjoint splits.

TimeNet 0.1.0 publishes to a local registry; its official documentation says a
hosted backend is planned.

## Tests

The `tests/` directory covers both pipelines, so run each group in its own
environment. With `.venv` active:

```bash
python -m unittest tests.test_timenet_ingestion tests.test_opentslm_vitaldb_dataset -v
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

## Data license

The source files were acquired through the
[VitalDB dataset API](https://vitaldb.net/dataset/) and retain its
[CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The connector records that license as `other` because TimeNet 0.1.0 has no
combined CC BY-NC-SA enum.
