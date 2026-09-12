# timelapse

Exploratory VitalDB pipeline for predicting the onset of sustained
intraoperative hypotension from a 20-second observation window.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The package imported by the scripts is `vitaldb` (not `vitaldba`).

## Feature screen

```bash
.venv/bin/python scripts/feature_screen.py --max-cases 40
```

The command downloads and caches selected public VitalDB tracks, creates
patient-disjoint development and validation splits, derives a deterministic
target (MAP below 65 mmHg for at least 60 seconds within 15 minutes), and
writes:

- `artifacts/feature_screen/parameter_ranking.csv`
- `artifacts/feature_screen/reliable_parameter_ranking.csv`
- `artifacts/feature_screen/group_ablation.csv`
- `artifacts/feature_screen/top20_feature_heatmap.png`
- `artifacts/feature_screen/report.json`

The heatmap shows Spearman relationships and redundancy. Parameter selection
uses grouped permutation loss in validation AUPRC. This is an exploratory
screen; the validation patients used for selection are not a final test set.

## LSTM dataset

The positional argument always selects the best `N` dynamic parameters from
the saved reliable ranking. It does not accept an arbitrary feature list.

```bash
.venv/bin/python scripts/prepare_lstm_dataset.py 5 --max-cases 40
.venv/bin/python scripts/prepare_lstm_dataset.py 10 --max-cases 40
```

Each run writes patient-disjoint `train.npz`, `validation.npz`, and `test.npz`
files under `data/lstm/top_N/`, plus a `manifest.json` containing the exact
parameter order, labels, split counts, availability, and train-only scaling.
Every NPZ contains normalized `x_values`, a separate `x_mask`, multiclass `y`,
and case, subject, cutoff, and future-coverage metadata.

The target is generated with code: first onset of MAP below 65 mmHg for at
least 60 contiguous observed seconds, bucketed into 3, 5, 10, 15, or no event
within 15 minutes. A fully observed preceding minute at or above 65 is required
so the task predicts a new onset rather than continuation of an existing event.
