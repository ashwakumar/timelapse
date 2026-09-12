# AeroGuard data

```text
data/
├── README.md
├── raw/
│   └── train_FD001.txt
└── processed/
    ├── windows.jsonl
    └── dataset_manifest.json
```

Keep one raw source and one processed dataset. Model weights live under `models/`; optional TimeF registry and benchmark outputs are generated under `artifacts/`. Older processed copies, the stale registry, and the simulated benchmark file have been removed.

## 1. Download

```bash
uv run --no-sync python -m scripts.download_data
```

Downloads `train_FD001.txt` only when absent and validates cached or downloaded data. Validation requires 26 finite numeric columns, integer engine IDs 1–100, ascending engine order, and contiguous cycles starting at 1. Invalid cached data raises an error rather than silently being replaced.

A fresh download also writes `raw/train_FD001.source.json` with the mirror URL and checksum. The existing cache has no verified mirror receipt, so its source URL remains null in the processed manifest.

## 2. Preprocess

```bash
uv run --no-sync python -m scripts.preprocess_data
```

Reads local raw data, validates it, and writes `processed/windows.jsonl` and `processed/dataset_manifest.json`. It never downloads data. Optional flags are `--raw-dir`, `--output-dir`, `--window-size`, and `--stride`.

Preprocessing currently uses deterministic rules and templates. It does not invoke an LLM agent. Agent-generated annotations would be a separate enhancement; numerical labels and split membership should remain reproducible.

The pipeline scans telemetry to collect engine lifetimes, then emits windows using a bounded buffer. It does not load all sensor histories or output records into memory.

## Data contract

- Source: 20,631 rows from `train_FD001.txt`.
- Channels: sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.
- Window: 30 observed cycles, stride 5, plus each engine's terminal window exactly once.
- No normalization, imputation, padding, or RUL cap during preprocessing.
- Each engine belongs to exactly one split.

| Split | Engine IDs | Windows |
| --- | --- | ---: |
| Train | 1–70 | 2,502 |
| Validation | 71–80 | 355 |
| Test | 81–100 | 806 |
| Total | 1–100 | 3,663 |

This is an internal engine holdout from the training file, not the official C-MAPSS test protocol. Existing model adapters were created with the earlier split, which included engines 71–80 in training. They cannot provide a clean validation result on this dataset.

## Questions and targets

The prompt asks for RUL, a RUL band, and a summary of observed sensor changes. It includes current cycle and window length, but no engine ID or answer. Engine ID remains record metadata for split auditing.

| Field | Definition |
| --- | --- |
| `rul` | Final recorded cycle minus window end; includes zero at the terminal window. |
| `status` | Synthetic RUL band: CRITICAL for 0–30, WARNING for 31–75, NORMAL above 75. |
| `rationale` | Observed last-minus-first differences in sensors 4, 11, and 15, expressed through a template. |
| `target` | RUL and status text; component diagnosis and maintenance action are marked as not labeled. |
| `label_provenance` | How each label was obtained; unsupported diagnosis/action annotations are null. |

Use `series` and `prompt` as prediction inputs. Keep labels such as `rul`, `status`, and `target` out of evaluation inputs. RUL bands are not independently annotated health diagnoses, and sensor differences alone do not establish a specific component fault.

The manifest records schema version 2, raw/output hashes, hashes of both preparation modules, window settings, channel order, split engine IDs, label definitions, and counts. Versioning is recorded in metadata rather than duplicate folders. The source license remains unverified; hashes identify local artifacts but do not authenticate them against NASA.

## 3. Load for training

`training/dataset_loader.py` scans the canonical JSONL to validate/index records, then reads individual windows by byte offset. It stores offsets rather than all sensor arrays and works with PyTorch DataLoader batching.

Normalization uses the population mean/standard deviation of **training windows only**, counting repeated observations each time they occur in overlapping windows. Channels with standard deviation below `1e-6` use scale 1. The saved metadata includes channel order, window length, fit policy, sample count, training engine IDs, and full/training data hashes.

```bash
uv run --no-sync python -m training.dataset_loader --save-normalization models/preprocessing.json
uv run --no-sync python -m pytest training/tests/test_dataset_loader.py -q
```

```python
from training.dataset_loader import CMAPSSCoTDataset, NormalizationStats, prepare_inputs

train = CMAPSSCoTDataset(split="train")
train.normalization.save("models/preprocessing.json")

validation = CMAPSSCoTDataset(
    split="validation", normalization="models/preprocessing.json"
)
test = CMAPSSCoTDataset(split="test", normalization="models/preprocessing.json")

# Training samples contain labels. Held-out samples omit them by default.
# Use include_targets=True explicitly for validation loss/scoring, not generation.
# Passing tokenizer=... adds input_ids, attention_mask, and supervised labels
# when include_targets=True. Overflow raises instead of truncating the answer.

stats = NormalizationStats.load("models/preprocessing.json")
# raw_series and observation_prompt come from the observed sensor window.
# prediction_inputs = prepare_inputs(raw_series, observation_prompt, stats)
```

Validation/test loading without training statistics fails explicitly. A saved artifact must match the canonical dataset hash, channel order, and window size; it is never silently refitted. For a new unlabeled inference window, `prepare_inputs()` applies the saved transform without needing the training file or any answer labels.

Supervised labels mask the prompt and padding with `-100`; attention masks distinguish padding from actual EOS tokens even when they use the same token ID. The model prepends visible positions for temporal tokens. New training runs save `preprocessing.json` beside their checkpoints.

`models/preprocessing.json` is a standalone artifact for the current dataset. It does not make the older adapters compatible with the new preprocessing. The next project step is reproducible environment and package setup.

The separate `scripts/build_timef_registry.py` is an optional TimeNet export step, not required by the current training loader.

## Tests

```bash
uv run --no-sync python -m pytest scripts/tests/test_data_pipeline.py -q
```

Offline tests cover data integrity, download validation, stage separation, engine splits, terminal windows, label boundaries, checksums, and deterministic regeneration.
