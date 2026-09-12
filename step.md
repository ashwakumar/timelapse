# AeroGuard — project steps

**Next task: Step 4 — make the development setup reproducible.**

Work through one step at a time. Finish its verification before moving on. This roadmap describes work to do; commands for features that do not exist yet are intentionally omitted.

The first complete version should predict remaining engine life from sensor windows, report a RUL band, summarize observed sensor changes, and show measured performance in a working dashboard. Specific component diagnoses and maintenance prescriptions require additional validated labels.

## Progress overview

| Step | Work | Status |
| --- | --- | --- |
| 1 | Download and validate raw data | Done |
| 2 | Preprocess, split, and define targets | Done: deterministic version |
| 3 | Fix data loading and normalization | Done |
| 4 | Make development setup reproducible | Next |
| 5 | Finish TimeNet connector and export | Partial |
| 6 | Decide and implement agentic annotations | Decision pending; optional for first working model |
| 7 | Establish real baseline measurements | Partial |
| 8 | Verify model architecture and training inputs | Partial |
| 9 | Fine-tune with validation | Smoke-run implementation only |
| 10 | Load saved models and run inference | Partial |
| 11 | Evaluate the frozen model on held-out engines | Pending |
| 12 | Connect and repair the dashboard | Partial |
| 13 | Package the project and prepare the demo | Pending |

## 1. Download and validate raw data — done

**File:** [scripts/download_data.py](scripts/download_data.py)

- [x] Keep one raw file: `data/raw/train_FD001.txt`.
- [x] Download only when the raw file is absent.
- [x] Validate 26 finite numeric columns, engine IDs, ordering, and continuous cycles.
- [x] Validate downloads before accepting them into the cache.
- [x] Record the download URL and checksum when a fresh download occurs.

```bash
uv run --no-sync python -m scripts.download_data
```

**Verified:** the existing file has 20,631 rows across 100 engines. Its original mirror URL and license provenance still need verification before distribution; structural validation does not authenticate the source.

## 2. Preprocess and define learning targets — done

**File:** [scripts/preprocess_data.py](scripts/preprocess_data.py)

- [x] Select 14 sensor channels.
- [x] Create 30-cycle windows with stride 5 and include each terminal window once.
- [x] Split by engine, keeping every engine in exactly one split.
- [x] Define numeric RUL, synthetic RUL bands, and sensor-observation text.
- [x] Mark component diagnosis and maintenance action as unlabeled.
- [x] Save a manifest with configuration, split assignments, counts, and hashes.
- [x] Keep one processed dataset, without duplicate version directories.

| Split | Engines | Windows |
| --- | --- | ---: |
| Training | 1–70 | 2,502 |
| Validation | 71–80 | 355 |
| Test | 81–100 | 806 |

```bash
uv run --no-sync python -m scripts.preprocess_data
uv run --no-sync python -m pytest scripts/tests/test_data_pipeline.py -q
```

**Output:** `data/processed/windows.jsonl` and `dataset_manifest.json`.

**Verified:** 28 data tests passed at the latest check. See [data/README.md](data/README.md) for the contract. This stage is currently rule-based, not agentic. These splits come from `train_FD001.txt`; they are not the official C-MAPSS test protocol.

## 3. Fix data loading and normalization — done

**File:** [training/dataset_loader.py](training/dataset_loader.py)

The loader indexes JSONL byte offsets, fits population statistics on training windows only, and requires the saved statistics for validation/test loading. Overlapping window samples count each time they appear. Constant or near-constant channels (standard deviation below 1e-6) use scale 1.

- [x] Fit channel means and standard deviations using training engines only.
- [x] Specify whether statistics count overlapping window samples or unique raw observations; use the same policy consistently.
- [x] Save normalization statistics, channel order, window size, and dataset hash as model preprocessing metadata.
- [x] Reuse those exact statistics for validation, test, and inference; never refit on those inputs.
- [x] Handle zero-variance channels and reject invalid channel/window shapes explicitly.
- [x] Use lazy record access instead of loading every sensor window into memory.
- [x] Return sensor tensors, token IDs, attention masks, and correctly masked training labels.
- [x] Keep answer labels separate from prediction inputs.
- [x] Test that changing test values cannot change normalization statistics.

**Done when:** all three splits load correctly, a sample has shape `[14, 30]`, training-only statistics survive save/reload, and automated checks catch preprocessing leakage.

**Verified:** all 3,663 windows loaded with finite tensors; the local tokenizer needed at most 186 tokens for training examples within the 256-token limit. Normalization is saved in `models/preprocessing.json`. It belongs to the current data, not the old model adapters. Loader regression tests cover leakage, save/reload, masking, and lazy reads. Training now forwards attention masks and saves preprocessing beside new checkpoints.

## 4. Make development setup reproducible

**Files:** `pyproject.toml`, `uv.lock`, package configuration, and a new `Makefile`/root `.gitignore` as needed.

- [ ] Verify `uv sync --locked` in a clean environment.
- [ ] Make imports resolve to workspace source, including the local C-MAPSS connector.
- [ ] Configure formatting, linting, type checking, and meaningful tests.
- [ ] Implement `make check`, as required by `AGENT.md`; it currently has no target.
- [ ] Ignore virtual environments, Python caches, and disposable run outputs.
- [ ] Choose an explicit policy for tracking raw data, processed data, and model weights.
- [ ] Keep dependencies and the lockfile synchronized.

**Done when:** a fresh checkout can install, import the intended source, and run the documented checks without relying on manually copied packages in `.venv`.

## 5. Finish the TimeNet connector and export

**Files:** [connector directory](packages/timenet-connectors/src/timenet_connectors/datasets/nasa/cmapss/), [scripts/build_timef_registry.py](scripts/build_timef_registry.py).

- [ ] Update dataset metadata to describe 14 selected channels and the current targets.
- [ ] Verify sensor units and dataset provenance instead of inheriting unsupported metadata claims.
- [ ] Preserve train/validation/test annotations and label provenance in TimeF.
- [ ] Use lazy/streaming conversion in line with the repository instructions.
- [ ] Make builder discovery use the workspace connector.
- [ ] Export to `artifacts/registry/` and verify counts, values, tasks, and split membership after reload.
- [ ] Decide whether training should consume TimeF directly; if required for the project submission, connect it to the loader and check parity with JSONL.

**Done when:** the canonical dataset survives export/reload without changing values, labels, or splits, and fixture tests exercise the workspace connector.

TimeNet export is optional for the current JSONL training path, but it is a deliverable in the original project specification.

## 6. Decide whether to add agentic annotations

**Current file:** [scripts/preprocess_data.py](scripts/preprocess_data.py). A separate annotation module would be added only if needed.

First decide whether the first release uses deterministic observation templates or includes an LLM agent that drafts annotations. Do not call the existing templates an agent.

If adding an agent:

- [ ] Define its narrow role: drafting explanations from observed sensor features and reviewed reference material.
- [ ] Keep numeric RUL, split assignment, and validation under deterministic code.
- [ ] Define structured outputs, allowed claims, missing-evidence behavior, and rejection rules.
- [ ] Record model/prompt versions and annotation provenance; cache outputs and control retries/cost.
- [ ] Keep held-out answers unavailable to the agent during evaluation.
- [ ] Review a sample for unsupported fault diagnoses, contradictions, and misleading maintenance instructions.
- [ ] Mark generated annotations as synthetic; do not use them as independent proof of diagnosis accuracy.

**Done when:** the decision is recorded and, if implemented, annotations pass defined validation and review. Freeze the target format before the main fine-tuning run. Retaining templates is acceptable for the first working RUL model.

## 7. Establish real baseline measurements

**File:** [training/evaluate_baselines.py](training/evaluate_baselines.py)

- [ ] Remove ground-truth-plus-noise predictions and hardcoded qualitative conclusions.
- [ ] Add a simple training-derived reference predictor and retain the real XGBoost regressor.
- [ ] Train on engines 1–70 and use engines 71–80 for model selection.
- [ ] If comparing against a text-only LLM, actually run it on observed sensor summaries without labels.
- [ ] Define fair input access: include current cycle consistently or explicitly report differences between models.
- [ ] Save per-record predictions, configurations, and data hashes under `artifacts/`.
- [ ] Report RMSE, MAE, and the implemented asymmetric RUL score; count text parsing failures explicitly.

**Done when:** validation metrics are calculated from actual predictions and can be reproduced. Reserve the test split for the final comparison after configurations are fixed.

## 8. Verify the model and training inputs

**File:** [training/train_opentslm.py](training/train_opentslm.py)

The current implementation fine-tunes SmolLM-135M-Instruct with LoRA while training a temporal encoder and scalar RUL head from scratch. It is a custom implementation rather than an imported OpenTSLM package.

- [ ] Decide whether to retain this implementation or integrate the intended OpenTSLM architecture; document the choice.
- [ ] Confirm only LoRA adapters, the temporal encoder, and the RUL head receive updates.
- [ ] Check sensor patch shapes, embedding dimensions, and temporal information handling.
- [ ] Fix padding/attention masks and ensure truncation does not discard the intended response targets.
- [ ] Handle CPU, CUDA, and MPS tensor placement consistently for supported devices.
- [ ] Check language and RUL losses separately, including their relative scale.
- [ ] Confirm the generated RUL text and separate numeric head can be checked for consistency.
- [ ] Run a tiny-batch training check to verify the model can learn and produce finite gradients.

**Done when:** the training path uses the intended inputs/targets, updates the intended parameters, and completes a small verified run.

## 9. Fine-tune with validation

**File:** [training/train_opentslm.py](training/train_opentslm.py)

- [ ] Start a new run using the current split. Existing adapters used the earlier training split, which included the current validation engines.
- [ ] Save seeds, backbone revision, optimizer settings, loss weights, data hash, and preprocessing configuration.
- [ ] Use a short smoke run first, then a justified training budget on available hardware.
- [ ] Track training losses and validation RUL/text metrics.
- [ ] Select checkpoints using validation results and a documented stopping rule.
- [ ] Save the selected adapters, temporal encoder, RUL head, tokenizer, and preprocessing metadata together under `models/<run>/`.

**Done when:** a selected checkpoint has a reproducible configuration and measured validation results. The existing 15-step default is only a smoke run, not evidence of useful accuracy.

## 10. Add a complete inference path

**Files:** model code plus a small inference module/command to be added.

- [ ] Reload the backbone, adapters, temporal encoder, scalar head, tokenizer, and saved normalization.
- [ ] Accept one valid sensor window and observed metadata without any ground-truth label.
- [ ] Apply training preprocessing, predict RUL, and generate an assessment.
- [ ] Return a structured result with predicted RUL, derived band, generated text, and model identifier.
- [ ] Define how to handle invalid numeric outputs and disagreement between text and the scalar prediction.
- [ ] Check predictions before and after checkpoint reload agree within a stated numerical tolerance.
- [ ] Record latency on the actual target hardware; do not invent confidence intervals.

**Done when:** a fresh process can produce a prediction from sensor inputs and the saved checkpoint alone.

## 11. Run the final held-out evaluation

**File:** [training/evaluate_baselines.py](training/evaluate_baselines.py), using the inference path from Step 10.

- [ ] Freeze model/preprocessing configurations selected on validation data.
- [ ] Evaluate on engines 81–100 with labels used only for scoring.
- [ ] Use identical record sets and documented input access across comparisons.
- [ ] Save actual predictions and calculate RMSE, MAE, and the asymmetric RUL score.
- [ ] Report window-level metrics and per-engine results; distinguish all-window evaluation from terminal-window-only evaluation.
- [ ] Measure status agreement, sensor-observation correctness, and text/numeric RUL consistency.
- [ ] Include failure cases and text-generation/parsing failures, not only favorable examples.
- [ ] State that template-text agreement does not validate component diagnosis or maintenance advice.

**Done when:** every reported metric is traceable to a saved prediction, target, dataset version, and model configuration. Results remain explicitly tied to the internal engine holdout protocol.

## 12. Connect and repair the dashboard

**File:** [demo/app.py](demo/app.py)

- [ ] Fix the Plotly `YAxis.titlefont` rendering error.
- [ ] Load the selected model once and call real inference when the assessment button is pressed.
- [ ] Keep actual RUL visibly separate from predicted RUL in demonstration examples.
- [ ] Remove fixed confidence/error values and unsupported dispatch instructions.
- [ ] Display the real backbone, channel count, dataset version, and benchmark provenance.
- [ ] Load real benchmark artifacts; show an unavailable state when they are missing.
- [ ] Check engine/cycle selection, sensor toggles, loading states, and inference errors.
- [ ] Label any retained scripted example explicitly as a demonstration.

**Done when:** the dashboard renders, predictions originate from the selected checkpoint, and all visible performance claims match measured artifacts.

## 13. Package the project and prepare the demo

**Files:** `README.md`, `data/README.md`, `AGENT.md`, this roadmap, and selected model/result artifacts.

- [ ] Update documentation to match the final architecture, commands, and measurements.
- [ ] Resolve source attribution and licensing for distributed data, code, and model artifacts.
- [ ] Keep `data/` minimal; store model runs under `models/` and generated reports/registry under `artifacts/`.
- [ ] Reconcile stale paths and architectural claims in the original agent/spec documents.
- [ ] Verify setup, data preparation, model loading, and demo launch in a clean environment.
- [ ] Run configured checks and relevant end-to-end tests.
- [ ] Prepare a short demo: select an unseen engine, inspect telemetry, run inference, compare with ground truth, and show honest baseline results.
- [ ] Present limitations and representative failure cases alongside achievements.

**Done when:** another person can reproduce the documented workflow and understand what the model actually learns, what was measured, and what remains unsupported.

## Immediate next working session

Start with Step 4 only: make clean installation, workspace imports, and `make check` reproducible. Step 3 is complete; full model training and inference validation remain later steps.
