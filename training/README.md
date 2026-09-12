# Surgical telemetry fine-tuning

Train an OpenTSLM SoftPrompt encoder/projector and a PEFT LoRA language adapter
on synchronized ABP, HR, SpO2, and EtCO2 histories plus a clinical question.
Targets are supplied, authored **Observation → Rationale → Recommendation**
explanations. The pipeline does not generate clinical ground truth.

## Installation

Use a separate Python 3.12 environment from the TimeNet connector:

```bash
cd timelapse
python3.12 -m venv /tmp/telemetry-training-env
source /tmp/telemetry-training-env/bin/activate
python -m pip install -r requirements-training.txt
hf auth login
```

Downloads need a Hugging Face token, either from `hf auth login` or from
`HF_TOKEN` in the environment; copy `.env.example` to `.env` and load it with
`set -a; source .env; set +a` to use the latter.

The dataclass defaults resolve the base model to the gated
`meta-llama/Llama-3.2-1B`; the supplied configuration names an ungated mirror of
the same weights instead, as described under Run. OpenTSLM contains an
encoder/projector checkpoint, not a standalone `AutoModelForCausalLM` model. The training loader uses that architecture and
checkpoint together with the matching base language model. The OpenTSLM source
is pinned to commit `2968f4b891baab4307f7e9d0043e87677b593a30`.

## Split guarantees and limits

The default 80/20 split is a `GroupShuffleSplit` over connected components:
records linked by a patient, case, physical device, recording, or signal-file
identity stay together, including transitive links. This is stronger than
grouping `(patient_id, device_id)` tuples, which could leak both identities.
The 20% is a fraction of components; the actual fraction of windows can differ
when groups have unequal sizes. The split audit records the achieved sizes.

Missing required provenance fails validation. If all patients share the same
physical device, a device-disjoint split cannot be made; provide independent
devices/data. A device family or model name is not a physical device identity.
Case-only grouping requires an explicit assumption that cases cannot belong
to the same patient. Disabling the device requirement cannot establish a
device-disjoint evaluation.

Signals cover a half-open observation interval ending no later than the query.
Whole patients/cases/recordings remain together, so overlapping histories from
a known source cannot cross splits. Random grouped splitting does **not**
establish global chronological separation between unrelated cases. For a
prospective evaluation, build a separate time-based cohort with a purge gap and
the same identity exclusions. All guarantees depend on complete, accurate
source metadata and input texts that contain only information available at the
query time. Normalization uses only the retained training histories; targets
are never used to calculate it.

The held-out 20% is used for validation/checkpoint selection, so it is **not an
untouched final test cohort**. Report performance on another patient/device
cohort before making claims about clinical usefulness. Validation language-model
loss measures target-text fit, not treatment correctness.

## Alignment with the existing connector

The concurrent TimeNet connector and the existing
`scripts/opentslm_vitaldb_dataset.py` adapter own the prepared
`data/lstm/top_5` and `top_10` corpus. They use a 20-second history sampled every
2 seconds, with `cutoff_sec` as the first excluded sample. Their generated
hypotension-onset labels and `INTERPRET/ANTICIPATE/ACT` explanations are a
different task from authored clinical Observation/Rationale/Recommendation.

That corpus includes MAP, remifentanil, respiration, and BIS features; top-10
also includes EtCO2. It does not supply the requested four-channel contract.
MAP is not silently renamed ABP, and absent HR/SpO2 channels are not invented.
`subject_id` identifies patients and `case_id` identifies cases, but physical
device-unit IDs are unavailable. Therefore the existing corpus cannot satisfy
the strict default device-disjoint training contract.

Both paths use OpenTSLM's `pre_prompt`, `time_series_text`, `time_series`,
`post_prompt`, and `answer` vocabulary and preserve explicit observation masks.
To prepare this training task from a connector, export the four synchronized
raw channels in declared units, recover complete patient/physical-device
provenance, and supply reviewed target explanations. Export before
normalization, then let this pipeline split and fit its training statistics.
Do not recombine previously selected validation/test cohorts and call them an
independent new test set.

## Reference implementation

- [OpenTSLM source](https://github.com/OpenTSLM/OpenTSLM/tree/2968f4b891baab4307f7e9d0043e87677b593a30)
- [Default SP checkpoint](https://huggingface.co/OpenTSLM/llama-3.2-1b-tsqa-sp)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

Connector coordination was verified against commit `7b74133` on
`codex/vitaldb-lstm-dataset`. Its TimeNet registry is an ingestion artifact;
this trainer reads the manifest described below, not that registry directly.

## Manifest format

Each JSONL line describes one history. Paths are relative to the manifest:

```json
{"sample_id":"p01-window01","patient_id":"p01","case_id":"case01","device_id":"physical-monitor-17","recording_id":"recording01","signal_path":"p01-window01.npy","channels":["ABP","HR","SpO2","EtCO2"],"sample_rate_hz":1.0,"time_base":"seconds_since_case_start","window_start":0.0,"window_end":300.0,"query_time":300.0,"prompt":"Analyze the last 5 minutes of vitals and provide an assessment and recommendation.","target":{"Observation":"Reviewed observations from this history.","Rationale":"Reviewed supporting interpretation, with uncertainty.","Recommendation":"Reviewed next steps for clinical decision support."}}
```

The `.npy` file must contain a numeric array `[4, T]` in exactly that channel
order. Units are ABP mmHg, HR beats/min, SpO2 percent (0–100), EtCO2 mmHg;
convert other units before export. ABP's measurement type (waveform or a
particular pressure summary) must be consistent across the corpus and declared
in clinical context. The example is a software fixture, not a validated ABP
waveform or a labeled clinical example.

Channels must be synchronized at the declared common sampling rate; acquisition,
antialiasing, resampling, and unit conversion belong in data preparation. Do
not resample the full recording with future-aware interpolation before cutting
windows. The trainer does not invent missing channels or infer sample rates.
Missing observations are NaN, not zero; their masks are retained. Every channel
needs observed training values for normalization. Timestamp units are seconds,
and duration must match `T / sample_rate_hz`. Use complete and consistent
patient/case/recording identities across repeated windows and exports.

`max_signal_length` is a sample count, not seconds. At 1 Hz, 300 samples are
five minutes; at 100 Hz, five minutes are 30,000 samples. The pretrained encoder
supports at most 1,024 patches of size 4. Longer data needs an explicitly chosen
causal aggregation or a shorter retained history. The dataset retains the most
recent samples and records the retained duration in model context. Padding is
right-sided to a patch-size multiple; an explicit observation-mask series
accompanies each channel.

For reproducible new runs, replace mutable base-model revisions with the
resolved SHA recorded by the first run and retain the manifest, signals,
configuration, split audit, and normalization statistics together.

## Run

From the `timelapse` directory, first exercise the input pipeline without model
downloads. These commands create fresh directories and refuse to overwrite them:

```bash
python -m training.make_example /tmp/telemetry-example
python -m training.validate_data /tmp/telemetry-example/manifest.jsonl \
  --output-dir /tmp/telemetry-audit --max-signal-length 300
```

`training.train` reads `training/example_config.json` when `--config` is not
given, so a run that uses the supplied small-compute configuration needs no
arguments at all:

```bash
python -m training.make_example data/surgical_telemetry
python -m training.train
```

That configuration writes to `artifacts/opentslm_surgical_telemetry`. Because a
new run requires an empty output directory, override the paths to keep several
runs side by side:

```bash
python -m training.train --config training/example_config.json \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/new-training-run
```

An explicit `--config` path that does not exist is an error; only the default
path may be absent, in which case the dataclass defaults apply.

`base_model_id` selects the base language model. When it is null the repository
is inferred from `model_id`, which resolves to the gated
`meta-llama/Llama-3.2-1B`. The supplied configuration instead names
`NousResearch/Llama-3.2-1B`, an ungated mirror of the same weights, so the run
works without a Meta access grant; the Llama 3.2 Community License still
applies. Override it with `--base-model-id`. The value must name the same
architecture and hidden size the checkpoint's projector was trained against, or
loading fails with an explicit dimension mismatch.

`example_config.json` uses 300 retained samples, batch size 1, accumulation over
8 batches, rank-8 LoRA, gradient checkpointing, AdamW, and cosine decay with
warmup. Adjust `max_signal_length` to your sampling rate and chosen history.
`max_context_tokens` includes text **and** signal/mask patches. Targets that do
not fit are rejected; prompt truncation is off by default. The base model is
frozen; the time-series encoder, projector, and LoRA weights are optimized.

On CUDA, `precision: "auto"` selects bf16 when supported and otherwise fp16
with gradient scaling. Use `device: "cpu", precision: "fp32"` for an offline
smoke run; this is not a claim that full 1B training is fast on CPU. The default
pretrained path supports CUDA and CPU, not MPS. Gemma SP checkpoints must have
the matching base-model repository and architecture; changing only a model name
does not convert a Llama adapter into a Gemma adapter.

The output includes `split_audit.json`, `normalization.json`,
`resolved_config.json`, `metrics.json`, `last.pt`, `best.pt`, and epoch
checkpoints. Checkpoints retain LoRA and encoder/projector state plus optimizer,
scheduler, scaler, and reproducibility state. Keep the run directory together.
`best.pt` is selected using validation loss. Resume at an epoch boundary:

```bash
python -m training.train --config training/example_config.json \
  --manifest /path/to/manifest.jsonl --output-dir /path/to/existing-training-run \
  --resume-from /path/to/existing-training-run/last.pt
```

Resume with the original experiment configuration and planned total epochs;
changing the schedule is a new experiment. For generation,
`training.model.load_for_inference(checkpoint, device)` reloads the saved
backbone revisions and trainable components. Reuse the adjacent normalization
statistics and sampling/channel contract; `model.generate(...)` does not use
`answer` as input.

## Verification

```bash
python -m pytest -q tests/test_training_*.py
```

Tests require no Hub credentials or model downloads. They cover identity
bridges and impossible splits, future-window rejection, train-only scaling,
mask/padding behavior, supervision masks, real tiny Hugging Face Llama + PEFT
training through the CLI, partial gradient accumulation, and checkpoint
restoration. Tiny models and synthetic histories test software correctness;
they do not establish clinical performance.

Each run also stores `dataset_fingerprint.json` (SHA-256 of the manifest and
all signal files) and `environment.json` (Python and core package versions).
Resume rejects changed inputs, missing provenance artifacts, different library
versions, or altered optimization settings. New runs require an empty output
directory. The offline integration test confirms that resuming from epoch 1
reproduces the uninterrupted epoch-2 trainable weights exactly on CPU; floating
point results need not be identical across different hardware.
