# Real VitalDB → OpenTSLM → portable evaluation bundle

This is the handoff for a person with a Nebius CUDA GPU machine. These commands
train **new sustained hypotension onset categories**, not numerical future MAP.
They do not use the synthetic `data/surgical_telemetry` examples or the old small
`tslm_head.pt` classifier. No Nebius SDK, account configuration, or deployment
service is required by this repository: run the commands inside the GPU VM.

**Single-command route after cloning:** with Python 3.12 and a working CUDA GPU,
run `bash scripts/run_nebius_vitaldb.sh --epochs 5 --cases 1000 --name vitaldb-run-001`.
It installs dependencies, downloads/prepares real data, validates, trains, and
exports the bundle. Use the identical command with `--resume` after interruption.
The named route writes `data/vitaldb/vitaldb-run-001`,
`artifacts/run-vitaldb-run-001`, and
`artifacts/vitaldb-bundles/vitaldb-run-001.tar.gz`; substitute that name for
`vitaldb-run` in the upload/download examples below. Upload is a separate explicit
step requiring the operator's GitHub credentials. The detailed route follows.

## 1. Pull and install on the GPU machine

The VM needs Python **3.12**, Git, Make, a working NVIDIA driver, and network
access to GitHub, Hugging Face and VitalDB. Run in a persistent terminal such as
tmux so disconnecting SSH does not stop training. Leave substantial disk space
for raw VitalDB tracks, the base model, and two optimizer checkpoints.

```bash
git clone --branch sid https://github.com/ashwakumar/timelapse.git
cd timelapse
# For an existing checkout instead: git switch sid && git pull --ff-only origin sid
nvidia-smi
make vitaldb-setup
.venv-vitaldb/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
```

This installs a dedicated `.venv-vitaldb` containing both data preparation and
training dependencies, without TimeNet's conflicting pins. Do not install the
old `requirements.txt` into this environment. If you already have a compatible
environment, pass `VITALDB_PYTHON=/path/to/python` to every make command.

The configured Hugging Face repositories are public. A token can help avoid
anonymous rate limits; if needed, put it in the gitignored `.env`, then:

```bash
set -a
. ./.env
set +a
```

The backbone and OpenTSLM revisions are pinned in `training/vitaldb_config.json`.
The loader downloads them on first training/evaluation use. It restores the
published pretrained components, then learns new encoder/projector/LoRA weights.
It does not start from the synthetic 50-epoch run. Cache files are never exported.

## 2. Download real data and freeze the split

```bash
make vitaldb-data CASES=1000
make vitaldb-check
```

The first command calls the VitalDB Python library directly. It prepares
`data/vitaldb/onset_v2/manifest.json`, `train.npz`, `validation.npz`, and `test.npz`.
It refuses to overwrite a previous corpus. Reuse it for resume and final export.
Download failures and exclusions are recorded; inspect the manifest/preflight
counts rather than assuming the requested number of cases all produced windows.

The default split is approximately **70% / 15% / 15% of patients**, subject to
rounding and usable windows. It is not a temporal last-15% slice. Each patient's
cases stay together. Validation chooses the checkpoint; test is reserved for the
final run here. Device-unit identities are unavailable and are not invented.

The five input parameters come from the committed feature ranking. Strict
preparation excludes subjects in that exploratory screening cohort from this new
corpus. The inputs are 20 seconds at two-second intervals, with explicit masks
and train-only median/IQR scaling. New labels require a clean preceding minute,
full future observation, and enough extra follow-up to confirm 60 seconds of
low MAP even at the end of the 15-minute horizon. Missing future values cause a
window to be excluded rather than assumed to be a negative event.

Targets are the five canonical answers: `hypotension within 3 minutes`, `within
5`, `within 10`, `within 15`, or `no hypotension within 15 minutes` (all positive
answers include the word `hypotension`). Horizons are exclusive onset buckets:
0–3, >3–5, >5–10, >10–15 minutes. These are code-generated research outcomes.

Preflight rejects legacy or synthetic corpora, overlapping patient/case splits,
empty splits, missing training classes, and splits without both event/no-event
windows. If a small pilot cohort fails coverage, prepare more cases in a **new**
directory. Do not disable the checks or alter labels to get the run to start.

## 3. Train for the requested number of epochs

```bash
make vitaldb-train EPOCHS=5
```

Use `EPOCHS=3` or another positive total if desired. The default is CUDA mixed
precision, microbatch 1, gradient accumulation 16, and rank-8 LoRA. Training
uses inverse-class-frequency sampling; validation retains its natural class
distribution. This makes learning minority classes possible but is not a
guarantee of good detection or calibrated risk probabilities.

The run logs optimizer steps, training loss and validation token loss. It saves
`best.pt`, `last.pt`, metrics, exact dataset hashes, split audit, run configuration,
package versions and backbone provenance in `artifacts/run-vitaldb`.
`best.pt` is selected by minimum **validation** token loss. The training command
does not generate test predictions or choose settings using test performance.

For a separately named pilot run, use `VITALDB_RUN=artifacts/run-vitaldb-pilot`.
Keep the default 20-second history and task unchanged for the first run. Do not
substitute `make demo`, `make eval`, `training.train`, or
`scripts/finetune_opentslm_hypotension.py`: those are older, different paths.

### Resume an interrupted run

```bash
make vitaldb-resume EPOCHS=5
```

Use the **same original total epoch count**, dataset, balancing setting and
optimization configuration. Resume loads optimizer/scheduler/RNG state from
`last.pt` and checks dataset/config identity. It resumes at the next complete
epoch; work since the last completed epoch is repeated. It does not support
extending the original scheduler by changing `EPOCHS`. For a new experiment,
choose a new run directory and start from the pretrained backbone.

## 4. Export and return through GitHub

Do not run the reserved test evaluation on Nebius when the agreed final
evaluation will run locally. Export the validation-selected model instead:

```bash
make vitaldb-export
```

This creates `artifacts/vitaldb-bundles/vitaldb-run/` and
`artifacts/vitaldb-bundles/vitaldb-run.tar.gz` with a `.sha256` sidecar. The bundle
contains only tuned encoder/projector/LoRA weights (no 1B backbone or optimizer),
the exact prepared splits, normalization in the manifest, metrics and hashes.
The data are needed to reproduce the final test and train-only baselines here.
They retain VitalDB's CC BY-NC-SA 4.0 terms; use only as allowed by those terms.

Upload the archive and checksum as **GitHub release assets**, rather than adding
large model/data binaries or any `.env` to source commits. With GitHub CLI and
repository write access, choose a unique run tag and upload:

```bash
RUN_TAG=vitaldb-run-001
TRAIN_COMMIT=$(git rev-parse HEAD)
gh release create "$RUN_TAG" \
  artifacts/vitaldb-bundles/vitaldb-run.tar.gz \
  artifacts/vitaldb-bundles/vitaldb-run.tar.gz.sha256 \
  --target "$TRAIN_COMMIT" --title "$RUN_TAG" \
  --notes 'Real VitalDB OpenTSLM run. Validation-selected weights and exact splits; final test evaluation pending.'
```

Send back the release link, source commit, requested/completed epochs and any
data failures. For files exceeding the service's asset limit, split the archive
and upload every part plus a checksum; communicate the reconstruction command.
No GitHub credentials or Hugging Face token should be included in the bundle.

## 5. Run the final evaluation locally after receiving the bundle

```bash
git pull --ff-only origin sid
make vitaldb-setup
mkdir -p artifacts/vitaldb-bundles
gh release download vitaldb-run-001 --dir artifacts/vitaldb-bundles
cd artifacts/vitaldb-bundles
shasum -a 256 -c vitaldb-run.tar.gz.sha256
tar -xzf vitaldb-run.tar.gz
cd ../..
make vitaldb-eval DEVICE=cpu
```

Use `DEVICE=cuda` on a GPU; the current model does not support Apple MPS.
CPU evaluation of a large test cohort can be slow, especially with five-candidate
scoring. The complete test split must still be evaluated to call it final.
The evaluator verifies bundle checksums, reconstructs the pinned model, and
scores the complete held-out test split. No data redownload or resplitting occurs.
Results are written under `artifacts/vitaldb-evaluation`.

The report distinguishes free-generated answers from optional five-answer
likelihood scoring. Malformed generated answers count as failures. It reports
class counts, patient counts, accuracy, macro F1, class recall, event sensitivity,
false-positive window rate and train-only summary/patch/majority baselines.
Candidate scoring adds event average precision and AUROC; its scores are not
calibrated clinical probabilities. Window metrics do not equal event-level lead
time or false alarms per hour; those require episode matching and alert-policy
evaluation. Low text loss alone is not evidence of successful forecasting.

For a wiring check before opening the test set, use `--split validation` with the
evaluator directly. `--max-samples N` is a diagnostic only and is labeled as such;
do not report it as final test performance. Avoid repeated test tuning.

## Verification before this handoff

Offline tests exercise actual tiny Llama + PEFT training, changing trainable
weights while freezing the backbone, deterministic resume, task/mask contracts,
export/checksums, labeling boundaries, split isolation, and evaluator behavior.
These are software checks. A complete Nebius CUDA training run and its eventual
predictive performance must still be measured by the GPU operator.
