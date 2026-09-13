#!/usr/bin/env bash
# Complete real-data GPU handoff. Run inside an already provisioned CUDA VM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

task_epochs=5
task_cases=1000
task_name=vitaldb-run-001
task_resume=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) task_epochs="${2:?Missing epoch count}"; shift 2 ;;
    --cases) task_cases="${2:?Missing case count}"; shift 2 ;;
    --name) task_name="${2:?Missing run name}"; shift 2 ;;
    --resume) task_resume=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--epochs 5] [--cases 1000] [--name vitaldb-run-001] [--resume]"
      echo "Requires Python 3.12, Git, Make and working CUDA. Resume keeps original epochs/data/config."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$task_epochs" =~ ^[1-9][0-9]*$ && "$task_cases" =~ ^[1-9][0-9]*$ ]] || {
  echo "Epochs and cases must be positive integers" >&2; exit 2;
}
[[ "$task_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || {
  echo "Run name must contain only letters, numbers, underscores or hyphens" >&2; exit 2;
}

task_data="data/vitaldb/$task_name"
task_run="artifacts/run-$task_name"
task_bundle="artifacts/vitaldb-bundles/$task_name"
if [[ -e "$task_bundle" || -e "$task_bundle.tar.gz" ]]; then
  echo "Export already exists; choose a new --name" >&2; exit 1
fi
if [[ "$task_resume" == true && ! -f "$task_run/last.pt" ]]; then
  echo "No completed epoch checkpoint exists at $task_run/last.pt" >&2; exit 1
fi
if [[ -f .env ]]; then
  set -a
  # The operator's optional Hugging Face token is never printed or exported.
  . ./.env
  set +a
fi
make vitaldb-setup
.venv-vitaldb/bin/python -c 'import torch; assert torch.cuda.is_available(), "A working CUDA GPU is required"; print("CUDA:", torch.cuda.get_device_name(0))'
task_make_args=("VITALDB_DATA=$task_data" "VITALDB_RUN=$task_run" "VITALDB_BUNDLE=$task_bundle" "EPOCHS=$task_epochs" "CASES=$task_cases" "DEVICE=cuda")
if [[ ! -f "$task_data/manifest.json" ]]; then
  if [[ "$task_resume" == true ]]; then
    echo "Resume requires the original prepared data at $task_data" >&2; exit 1
  fi
  make vitaldb-data "${task_make_args[@]}"
fi
make vitaldb-check "${task_make_args[@]}"
if [[ "$task_resume" == true ]]; then
  make vitaldb-resume "${task_make_args[@]}"
else
  make vitaldb-train "${task_make_args[@]}"
fi
make vitaldb-export "${task_make_args[@]}"
echo "Training and export complete. Return these GitHub release assets:"
echo "$task_bundle.tar.gz"
echo "$task_bundle.tar.gz.sha256"
echo "The reserved test split has not been evaluated. See NEBIUS_TRAINING.md for upload/local evaluation."
