#!/usr/bin/env bash
# ==============================================================================
# AeroGuard TSLM: Full End-to-End Preprocessing & Training Pipeline
#
# Runs:
# 1. Agentic Data Sourcing & Raw Telemetry Validation (FD001 C-MAPSS)
# 2. Window Generation & Engine Partitioning (Train: 1-70, Val: 71-80, Test: 81-100)
# 3. Aerospace Engineering Chain-of-Thought (CoT) Diagnostic Synthesis
# 4. Normalization Statistics Pre-computation & Integrity Verification
# 5. OpenTSLM Multi-Task Fine-Tuning (SmolLM-135M + LoRA + Temporal Patch Encoder)
# ==============================================================================

set -euo pipefail

# Default hyperparameters matching production training
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-2e-4}
SAVE_DIR=${SAVE_DIR:-"models/aeroguard_tslm"}

echo "================================================================="
echo "✈️  Starting AeroGuard TSLM End-to-End Pipeline"
echo "• Epochs:     $EPOCHS"
echo "• Batch Size: $BATCH_SIZE"
echo "• LR:         $LR"
echo "• Save Dir:   $SAVE_DIR"
echo "================================================================="

# Launch end-to-end Python pipeline
uv run python -m scripts.run_pipeline \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --save-dir "$SAVE_DIR" \
    "$@"
