# AeroGuard TSLM: Training & Architecture Tutorial

AeroGuard TSLM is a multimodal **Time-Series Language Model** tailored for turbofan engine predictive maintenance. It adapts continuous, multi-channel turbofan telemetry directly into the token embedding space of a causal language model (`SmolLM-135M-Instruct`) to simultaneously predict **Remaining Useful Life (RUL)** and synthesize **Chain-of-Thought (CoT) engineering diagnostics**.

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Understanding Steps vs. Epochs](#2-understanding-steps-vs-epochs)
3. [CLI Reference & Parameter Guide](#3-cli-reference--parameter-guide)
4. [Dataset & Preprocessing Pipeline](#4-dataset--preprocessing-pipeline)
5. [Saved Model Artifacts](#5-saved-model-artifacts)
6. [Training Recipes & Workflows](#6-training-recipes--workflows)
7. [Inference & Evaluation Guide](#7-inference--evaluation-guide)
8. [Troubleshooting & Gotchas](#8-troubleshooting--gotchas)

---

## 1. System Architecture

The core philosophy of AeroGuard TSLM is **Prefix Token Fusion**: rather than converting telemetry numbers into raw text tokens (which is verbose and token-inefficient), continuous sensor series are transformed directly into soft continuous patch tokens and prepended to text prompt embeddings.

```
                    ┌────────────────────────────┐
14 Sensor Channels  │ Temporal Patch Encoder     │
 (Window Length T)  │ Unfold (patch=10) + Linear │
───────────────────►│ + LayerNorm                │───► [B, P, 2048] (Patch Tokens)
                    └────────────────────────────┘              │
                                                                ▼ Concatenate
Prompt / Context    ┌────────────────────────────┐          ┌──────────────┐
Text Query          │ Tokenizer + Embedding      │─────────►│ Prefix Token │──► LoRA LLaMA / SmolLM
───────────────────►│ Layer                      │          │ Fusion       │    Transformer
                    └────────────────────────────┘          └──────────────┘
                                                                │        │
                                                ┌───────────────┘        └───────────────┐
                                                ▼                                        ▼
                                   Auxiliary RUL Head (MLP)                 Autoregressive Causal LM
                                   Mean-pool -> 128 -> GELU -> 1            Predicts next tokens:
                                   Output: Scalar RUL (Cycles)              Diagnostic CoT rationale
```

### Key Modules:

1. **`TimeSeriesPatchEncoder`**:
   - **Input**: Sensor tensor of shape `[batch, channels=14, time=30]`.
   - **Patching**: Non-overlapping 1D unfolding with `patch_len=10`.
   - **Projection**: Each patch of size `14 * 10 = 140` dimensions is linearly projected to `embed_dim=2048` (matching the LLM hidden size).
   - **Normalization**: Passed through `nn.LayerNorm(embed_dim)`.
   - **Output**: Patch embedding tensor `[batch, num_patches=3, embed_dim=2048]`.

2. **`AeroGuardTSLM` (Base Backbone & LoRA)**:
   - **Backbone**: `HuggingFaceTB/SmolLM-135M-Instruct` (LLaMA architecture).
   - **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**:
     - Rank $r = 16$, $\alpha = 32$, dropout $0.05$.
     - Targets query, key, value, and output projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
     - Freezes base model weights, training only adapter layers and temporal heads.

3. **Prefix Token Fusion**:
   - Sensor patch tokens $[B, P, D]$ are concatenated in front of prompt text embeddings $[B, L, D]$ to form an input sequence of length $P + L$.
   - **Label Masking**: Time-series prefix positions are padded with label `-100` so that language modeling loss is strictly computed over the diagnostic explanation tokens.

4. **Multi-Task Objective**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \mathcal{L}_{\text{RUL}}$$
   - $\mathcal{L}_{\text{LM}}$: Cross-entropy loss for generating diagnostic reasoning tokens.
   - $\mathcal{L}_{\text{RUL}}$: Mean Squared Error (MSE) between true RUL and predicted scalar RUL.
   - The scalar RUL is predicted by an auxiliary MLP head (`rul_head`) operating on the mean-pooled temporal patch embeddings.

---

## 2. Understanding Steps vs. Epochs

When running the training script, you will see output like:
```text
Epoch 1/30 | Step 157/4710 | Loss: 18.4102
```

### Definitions:
- **Batch**: A sub-collection of engine windows processed concurrently (e.g. 16 windows).
- **Step (Iteration)**: One forward pass, loss calculation, backward pass, and weight update for a single batch.
- **Epoch**: One complete pass through the entire training dataset.

### Why 4,710 Steps for 30 Epochs?
| Metric | Value | Computation |
| :--- | :--- | :--- |
| **Training Samples** | `2,502` | Extracted from `data/processed/windows.jsonl` |
| **Batch Size** | `16` | Passed via `--batch-size 16` |
| **Steps per Epoch** | `157` | $\lceil 2502 / 16 \rceil = 157$ batches per epoch |
| **Epochs** | `30` | Passed via `--epochs 30` |
| **Total Steps** | **`4,710`** | $157 \text{ steps/epoch} \times 30 \text{ epochs} = \mathbf{4,710}$ steps |

---

## 3. CLI Reference & Parameter Guide

Run the training script using `uv`:

```bash
uv run python -m training.train_opentslm [OPTIONS]
```

### Arguments:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model-id` | `str` | `HuggingFaceTB/SmolLM-135M-Instruct` | Hugging Face model repository ID for the base causal LM. |
| `--epochs` | `int` | `1` | Number of full training epochs over the dataset. |
| `--batch-size` | `int` | `4` | Batch size per GPU step. Use `16` for optimal GPU throughput. |
| `--lr` | `float` | `2e-4` | Learning rate for AdamW optimizer (includes linear warmup for 10 steps). |
| `--max-steps` | `int` | `None` | Optional step cap. If set, stops training after $N$ steps regardless of epochs. |
| `--save-dir` | `str` | `models/aeroguard_tslm` | Directory where trained checkpoints, adapters, and metadata are saved. |

---

## 4. Dataset & Preprocessing Pipeline

Training utilizes preprocessed telemetry from NASA C-MAPSS (FD001 turbofan simulation):

1. **Raw Ingestion**:
   ```bash
   uv run python -m scripts.download_data
   ```
   Downloads and validates `data/raw/train_FD001.txt`.

2. **Window Slicing**:
   ```bash
   uv run python -m scripts.preprocess_data
   ```
   Extracts rolling telemetry windows into `data/processed/windows.jsonl`:
   - `train`: 2,502 windows
   - `validation`: 355 windows
   - `test`: 806 windows

3. **14 Active Sensor Channels**:
   Telemetry filters out invariant channels and tracks:
   `T24, T30, T50, P30, Nf, Nc, Ps30, phi, NRf, NRc, BPR, htBleed, W31, W32`.

4. **Normalization**:
   `CMAPSSCoTDataset` computes population means and standard deviations strictly on the training partition. Constant sensors receive a scale of 1.0. These parameters are saved directly into `preprocessing.json` during training.

---

## 5. Saved Model Artifacts

At the conclusion of training, the model saves all necessary weights and inference configs to `--save-dir`:

```
models/aeroguard_tslm/
├── lora_adapters/
│   ├── adapter_config.json      # LoRA configuration (r=16, alpha=32)
│   └── adapter_model.safetensors# Fine-tuned adapter weights for LLM projections
├── tslm_adapters.pt             # PyTorch state dict for TimeSeriesPatchEncoder and RUL head
├── tokenizer/                   # Tokenizer vocabulary and configuration
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── preprocessing.json           # Per-sensor normalization stats (means & scales)
```

---

## 6. Training Recipes & Workflows

### Recipe A: Full Production Training (Recommended)
Train for 30 epochs with batch size 16:
```bash
uv run python -m training.train_opentslm \
    --epochs 30 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
```

### Recipe B: Standard Baseline Run (3 Epochs)
Train for 3 full epochs (~471 steps):
```bash
uv run python -m training.train_opentslm \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-4 \
    --save-dir models/aeroguard_tslm
```

### Recipe C: Rapid Smoke-Test / Verification
Test the pipeline and verify checkpoint writing in under 30 seconds by capping steps:
```bash
uv run python -m training.train_opentslm \
    --max-steps 15 \
    --save-dir models/aeroguard_tslm
```

---

## 7. Inference & Evaluation Guide

Once trained, use the model to evaluate telemetry windows programmatically:

```python
import torch
from pathlib import Path
from training.train_opentslm import AeroGuardTSLM

# 1. Initialize architecture
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AeroGuardTSLM(device=device)

# 2. Load trained adapters
save_dir = Path("models/aeroguard_tslm")
tslm_weights = torch.load(save_dir / "tslm_adapters.pt", map_location=device)
model.ts_encoder.load_state_dict(tslm_weights["patch_encoder"])
model.rul_head.load_state_dict(tslm_weights["rul_head"])
model.llm.load_adapter(str(save_dir / "lora_adapters"), adapter_name="default")
model.to(device)

# 3. Prepare normalized 14-channel sensor window [14, 30]
# Shape: [14 channels, 30 timesteps]
sample_sensor_window = torch.randn(14, 30).to(device)

# 4. Generate prediction and Chain-of-Thought diagnostic
pred_rul, diagnostic_text = model.generate_assessment(
    sensor_series=sample_sensor_window,
    prompt_text="Inspect engine operational state and predict remaining useful life.",
    max_new_tokens=128,
)

print(f"Predicted RUL: {pred_rul:.1f} cycles")
print(f"Diagnostics: {diagnostic_text}")
```

---

## 8. Troubleshooting & Gotchas

1. **`Python.h: No such file or directory` (Triton JIT error)**:
   - **Cause**: Triton JIT compiler needs Python C headers to compile `cuda_utils.so` for GPU rotary embeddings.
   - **Fix**: Install `python3-dev` / `python3.12-dev` on the host system:
     ```bash
     sudo apt-get update && sudo apt-get install -y python3-dev python3.12-dev
     ```

2. **Training finishes after 1 epoch (15 steps)**:
   - **Cause**: `--max-steps` previously defaulted to `15` for quick demo verification.
   - **Fix**: `--max-steps` now defaults to `None`. Ensure you do not pass `--max-steps` if you want full epoch training.

3. **CUDA Out of Memory (OOM)**:
   - **Fix**: Reduce batch size:
     ```bash
     --batch-size 8   # or --batch-size 4
     ```
