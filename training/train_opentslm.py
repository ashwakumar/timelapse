"""OpenTSLM Training script for AeroGuard TSLM.

Adapts a causal language model with a multivariate temporal patch encoder
to reason over continuous turbofan telemetry and synthesize engineering CoT rationales.
"""

import argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from training.dataset_loader import CMAPSSCoTDataset


class TimeSeriesPatchEncoder(nn.Module):
    """Encodes multivariate sensor patches into LLM token embeddings."""

    def __init__(self, num_channels: int = 14, patch_len: int = 10, embed_dim: int = 2048):
        super().__init__()
        self.patch_len = patch_len
        self.num_channels = num_channels
        self.proj = nn.Linear(num_channels * patch_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch, channels, time]

        Returns:
            Tensor of shape [batch, num_patches, embed_dim]
        """
        B, C, T = x.shape
        # unfold dimension -1 (time) with window=patch_len, step=patch_len
        patches = x.unfold(
            dimension=-1, size=self.patch_len, step=self.patch_len
        )  # [B, C, num_patches, patch_len]
        num_patches = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, num_patches, C * self.patch_len)
        return self.norm(self.proj(patches))


class AeroGuardTSLM(nn.Module):
    """End-to-end Time-Series Language Model for Turbofan Predictive Maintenance."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        num_channels: int = 14,
        patch_len: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_id = model_id
        self.tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )

        embed_dim = base_llm.config.hidden_size

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(base_llm, lora_config)

        self.ts_encoder = TimeSeriesPatchEncoder(
            num_channels=num_channels,
            patch_len=patch_len,
            embed_dim=embed_dim,
        )

        # Auxiliary scalar RUL regression head from temporal patch tokens
        self.rul_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        sensor_series: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        rul_targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """Forward pass integrating temporal patch tokens and text tokens."""
        ts_embeds = self.ts_encoder(sensor_series)  # [B, P, D]
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, D]
        inputs_embeds = torch.cat([ts_embeds, text_embeds], dim=1)  # [B, P+L, D]

        if labels is not None:
            # Mask out the time-series tokens in the language modeling loss
            pad_labels = torch.full(
                (labels.shape[0], ts_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([pad_labels, labels], dim=1)
        else:
            full_labels = None

        full_attention_mask = None
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], ts_embeds.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            labels=full_labels,
            attention_mask=full_attention_mask,
        )
        lm_loss = outputs.loss if labels is not None else None

        # Predict RUL from mean-pooled temporal embeddings
        pooled_ts = ts_embeds.mean(dim=1)
        pred_rul = self.rul_head(pooled_ts).squeeze(-1)

        total_loss = lm_loss
        if lm_loss is not None and rul_targets is not None:
            rul_loss = nn.functional.mse_loss(pred_rul, rul_targets)
            # Weighted multi-task loss
            total_loss = lm_loss + 0.01 * rul_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "logits": outputs.logits,
            "pred_rul": pred_rul,
        }

    @torch.no_grad()
    def generate_assessment(
        self,
        sensor_series: torch.Tensor,
        prompt_text: str,
        max_new_tokens: int = 128,
    ) -> tuple[float, str]:
        """Generate predicted RUL and diagnostic CoT response."""
        self.eval()
        if sensor_series.dim() == 2:
            sensor_series = sensor_series.unsqueeze(0)

        ts_embeds = self.ts_encoder(sensor_series)
        pred_rul = float(self.rul_head(ts_embeds.mean(dim=1)).squeeze().item())

        prompt_full = f"<|prompt|>{prompt_text}\n<|response|>"
        input_ids = self.tokenizer.encode(prompt_full, return_tensors="pt")

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ts_embeds, text_embeds], dim=1)

        # Autoregressive generation
        generated: list[int] = []
        curr_embeds = inputs_embeds
        for _ in range(max_new_tokens):
            out = self.llm(inputs_embeds=curr_embeds)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            token_id = int(next_token.item())
            if token_id == self.tokenizer.eos_token_id:
                break
            generated.append(token_id)
            next_embed = self.llm.get_input_embeddings()(next_token.unsqueeze(0))
            curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)

        response_text = cast(str, self.tokenizer.decode(generated, skip_special_tokens=True))
        return pred_rul, response_text


def train(
    model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_steps: int | None = None,
    save_dir: str = "models/aeroguard_tslm",
):
    """Execute training of AeroGuard TSLM."""
    print("=" * 65)
    print(f"✈️ Training AeroGuard TSLM on Turbofan Telemetry (Backbone: {model_id})")
    print("=" * 65)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = AeroGuardTSLM(model_id=model_id, device=device)
    model.to(device)

    dataset = CMAPSSCoTDataset(
        jsonl_path="data/processed/windows.jsonl",
        split="train",
        tokenizer=model.tokenizer,
        max_length=640,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs if max_steps is None else max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=total_steps
    )

    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            sensor_series = batch["sensor_series"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            rul = batch["rul"].to(device)

            outputs = model(
                sensor_series=sensor_series,
                input_ids=input_ids,
                labels=labels,
                rul_targets=rul,
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            if step % 20 == 0 or step == 1:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Step {step}/{total_steps} | Loss: {loss.item():.4f}"
                )

            if max_steps and step >= max_steps:
                break

        print(f"✅ Epoch {epoch + 1} Completed. Avg Loss: {epoch_loss / max(1, step):.4f}")
        if max_steps and step >= max_steps:
            break

    # Save model weights and configuration
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.normalization.save(save_path / "preprocessing.json")
    torch.save(
        {
            "patch_encoder": model.ts_encoder.state_dict(),
            "rul_head": model.rul_head.state_dict(),
        },
        save_path / "tslm_adapters.pt",
    )
    model.llm.save_pretrained(str(save_path / "lora_adapters"))
    model.tokenizer.save_pretrained(str(save_path / "tokenizer"))
    print(f"\n💾 Model adapters and tokenizer saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AeroGuard TSLM")
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max-steps", type=int, default=15, help="Quick calibration steps for demo verification"
    )
    parser.add_argument("--save-dir", type=str, default="models/aeroguard_tslm")
    args = parser.parse_args()

    train(
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
    )
