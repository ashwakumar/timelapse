#!/usr/bin/env python3
"""Download the pinned Hugging Face weights the trained best.pt needs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training/config_50_epochs.json"
CACHE = ROOT / "models/huggingface"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN in .env (copy .env.example). "
            "Create a read token at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    config = json.loads(CONFIG.read_text(encoding="utf-8"))["model"]
    from huggingface_hub import hf_hub_download, snapshot_download

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config['model_id']} …", flush=True)
    checkpoint = hf_hub_download(
        repo_id=config["model_id"],
        filename="model_checkpoint.pt",
        revision=config["model_revision"],
        cache_dir=CACHE,
        token=token,
    )
    print(f"Downloading {config['base_model_id']} …", flush=True)
    base = snapshot_download(
        repo_id=config["base_model_id"],
        revision=config["base_model_revision"],
        cache_dir=CACHE,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"],
    )
    print(f"OpenTSLM checkpoint: {checkpoint}")
    print(f"Base LLM snapshot: {base}")
    print("Hugging Face weights are cached locally. They are not committed to Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
