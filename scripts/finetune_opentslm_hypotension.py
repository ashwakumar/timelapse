#!/usr/bin/env python3
"""Optional neural TSLM fine-tune on the same leakage-safe hypotension splits.

Primary comparison (patch TSLM vs window baseline) is
``scripts/train_evaluate_tslm.py`` and does not need PyTorch.

This script, in the training environment, fits a small encoder/classifier on
OpenTSLM-style value/mask patches and writes a checkpoint. It is the neural
stand-in for SoftPrompt+LoRA when the gated 1B OpenTSLM weights are not
downloaded. If ``opentslm`` is installed it records the pinned commit so a
later GPU run can swap in the published SP checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/lstm/top_5"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/opentslm_hypotension"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def finetune(args: argparse.Namespace) -> dict:
    from evaluation.compare import LABEL_COUNT, score_model
    from evaluation.features import tslm_patch_matrix
    from evaluation.leakage import audit_prepared_splits
    from scripts.opentslm_vitaldb_dataset import (
        UPSTREAM_OPENTSLM_COMMIT,
        PreparedVitalDBCorpus,
    )

    corpus = PreparedVitalDBCorpus(args.data_dir)
    leakage = audit_prepared_splits(corpus)
    train = corpus.load_split("train")
    test = corpus.load_split("test")
    x_train, y_train = tslm_patch_matrix(train)
    x_test, y_test = tslm_patch_matrix(test)

    try:
        import torch
        from torch import nn
    except ImportError:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "skipped",
            "reason": "torch is not installed; use scripts/train_evaluate_tslm.py",
            "leakage": leakage,
            "opentslm_commit": UPSTREAM_OPENTSLM_COMMIT,
        }
        (args.output_dir / "finetune_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report

    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, LABEL_COUNT),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "train_loss": float(loss.detach())})

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
    scores = score_model("torch_patch_tslm", y_test, predictions, probabilities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "tslm_head.pt"
    torch.save({"state_dict": model.state_dict(), "in_features": x_train.shape[1]}, checkpoint)
    opentslm_version = None
    try:
        import opentslm

        opentslm_version = getattr(opentslm, "__version__", "present")
    except ImportError:
        opentslm_version = None

    report = {
        "status": "trained",
        "data_dir": str(args.data_dir),
        "held_out_split": "test",
        "leakage": leakage,
        "test_metrics": scores.to_dict(),
        "history": history,
        "checkpoint": str(checkpoint),
        "opentslm_commit": UPSTREAM_OPENTSLM_COMMIT,
        "opentslm_package": opentslm_version,
        "note": (
            "Test split was not used for training. This MLP reads the same "
            "OpenTSLM value/mask patches as evaluation.compare. Swap the head "
            "for OpenTSLM-SP LoRA when the 1B checkpoint is available."
        ),
    }
    (args.output_dir / "finetune_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = finetune(args)
    print(json.dumps({"status": report["status"], "reason": report.get("reason")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
