"""Audit a strict four-channel manifest without downloading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.data import (
    ChannelNormalizer,
    TelemetryCollator,
    TelemetryDataset,
    load_manifest,
    split_records,
)


def validate_data(
    manifest: Path, output: Path, *, max_signal_length: int = 300, seed: int = 42
) -> dict:
    """Validate metadata, split identities, fit train statistics, and collate every record."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use an empty output directory for the audit: {output}")
    records = load_manifest(manifest)
    train, validation, audit = split_records(records, seed=seed)
    normalizer = ChannelNormalizer.fit(train, max_signal_length=max_signal_length)
    collator = TelemetryCollator(patch_size=4, max_signal_length=max_signal_length)
    for split in (train, validation):
        dataset = TelemetryDataset(
            split, normalizer, max_signal_length=max_signal_length
        )
        for item in dataset:
            collator([item])
    output.mkdir(parents=True, exist_ok=True)
    normalizer.save(output / "normalization.json")
    (output / "split_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "train_windows": len(train),
        "validation_windows": len(validation),
        "audit_dir": str(output),
    }


def main() -> None:
    """Run a strict data-only validation before provisioning a training GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-signal-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_data(
                args.manifest,
                args.output_dir,
                max_signal_length=args.max_signal_length,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
