#!/usr/bin/env python3
"""Export tuned components plus exact data for local evaluation; no base LLM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from training.vitaldb import DATA_FILES, TASK, fingerprint, sha256, validate_corpus, verify_bundle


def export_run(run_dir: Path, data_dir: Path, output: Path) -> Path:
    validate_corpus(data_dir)
    expected = json.loads((run_dir / "dataset_fingerprint.json").read_text())
    if fingerprint(data_dir) != expected:
        raise ValueError("Export data differs from data used for training")
    metadata = json.loads((run_dir / "run_config.json").read_text())
    if metadata.get("task") != TASK:
        raise ValueError("Refusing to export a synthetic or unrelated training run")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("task_metadata") != {"task": TASK, "dataset_fingerprint": expected}:
        raise ValueError("Checkpoint is not bound to this real VitalDB task and dataset")
    if set(checkpoint.get("trainable_state", {})) != {"encoder", "projector", "lora"}:
        raise ValueError("Expected actual OpenTSLM encoder, projector and LoRA weights")
    if not checkpoint.get("provenance"):
        raise ValueError("Checkpoint must include pinned backbone provenance")
    output.mkdir(parents=True, exist_ok=False)
    keys = ("format_version", "model_config", "provenance", "trainable_state", "epoch", "global_step", "metrics", "task_metadata")
    slim = {key: checkpoint[key] for key in keys}
    slim["model_config"]["cache_dir"] = None
    slim["model_config"]["gradient_checkpointing"] = False
    torch.save(slim, output / "best.pt")
    for name in DATA_FILES:
        shutil.copy2(data_dir / name, output / name)
    for name in ("dataset_fingerprint.json", "run_config.json", "metrics.json", "split_audit.json", "training_identity.json"):
        shutil.copy2(run_dir / name, output / name)
    (output / "bundle.json").write_text(json.dumps({
        "schema_version": 1, "task": TASK, "selected_by": "minimum validation token loss",
        "test_evaluation_performed_by_export": False,
        "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }, indent=2) + "\n")
    verify_bundle(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/run-vitaldb"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/vitaldb/onset_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true", help="Also create .tar.gz and SHA256 for a GitHub release asset")
    args = parser.parse_args(argv)
    if args.archive and Path(str(args.output) + ".tar.gz").exists():
        raise FileExistsError("Archive already exists; choose a fresh output name")
    output = export_run(args.run_dir, args.data_dir, args.output)
    print(f"Verified evaluation bundle: {output}")
    if args.archive:
        archive = Path(str(output) + ".tar.gz")
        with tarfile.open(archive, "w:gz") as target:
            target.add(output, arcname=output.name)
        Path(str(archive) + ".sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
        print(f"GitHub release asset: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
