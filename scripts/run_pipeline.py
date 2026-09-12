"""End-to-End Preprocessing & Training Pipeline for AeroGuard TSLM.

Orchestrates the entire lifecycle:
1. Agentic Data Sourcing & Raw Telemetry Validation (FD001 C-MAPSS)
2. Window Generation & Leakage-Free Engine Splitting
3. Aerospace Engineering Chain-of-Thought (CoT) Diagnostic Synthesis
4. Dataset Normalization Verification & Artifact Generation
5. OpenTSLM Model Fine-Tuning & Checkpoint Export
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from scripts.agentic_cot_synthesizer import enrich_dataset_with_agentic_cot
from scripts.agentic_data_sourcing import run_data_sourcing_pipeline
from scripts.preprocess_data import preprocess_data
from training.dataset_loader import CMAPSSCoTDataset
from training.train_opentslm import train


def run_full_pipeline(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-4,
    save_dir: str = "models/aeroguard_tslm",
    max_steps: int | None = None,
    model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct",
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    skip_preprocess: bool = False,
    build_registry: bool = False,
    registry_dir: str = "artifacts/registry",
) -> None:
    """Execute the full data preparation and model training workflow."""
    start_time = time.perf_counter()

    print("\n" + "=" * 75)
    print("🚀 AEROGUARD TSLM: END-TO-END PREPROCESSING & TRAINING PIPELINE")
    print("=" * 75)
    print(f"• Target Save Directory: {save_dir}")
    print(f"• Training Settings: Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    if max_steps is not None:
        print(f"• Step Cap (Smoke Test): {max_steps} steps")
    print(f"• Base Model Backbone: {model_id}")
    print("=" * 75 + "\n")

    windows_file = Path(output_dir) / "windows.jsonl"

    if not skip_preprocess:
        # Step 1: Agentic Data Sourcing & Raw Telemetry Validation
        step1_start = time.perf_counter()
        print("─" * 75)
        print("📌 [STAGE 1/5] Agentic Data Sourcing & Raw Telemetry Validation")
        print("─" * 75)
        dossier = run_data_sourcing_pipeline(raw_dir=raw_dir, artifacts_dir=artifacts_dir)
        print(
            f"✔ Verified raw telemetry: {dossier.total_records} rows across {dossier.engine_count} engines."
        )
        print(f"✔ Dataset sourcing dossier generated in {artifacts_dir}/")
        print(f"⏱ Stage 1 duration: {time.perf_counter() - step1_start:.2f}s\n")

        # Step 2: Preprocessing & 30-Cycle Window Generation
        step2_start = time.perf_counter()
        print("─" * 75)
        print("📌 [STAGE 2/5] Telemetry Preprocessing & Window Slicing")
        print("─" * 75)
        preprocess_data(
            raw_dir=raw_dir,
            output_dir=output_dir,
            window_size=30,
            stride=5,
        )
        print(f"✔ Generated 30-cycle rolling windows at {windows_file}")
        print(f"⏱ Stage 2 duration: {time.perf_counter() - step2_start:.2f}s\n")

        # Step 3: Chain-of-Thought (CoT) Diagnostic Synthesis
        step3_start = time.perf_counter()
        print("─" * 75)
        print("📌 [STAGE 3/5] Chain-of-Thought (CoT) Diagnostic Target Synthesis")
        print("─" * 75)
        count = enrich_dataset_with_agentic_cot(
            input_file=windows_file,
            output_file=windows_file,
        )
        print(f"✔ Enriched {count} records with aerothermal reasoning and maintenance directives.")
        print(f"⏱ Stage 3 duration: {time.perf_counter() - step3_start:.2f}s\n")
    else:
        print("⏩ Skipping Stages 1-3 (--skip-preprocess specified).")
        if not windows_file.is_file():
            print(f"❌ Error: {windows_file} does not exist. Cannot skip preprocessing.")
            sys.exit(1)
        print(f"✔ Using existing preprocessed dataset: {windows_file}\n")

    # Optional: Build TimeNet / TimeF Registry if requested
    if build_registry:
        step_reg_start = time.perf_counter()
        print("─" * 75)
        print("📌 [OPTIONAL] Building TimeNet / TimeF Registry")
        print("─" * 75)
        from scripts.build_timef_registry import build_and_verify

        reg_version = build_and_verify(registry_dir=registry_dir)
        print(f"✔ TimeF registry built and verified: {reg_version}")
        print(f"⏱ Registry build duration: {time.perf_counter() - step_reg_start:.2f}s\n")

    # Step 4: Verification & Preprocessing Normalization Statistics
    step4_start = time.perf_counter()
    print("─" * 75)
    print("📌 [STAGE 4/5] Normalization Statistics & Split Verification")
    print("─" * 75)
    norm_dest = Path(save_dir) / "preprocessing.json"
    norm_dest.parent.mkdir(parents=True, exist_ok=True)
    dataset = CMAPSSCoTDataset(jsonl_path=str(windows_file), split="train")
    dataset.normalization.save(norm_dest)
    print(f"✔ Verified {len(dataset)} training windows.")
    print(f"✔ Computed and saved normalization statistics to: {norm_dest}")
    print(f"⏱ Stage 4 duration: {time.perf_counter() - step4_start:.2f}s\n")

    # Step 5: Model Training Execution
    step5_start = time.perf_counter()
    print("─" * 75)
    print("📌 [STAGE 5/5] OpenTSLM Model Fine-Tuning & Adapter Export")
    print("─" * 75)
    train(
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        save_dir=save_dir,
    )
    print(f"⏱ Stage 5 duration: {time.perf_counter() - step5_start:.2f}s\n")

    total_duration = time.perf_counter() - start_time
    minutes, seconds = divmod(int(total_duration), 60)
    print("=" * 75)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY IN {minutes}m {seconds}s")
    print(f"💾 Checkpoints and adapters saved at: {save_dir}/")
    print("=" * 75 + "\n")


def main() -> None:
    """CLI parser for run_pipeline."""
    parser = argparse.ArgumentParser(
        description="Execute full preprocessing and training pipeline for AeroGuard TSLM."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per step (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW optimizer (default: 2e-4)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/aeroguard_tslm",
        help="Directory to save model weights and tokenizer (default: models/aeroguard_tslm)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step cap (useful for quick smoke verification)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolLM-135M-Instruct",
        help="HuggingFace model ID for base causal LM",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw FD001 telemetry (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for processed windows.jsonl (default: data/processed)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory for dataset sourcing dossier (default: artifacts)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip raw download, window slicing, and CoT synthesis if data is already prepared",
    )
    parser.add_argument(
        "--build-registry",
        action="store_true",
        help="Also build and verify the TimeNet / TimeF registry",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="artifacts/registry",
        help="Directory for TimeNet registry (default: artifacts/registry)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        model_id=args.model_id,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        skip_preprocess=args.skip_preprocess,
        build_registry=args.build_registry,
        registry_dir=args.registry_dir,
    )


if __name__ == "__main__":
    main()
