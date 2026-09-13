"""Static jury-facing story assembled from existing challenge artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_story(root: Path = ROOT) -> dict:
    selection = _read_json(root / "artifacts/data_sourcing/selection.json") or {}
    ingest = _read_json(root / "artifacts/timenet_ingestion/report.json") or {}
    tslm = _read_json(root / "artifacts/tslm_evaluation/report.json") or {}
    metrics = _read_json(root / "artifacts/run-opentslm-3ep/metrics.json") or []
    split = _read_json(root / "artifacts/run-opentslm-3ep/split_audit.json") or {}
    user = selection.get("target_user") or {}
    models = tslm.get("models") or {}
    language = tslm.get("language_examples") or []
    compact_models = {
        name: {
            "accuracy": model.get("accuracy"),
            "macro_f1": model.get("macro_f1"),
            "hypotension_auprc": model.get("hypotension_auprc"),
        }
        for name, model in models.items()
        if isinstance(model, dict)
    }
    return {
        "target_user": user,
        "problem": selection.get("problem"),
        "sourcing": {
            "selected_dataset": selection.get("name"),
            "dataset_id": selection.get("dataset_id"),
            "landing_url": selection.get("landing_url"),
            "validation_ok": selection.get("validation_ok"),
        },
        "timenet": {
            "dataset_id": (ingest.get("timenet") or {}).get("dataset_id"),
            "records": ((ingest.get("timenet") or {}).get("audit") or {}).get("records"),
            "windows": (ingest.get("prepared_manifest_summary") or {}).get("windows"),
        },
        "held_out_vitaldb": {
            "split": tslm.get("held_out_split"),
            "parameters": tslm.get("parameters") or [],
            "models": compact_models,
            "language_example": language[0] if language else None,
        },
        "opentslm_run": {
            "checkpoint": "artifacts/run-opentslm-3ep/best.pt",
            "epochs": 3,
            "metrics": metrics,
            "split_passed": ((split.get("disjoint_verification") or {}).get("passed")),
            "val_sample_ids": split.get("val_sample_ids") or [],
            "note": (
                "best.pt is OpenTSLM-SP LoRA after 3 epochs on the synthetic "
                "four-channel fixture in data/surgical_telemetry, not on VitalDB windows."
            ),
        },
        "how_it_helps": (
            "At the monitor: read the last seconds of vitals, say whether low blood "
            "pressure is about to start, and ask the team to reassess — not to give a drug."
        ),
        "limitations": [
            "This is a research demo, not a hospital product.",
            "Low-pressure labels come from a blood-pressure rule, not from a doctor review.",
            "The 3-pass model was trained on a small practice set, not the full hospital files.",
            "On hospital test windows, saying “nothing happens” is already right most of the time.",
            "The action is reassess / check the signal, never a treatment.",
        ],
        "pitch": {
            "one_liner": (
                "From 20 seconds of vitals, tell an OR anesthesiologist whether a new "
                "low-blood-pressure episode is about to start — and only ask them to reassess."
            ),
            "say": {
                "problem": "Alarms fire when MAP is already low. We want an earlier language answer: interpret the window, anticipate a 3/5/10/15-minute horizon, act by reassessing — never by ordering a drug.",
                "source": "An agent searched open time-series sets, scored them for MAP/ABP, subject IDs, and uncredentialed access, then retrieved VitalDB.",
                "timenet": "A reusable connector standardised signals, masks, patient IDs, and two targets, then ingested a patient-disjoint TimeNet dataset.",
                "train": "On held-out VitalDB patients we compared a window-summary baseline with a TSLM. Accuracy matches the majority class; that is the result, not a success story. Separately we fine-tuned OpenTSLM for 3 epochs so the live generate button is real.",
                "live": "This live generate uses the 3-epoch OpenTSLM checkpoint on a held-out software fixture (ABP/HR/SpO2/EtCO2). It is not yet trained on VitalDB windows.",
                "limits": "Labels are code-generated, not clinician-adjudicated. Three epochs on 20 windows drop loss but do not copy the gold format. More epochs come after the jury likes the story.",
            },
        },
    }
