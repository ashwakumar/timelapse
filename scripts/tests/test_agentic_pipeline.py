"""Unit tests for the Agentic Data Sourcing and CoT Diagnostic Pipeline."""

from pathlib import Path

from scripts.agentic_cot_synthesizer import (
    calculate_sensor_drifts,
    enrich_dataset_with_agentic_cot,
    synthesize_aerospace_cot,
)
from scripts.agentic_data_sourcing import assess_candidates, run_data_sourcing_pipeline


def test_agentic_sourcing_candidate_selection():
    """Verify autonomous candidate evaluation selects NASA C-MAPSS FD001."""
    candidate = assess_candidates()
    assert candidate["id"] == "nasa/cmapss-fd001"
    assert candidate["selected"] is True
    assert candidate["suitability_score"] >= 9.0


def test_agentic_sourcing_dossier_generation(tmp_path: Path):
    """Verify complete dossier generation with valid data integrity and checksums."""
    dossier = run_data_sourcing_pipeline(raw_dir="data/raw", artifacts_dir=tmp_path)
    assert dossier.selected_dataset["id"] == "nasa/cmapss-fd001"
    assert dossier.total_records == 20631
    assert dossier.engine_count == 100
    assert len(dossier.sha256_checksum) == 64
    assert len(dossier.selected_channels) == 14

    # Check files created
    assert (tmp_path / "dataset_sourcing_dossier.json").is_file()
    assert (tmp_path / "dataset_sourcing_dossier.md").is_file()


def test_cot_severity_staging():
    """Verify 4-stage CoT synthesis across NORMAL, WARNING, and CRITICAL regimes."""
    mock_series = {
        "sensor_4": [1400.0, 1405.0],
        "sensor_11": [47.5, 47.3],
        "sensor_15": [8.40, 8.42],
    }

    # 1. Normal regime
    normal_rep = synthesize_aerospace_cot(1, 40, mock_series, rul=120, split="train")
    assert normal_rep.status == "NORMAL"
    assert "healthy operational envelope" in normal_rep.health_assessment
    assert "routine C-check" in normal_rep.maintenance_directive
    assert "[TELEMETRY OBSERVATION]" in normal_rep.full_rationale

    # 2. Warning regime
    warning_rep = synthesize_aerospace_cot(1, 140, mock_series, rul=50, split="train")
    assert warning_rep.status == "WARNING"
    assert "accelerating wear regime" in warning_rep.health_assessment
    assert "borescope inspection" in warning_rep.maintenance_directive

    # 3. Critical regime
    critical_rep = synthesize_aerospace_cot(1, 190, mock_series, rul=15, split="train")
    assert critical_rep.status == "CRITICAL"
    assert "critical wear regime" in critical_rep.health_assessment
    assert "AOG" in critical_rep.maintenance_directive


def test_cot_zero_leakage_on_test_split():
    """Verify zero-leakage invariant: test engines must NEVER leak ground-truth RUL."""
    mock_series = {
        "sensor_4": [1400.0, 1415.0],
        "sensor_11": [47.5, 46.5],
    }
    test_rep = synthesize_aerospace_cot(85, 150, mock_series, rul=18, split="test")

    # In test split, ground-truth 18 must be hidden
    assert test_rep.projected_rul is None
    assert "18 cycles" not in test_rep.target_summary
    assert "18 cycles" not in test_rep.full_rationale
    assert "Under evaluation" in test_rep.target_summary


def test_calculate_sensor_drifts():
    """Verify accurate delta calculation across sensor window series."""
    series = {
        "sensor_4": [1400.0, 1405.0, 1410.5],
        "sensor_11": [47.5, 47.0, 46.8],
        "sensor_empty": [],
    }
    drifts = calculate_sensor_drifts(series)
    assert drifts["sensor_4"] == 10.5
    assert drifts["sensor_11"] == -0.7
    assert drifts["sensor_empty"] == 0.0


def test_enrich_dataset_smoke(tmp_path: Path):
    """Verify reading and enriching a batch of records."""
    sample_file = tmp_path / "sample_windows.jsonl"
    out_file = tmp_path / "enriched_windows.jsonl"

    import json

    sample_record = {
        "record_id": "FD001-U001-C0030",
        "unit_number": 1,
        "cycle": 30,
        "split": "train",
        "rul": 162,
        "series": {"sensor_4": [1400.0, 1402.0], "sensor_11": [47.5, 47.4]},
    }
    sample_file.write_text(json.dumps(sample_record) + "\n", encoding="utf-8")

    count = enrich_dataset_with_agentic_cot(
        input_file=sample_file, output_file=out_file, sample_limit=1
    )
    assert count == 1
    assert out_file.is_file()

    enriched = json.loads(out_file.read_text(encoding="utf-8").strip())
    assert enriched["status"] == "NORMAL"
    assert "cot_diagnostics" in enriched
    assert enriched["label_provenance"]["rationale"] == "aerospace_thermodynamic_cot_synthesis"
