"""TimeNet connector for NASA C-MAPSS Turbofan Degradation dataset."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from timenet.connectors.base import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    DataSource,
    ScalarPredictionTask,
    TimeSeriesSpec,
    ureg,
)

_DATA_SOURCE = DataSource(data_source_type="nasa", name="C-MAPSS", provider="NASA Ames")

_SPEC_TEMP = TimeSeriesSpec(
    spec_type="temp",
    name="Temperature",
    unit_value=ureg.degree_Rankine,
    data_source=_DATA_SOURCE,
)
_SPEC_PRESS = TimeSeriesSpec(
    spec_type="press",
    name="Pressure",
    unit_value=ureg.psi,
    data_source=_DATA_SOURCE,
)
_SPEC_SPEED = TimeSeriesSpec(
    spec_type="speed",
    name="Rotational Speed",
    unit_value=ureg.rpm,
    data_source=_DATA_SOURCE,
)
_SPEC_RATIO = TimeSeriesSpec(
    spec_type="ratio",
    name="Ratio",
    unit_value=ureg.dimensionless,
    data_source=_DATA_SOURCE,
)
_SPEC_ENTHALPY = TimeSeriesSpec(
    spec_type="enthalpy",
    name="Enthalpy",
    unit_value=ureg.dimensionless,
    data_source=_DATA_SOURCE,
)
_SPEC_BLEED = TimeSeriesSpec(
    spec_type="bleed",
    name="Coolant Bleed",
    unit_value=ureg.pound / ureg.second,
    data_source=_DATA_SOURCE,
)

# Map sensor keys to their shared measurement spec and human-readable signal name
SENSOR_DEFINITIONS = {
    "sensor_2": (_SPEC_TEMP, "T24_lpc_outlet_temp"),
    "sensor_3": (_SPEC_TEMP, "T30_hpc_outlet_temp"),
    "sensor_4": (_SPEC_TEMP, "T50_lpt_outlet_temp"),
    "sensor_7": (_SPEC_PRESS, "P30_hpc_outlet_press"),
    "sensor_8": (_SPEC_SPEED, "Nf_fan_speed"),
    "sensor_9": (_SPEC_SPEED, "Nc_core_speed"),
    "sensor_11": (_SPEC_PRESS, "Ps30_hpc_static_press"),
    "sensor_12": (_SPEC_RATIO, "phi_fuel_flow_ratio"),
    "sensor_13": (_SPEC_SPEED, "NRf_corr_fan_speed"),
    "sensor_14": (_SPEC_SPEED, "NRc_corr_core_speed"),
    "sensor_15": (_SPEC_RATIO, "BPR_bypass_ratio"),
    "sensor_17": (_SPEC_ENTHALPY, "htBleed_enthalpy"),
    "sensor_20": (_SPEC_BLEED, "W31_hpt_coolant_bleed"),
    "sensor_21": (_SPEC_BLEED, "W32_lpt_coolant_bleed"),
}


class CMAPSSConnector(BaseConnector[dict[str, Any]]):
    """Connector that converts processed C-MAPSS windows into TimeF format."""

    def download(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Fetch and prepare C-MAPSS records, executing automated pipeline if data is missing."""
        data_path = Path("data/processed/windows.jsonl")
        if not data_path.exists():
            data_path = cache_dir / "windows.jsonl"

        if not data_path.exists():
            from scripts.download_data import download_raw_data
            from scripts.preprocess_data import preprocess_data

            raw_path = Path("data/raw")
            output_path = Path("data/processed")

            if not raw_path.parent.exists():
                raw_path = cache_dir / "raw"
                output_path = cache_dir / "processed"

            download_raw_data(raw_path)
            data_path = preprocess_data(raw_dir=str(raw_path), output_dir=str(output_path))

        records: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        """Convert JSONL rows into a populated TimeFDataset."""
        dataset = TimeFDataset(metadata=self.metadata())

        for row in raw_refs:
            # Build TimeSeries signals
            time_series_list: list[TimeSeries] = []
            for sensor_name, (spec, signal_name) in SENSOR_DEFINITIONS.items():
                if sensor_name in row["series"]:
                    values = np.array(row["series"][sensor_name], dtype=np.float32)
                    ts = TimeSeries.from_values(
                        values,
                        spec=spec,
                        signal=signal_name,
                        time_axis=OrdinalAxis(),
                    )
                    time_series_list.append(ts)

            # Record
            record = dataset.add_record(
                time_series=tuple(time_series_list),
                record_id=row["record_id"],
            )

            # Annotations: preserve unit, split, cycle, status, window, and provenance
            annotations = [
                Annotation(key="unit_number", value=str(row["unit_number"])),
                Annotation(key="split", value=str(row["split"])),
                Annotation(key="cycle", value=str(row["cycle"])),
            ]
            if "status" in row:
                annotations.append(Annotation(key="status", value=str(row["status"])))
            if "window_size" in row:
                annotations.append(Annotation(key="window_size", value=str(row["window_size"])))
            elif "series" in row and row["series"]:
                first_sensor = next(iter(row["series"].values()))
                annotations.append(Annotation(key="window_size", value=str(len(first_sensor))))
            if "window_start" in row:
                annotations.append(Annotation(key="window_start", value=str(row["window_start"])))
            if "label_provenance" in row and row["label_provenance"]:
                annotations.append(
                    Annotation(key="label_provenance", value=json.dumps(row["label_provenance"]))
                )

            record.add_annotations(annotations)

            # AnswerTask with CoT
            task_qa = AnswerTask(
                id=f"qa-{row['record_id']}",
                prompt=row["prompt"],
                target=row["target"],
                rationale=row["rationale"],
            )

            # ScalarPredictionTask for numerical RUL benchmark
            task_rul = ScalarPredictionTask(
                id=f"rul-{row['record_id']}",
                target=float(row["rul"]),
                unit=ureg.cycle,
                target_name="RUL",
            )

            dataset.add_tasks(record, [task_qa, task_rul])

        return dataset


CONNECTOR = CMAPSSConnector
