"""TimeNet 0.1.0 connector for prepared VitalDB hypotension windows.

The connector consumes the immutable ``manifest.json`` and split NPZ files made
by ``scripts/prepare_lstm_dataset.py``.  It performs no network I/O.  The
future-derived label appears only in TimeNet tasks; signal and annotation inputs
come exclusively from the 20-second history and non-target provenance.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    Annotation,
    AnswerTask,
    ClassificationTask,
    DataSource,
    TimeSeriesSpec,
    ureg,
)


TIMENET_VERSION = "0.1.0"
SUPPORTED_PARAMETER_COUNTS = tuple(range(1, 18))
SPLITS = ("train", "validation", "test")
CARD = Path(__file__).with_name("dataset.yaml")

PARAMETER_UNITS = {
    "MAP": "mmHg",
    "RemiEffectSite": "ng/mL",
    "RR_CO2": "1/min",
    "RemiRate": "mL/hour",
    "BIS_SQI": "percent",
    "InCO2": "mmHg",
    "EtCO2": "mmHg",
    "BIS_EMG": "dB",
    "BIS": "dimensionless",
    "BIS_SR": "percent",
    "ST_II": "mm",
    "Temperature": "degC",
    "FeO2": "percent",
    "FiO2": "percent",
    "BIS_SEF": "Hz",
    "SpO2": "percent",
    "HR": "1/min",
}

ANSWER_TEXT = {
    "within_3": "hypotension within 3 minutes",
    "within_5": "hypotension within 5 minutes",
    "within_10": "hypotension within 10 minutes",
    "within_15": "hypotension within 15 minutes",
    "none_within_15": "no hypotension within 15 minutes",
}

QA_PROMPT = (
    "Using only the supplied 20-second signal history and observation masks, predict the tightest "
    "available horizon containing the first onset of sustained MAP below 65 mmHg. Do not infer a "
    "causal mechanism or recommend a drug or dose."
)

_SOURCE = DataSource(
    data_source_type="open-clinical-dataset",
    name="VitalDB Open Dataset",
    provider="Seoul National University Hospital",
)
_MASK_SPEC = TimeSeriesSpec(
    spec_type="observation_mask",
    name="Observation mask",
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
    dtype="bool",
)


def _prepared_dir() -> Path:
    return Path(os.environ.get("VITALDB_PREPARED_DIR", "data/lstm/top_10")).expanduser().resolve()


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing prepared manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    parameters = manifest.get("parameters_in_tensor_order", [])
    if len(parameters) not in SUPPORTED_PARAMETER_COUNTS:
        raise ValueError(
            f"this connector supports prepared top-{SUPPORTED_PARAMETER_COUNTS}; "
            f"{manifest_path} declares {len(parameters)} parameters"
        )
    unknown = sorted(set(parameters) - set(PARAMETER_UNITS))
    if unknown:
        raise ValueError(f"missing verified VitalDB units for: {unknown}")
    if int(manifest.get("history_sec", 0)) != 20:
        raise ValueError("this connector's task contract requires history_sec=20")
    if int(manifest.get("horizon_sec", 0)) != 900:
        raise ValueError("this connector's five-class task requires horizon_sec=900")
    labels = {int(key): value for key, value in manifest.get("labels", {}).items()}
    if set(labels.values()) != set(ANSWER_TEXT):
        raise ValueError(f"unexpected prepared label mapping: {labels}")
    return manifest


def _answer_rationale(raw_map: np.ndarray, label: str) -> str:
    observed = raw_map[np.isfinite(raw_map)]
    if observed.size:
        delta = float(observed[-1] - observed[0])
        interpret = (
            f"MAP moved from {observed[0]:.1f} to {observed[-1]:.1f} mmHg across "
            f"{observed.size}/{raw_map.size} observed input samples (net change {delta:+.1f} mmHg)."
        )
    else:
        interpret = "No MAP samples were observed in the input window."
    return (
        f"INTERPRET: {interpret}\n"
        f"ANTICIPATE: The generated research label is {ANSWER_TEXT[label]}.\n"
        "ACT: Verify signal quality and reassess the current hemodynamic state; this research label "
        "does not prescribe treatment."
    )


class VitalDBHypotensionConnector(BaseConnector[Path]):
    """Build a prepared top-N corpus as a local TimeF dataset."""

    CARD = CARD

    def __init__(self) -> None:
        super().__init__()
        self.data_dir = _prepared_dir()

    def metadata(self):
        """Give each feature-count configuration its own TimeNet identity."""

        base = super().metadata()
        parameter_count = len(_load_manifest(self.data_dir)["parameters_in_tensor_order"])
        return replace(
            base,
            dataset_id=f"hackzurich/vitaldb-hypotension-top-{parameter_count}",
            name=f"VitalDB Hypotension Onset Windows (Top {parameter_count})",
            description=(
                f"Patient-disjoint 20-second VitalDB windows with the top {parameter_count} "
                "numeric signals and paired observation masks for generated hypotension-onset tasks. "
                "The research labels are not clinically adjudicated."
            ),
        )

    def download(self, cache_dir: Path) -> list[Path]:  # noqa: ARG002
        """Return the three already-prepared local NPZ references."""

        _load_manifest(self.data_dir)
        paths = [self.data_dir / f"{split}.npz" for split in SPLITS]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing prepared splits: " + ", ".join(missing))
        return paths

    def convert(self, raw_refs: list[Path]) -> TimeFDataset:
        """Convert split NPZs into TimeNet records, annotations, and tasks."""

        manifest = _load_manifest(self.data_dir)
        parameters = list(manifest["parameters_in_tensor_order"])
        labels = {int(key): value for key, value in manifest["labels"].items()}
        normalization = manifest["normalization"]
        median = np.asarray([normalization["median"][name] for name in parameters], dtype=np.float32)
        scale = np.asarray([normalization["iqr_scale"][name] for name in parameters], dtype=np.float32)
        interval_sec = int(manifest["interval_sec"])
        history_sec = int(manifest["history_sec"])
        base_axis = RegularAxis.from_rate_hz(Fraction(1, interval_sec))
        specs = {
            name: TimeSeriesSpec(
                spec_type=f"vitaldb_{name.lower()}",
                name=name,
                unit_value=ureg.Unit(PARAMETER_UNITS[name]),
                data_source=_SOURCE,
                dtype="float32",
                nullable=True,
            )
            for name in parameters
        }

        dataset = TimeFDataset(metadata=self.metadata())
        for split_path in raw_refs:
            split = split_path.stem
            if split not in SPLITS:
                raise ValueError(f"unexpected split reference: {split_path}")
            with np.load(split_path, allow_pickle=False) as archive:
                required = {"x_values", "x_mask", "y", "case_id", "subject_id", "cutoff_sec"}
                missing = sorted(required - set(archive.files))
                if missing:
                    raise ValueError(f"{split_path} is missing arrays: {missing}")
                values = archive["x_values"]
                masks = archive["x_mask"]
                targets = archive["y"]
                case_ids = archive["case_id"]
                subject_ids = archive["subject_id"]
                cutoffs = archive["cutoff_sec"]
            self._validate_split(split_path, values, masks, targets, case_ids, subject_ids, cutoffs, parameters)

            for index in range(values.shape[0]):
                case_id = int(case_ids[index])
                subject_id = int(subject_ids[index])
                cutoff_sec = int(cutoffs[index])
                start_sec = cutoff_sec - history_sec
                axis = base_axis.at_index(start_sec // interval_sec)
                source_id = f"vitaldb-case-{case_id}"
                record_id = f"{split}-case-{case_id}-cutoff-{cutoff_sec}"
                raw = values[index] * scale + median
                series: list[TimeSeries] = []
                for column, parameter in enumerate(parameters):
                    observed = masks[index, :, column]
                    nullable_values = [
                        float(value) if present else None
                        for value, present in zip(raw[:, column], observed, strict=True)
                    ]
                    series.append(
                        TimeSeries.from_values(
                            nullable_values,
                            spec=specs[parameter],
                            signal=parameter,
                            time_axis=axis,
                            source_id=source_id,
                            time_series_id=f"{record_id}-{parameter}",
                        )
                    )
                    series.append(
                        TimeSeries.from_values(
                            observed,
                            spec=_MASK_SPEC,
                            signal=f"{parameter}_observed",
                            time_axis=axis,
                            source_id=source_id,
                            time_series_id=f"{record_id}-{parameter}-observed",
                        )
                    )

                record = dataset.add_record(
                    time_series=tuple(series),
                    subject_ids=(f"vitaldb-subject-{subject_id}",),
                    record_id=record_id,
                )
                record.add_annotations(
                    (
                        Annotation(key="source_split", value=split, id=f"{record_id}-split"),
                        Annotation(key="source_case_id", value=case_id, id=f"{record_id}-case"),
                        Annotation(
                            key="input_start_offset",
                            value=start_sec,
                            unit="second",
                            id=f"{record_id}-input-start",
                        ),
                        Annotation(
                            key="cutoff_offset",
                            value=cutoff_sec,
                            unit="second",
                            description="Timestamp of the first excluded sample after the 20-second input window.",
                            id=f"{record_id}-cutoff",
                        ),
                        Annotation(
                            key="label_provenance",
                            value="generated_research_label_v1",
                            description=(
                                "Generated from the available 900-second future MAP slice; not clinically "
                                "adjudicated. Missing future samples can hide qualifying events."
                            ),
                            id=f"{record_id}-label-provenance",
                        ),
                        Annotation(
                            key="mask_semantics",
                            value="1=observed, 0=missing",
                            id=f"{record_id}-mask-semantics",
                        ),
                    )
                )

                label = labels[int(targets[index])]
                classification = ClassificationTask(
                    target=label,
                    target_schema="hypotension_onset_horizon_v1",
                    id=f"{record_id}-classification",
                )
                dataset.add_task(record, classification)
                map_index = parameters.index("MAP")
                map_values = raw[:, map_index].astype(np.float32, copy=True)
                map_values[~masks[index, :, map_index]] = np.nan
                dataset.add_task(
                    record,
                    AnswerTask(
                        prompt=QA_PROMPT,
                        target=ANSWER_TEXT[label],
                        rationale=_answer_rationale(map_values, label),
                        from_tasks=(classification,),
                        id=f"{record_id}-qa",
                    ),
                )
        return dataset

    @staticmethod
    def _validate_split(
        path: Path,
        values: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        case_ids: np.ndarray,
        subject_ids: np.ndarray,
        cutoffs: np.ndarray,
        parameters: list[str],
    ) -> None:
        if values.ndim != 3 or values.shape[2] != len(parameters):
            raise ValueError(f"{path}: x_values has incompatible shape {values.shape}")
        if masks.shape != values.shape or masks.dtype != np.bool_:
            raise ValueError(f"{path}: x_mask must be boolean and match x_values")
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: x_values contains NaN or infinity")
        if not np.all(values[~masks] == 0):
            raise ValueError(f"{path}: missing x_values must be zero where x_mask is false")
        for name, array in (
            ("y", targets),
            ("case_id", case_ids),
            ("subject_id", subject_ids),
            ("cutoff_sec", cutoffs),
        ):
            if array.shape != (values.shape[0],):
                raise ValueError(f"{path}: {name} must have shape ({values.shape[0]},)")
        if not set(targets.astype(int).tolist()).issubset(range(len(ANSWER_TEXT))):
            raise ValueError(f"{path}: y contains an unknown label index")


CONNECTOR = VitalDBHypotensionConnector
