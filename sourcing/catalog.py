"""Seed catalog of known open time-series sources relevant to the problem card.

Live search (Hugging Face Hub, landing-page fetches) can add more candidates.
This list exists so assessment is explicit: nearby ICU, wearable, and industrial
sets are considered and rejected with reasons, not ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DatasetCandidate:
    dataset_id: str
    name: str
    source: str
    landing_url: str
    access_url: str
    license: str
    domain: str
    description: str
    signals: tuple[str, ...]
    has_arterial_pressure: bool
    has_subject_ids: bool
    sampling_hz: float | None
    programmatic_access: bool
    credentialed: bool
    time_series: bool
    notes: str = ""
    discovered_by: str = "seed_catalog"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["signals"] = list(self.signals)
        return payload


SEED_CATALOG: tuple[DatasetCandidate, ...] = (
    DatasetCandidate(
        dataset_id="vitaldb",
        name="VitalDB Open Dataset",
        source="vitaldb",
        landing_url="https://vitaldb.net/dataset/",
        access_url="https://api.vitaldb.net/cases",
        license="CC BY-NC-SA 4.0",
        domain="intraoperative",
        description=(
            "High-fidelity multi-parameter vital signs from surgical patients, with "
            "case and subject identifiers and a public Python API."
        ),
        signals=(
            "MAP",
            "ART",
            "HR",
            "SpO2",
            "EtCO2",
            "BIS",
            "remifentanil",
            "propofol",
        ),
        has_arterial_pressure=True,
        has_subject_ids=True,
        sampling_hz=1.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="Primary OR telemetry source used by the rest of this repository.",
    ),
    DatasetCandidate(
        dataset_id="hirid",
        name="HiRID",
        source="physionet",
        landing_url="https://physionet.org/content/hirid/1.1.1/",
        access_url="https://physionet.org/content/hirid/1.1.1/",
        license="PhysioNet Credentialed Health Data License",
        domain="icu",
        description="High-resolution ICU time series, including blood pressure.",
        signals=("arterial blood pressure", "heart rate", "SpO2"),
        has_arterial_pressure=True,
        has_subject_ids=True,
        sampling_hz=0.5,
        programmatic_access=True,
        credentialed=True,
        time_series=True,
        notes="Relevant physiology, but credentialed ICU access, not open OR data.",
    ),
    DatasetCandidate(
        dataset_id="mimic-iii-waveform",
        name="MIMIC-III Waveform Database",
        source="physionet",
        landing_url="https://physionet.org/content/mimic3wdb/1.0/",
        access_url="https://physionet.org/content/mimic3wdb/1.0/",
        license="Open Data Commons Open Database License v1.0",
        domain="icu",
        description="Bedside ICU waveforms; not intraoperative anesthesia cases.",
        signals=("ART", "PPG", "ECG"),
        has_arterial_pressure=True,
        has_subject_ids=True,
        sampling_hz=125.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="Wrong care setting for the stated OR user; no anesthesia infusions.",
    ),
    DatasetCandidate(
        dataset_id="eicu-crd",
        name="eICU Collaborative Research Database",
        source="physionet",
        landing_url="https://physionet.org/content/eicu-crd/2.0/",
        access_url="https://physionet.org/content/eicu-crd/2.0/",
        license="PhysioNet Credentialed Health Data License",
        domain="icu",
        description="Multi-center ICU clinical database with vital-sign tables.",
        signals=("invasive blood pressure", "heart rate", "SpO2"),
        has_arterial_pressure=True,
        has_subject_ids=True,
        sampling_hz=0.016,
        programmatic_access=True,
        credentialed=True,
        time_series=True,
        notes="Credentialed; sparse vitals versus OR high-frequency tracks.",
    ),
    DatasetCandidate(
        dataset_id="pulsedb",
        name="PulseDB",
        source="github",
        landing_url="https://github.com/pulselabteam/PulseDB",
        access_url="https://github.com/pulselabteam/PulseDB",
        license="custom research terms",
        domain="icu-derived-abp-ppg",
        description="Calibrated PPG–ABP pairs derived from MIMIC and VitalDB.",
        signals=("ABP", "PPG"),
        has_arterial_pressure=True,
        has_subject_ids=True,
        sampling_hz=125.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="Only two channels; no OR context, infusions, or EtCO2.",
    ),
    DatasetCandidate(
        dataset_id="bidmc-ppg",
        name="BIDMC PPG and Respiration Dataset",
        source="physionet",
        landing_url="https://physionet.org/content/bidmc/1.0.0/",
        access_url="https://physionet.org/content/bidmc/1.0.0/",
        license="ODC-BY 1.0",
        domain="icu",
        description="PPG, impedance respiration, and ECG from critically ill adults.",
        signals=("PPG", "impedance respiration", "ECG"),
        has_arterial_pressure=False,
        has_subject_ids=True,
        sampling_hz=125.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="No arterial pressure series, so the MAP onset target cannot be built.",
    ),
    DatasetCandidate(
        dataset_id="wesad",
        name="WESAD",
        source="uci",
        landing_url="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
        access_url="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
        license="UCI terms",
        domain="wearable",
        description="Wearable stress and affect detection from laboratory subjects.",
        signals=("ACC", "BVP", "EDA", "TEMP"),
        has_arterial_pressure=False,
        has_subject_ids=True,
        sampling_hz=64.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="Wearable affect, not OR hypotension.",
    ),
    DatasetCandidate(
        dataset_id="cmapss",
        name="NASA C-MAPSS turbofan degradation",
        source="nasa",
        landing_url="https://www.nasa.gov/intelligent-systems-division/",
        access_url="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/",
        license="public domain",
        domain="industrial",
        description="Simulated turbofan run-to-failure sensor traces.",
        signals=("temperature", "pressure", "fan speed"),
        has_arterial_pressure=False,
        has_subject_ids=True,
        sampling_hz=1.0,
        programmatic_access=True,
        credentialed=False,
        time_series=True,
        notes="Industrial remaining-useful-life, not clinical.",
    ),
)


def seed_candidates() -> list[DatasetCandidate]:
    return [candidate for candidate in SEED_CATALOG]
