"""Target user, clinical problem, and the dataset constraints they imply."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TargetUser:
    role: str
    setting: str
    job: str
    decision: str
    constraint: str


@dataclass(frozen=True)
class DatasetConstraints:
    """Hard requirements a source must meet before we will train on it."""

    must_be_time_series: bool = True
    must_include_arterial_pressure: bool = True
    must_have_subject_or_case_ids: bool = True
    must_be_programmatically_retrievable: bool = True
    must_allow_research_use: bool = True
    credentialed_access_ok: bool = False
    minimum_sampling_hz: float = 0.5
    preferred_domain: str = "intraoperative"


@dataclass(frozen=True)
class ProblemCard:
    id: str
    title: str
    target_user: TargetUser
    problem: str
    why_worth_solving: str
    language_task: str
    required_signals: tuple[str, ...]
    useful_signals: tuple[str, ...]
    constraints: DatasetConstraints = field(default_factory=DatasetConstraints)
    search_queries: tuple[str, ...] = ()
    hub_search_queries: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


PROBLEM = ProblemCard(
    id="intraoperative-hypotension-onset",
    title="Warn before a new intraoperative hypotensive episode becomes sustained",
    target_user=TargetUser(
        role="Anesthesiologist or anesthesia trainee",
        setting="Operating room, watching a multi-parameter physiologic monitor",
        job=(
            "Interpret the last seconds of vitals, anticipate whether mean arterial "
            "pressure is about to fall and stay down, and decide whether to reassess "
            "the patient and the signal quality."
        ),
        decision=(
            "Reassess now versus keep watching. The system must not invent a drug or "
            "dose from automatically generated labels."
        ),
        constraint=(
            "Only information available at the cutoff time may be shown. Splits must "
            "not leak across patients."
        ),
    ),
    problem=(
        "Intraoperative hypotension is common and associated with organ injury, but "
        "bedside alarms fire on the current threshold crossing. The useful question "
        "is earlier: given a short observation window, is a *new* sustained MAP < 65 "
        "mmHg episode about to start in 3, 5, 10, or 15 minutes?"
    ),
    why_worth_solving=(
        "A short-horizon, patient-held-out forecast from open OR telemetry can be "
        "shown to a clinician as INTERPRET / ANTICIPATE / ACT without claiming "
        "autonomous treatment. Generated research labels are explicit and auditable."
    ),
    language_task=(
        "Map a 20-second multivariate window plus observation masks to a short "
        "language answer: tightest onset horizon, or no event within 15 minutes."
    ),
    required_signals=("mean arterial pressure or arterial blood pressure",),
    useful_signals=(
        "heart rate",
        "SpO2",
        "end-tidal CO2",
        "opioid or hypnotic infusion",
        "EEG/BIS",
    ),
    search_queries=(
        "vitaldb intraoperative vital signs",
        "intraoperative hypotension arterial blood pressure time series",
        "anesthesia operating room MAP HR SpO2",
        "open dataset surgical patients multivariate waveforms",
        "physionet icu hypotension blood pressure",
    ),
    hub_search_queries=(
        "vitaldb",
        "hypotension",
        "intraoperative",
        "anesthesia",
        "physionet",
        "arterial blood pressure",
    ),
)
