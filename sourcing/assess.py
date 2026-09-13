"""Score dataset candidates against the frozen problem card."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sourcing.catalog import DatasetCandidate
from sourcing.problem import DatasetConstraints, ProblemCard

RESEARCH_LICENSES = (
    "cc by",
    "cc-by",
    "cc0",
    "odc",
    "odbl",
    "public domain",
    "mit",
    "apache",
    "bsd",
    "other",
    "custom research",
    "uci",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    weight: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    candidate: DatasetCandidate
    hard_pass: bool
    score: int
    checks: list[CheckResult] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.candidate.dataset_id,
            "name": self.candidate.name,
            "discovered_by": self.candidate.discovered_by,
            "hard_pass": self.hard_pass,
            "score": self.score,
            "checks": [check.to_dict() for check in self.checks],
            "rejection_reasons": list(self.rejection_reasons),
            "candidate": self.candidate.to_dict(),
        }


def _license_allows_research(license_name: str) -> bool:
    lowered = license_name.lower()
    if "credentialed" in lowered or "physionet credentialed" in lowered:
        return False
    return any(token in lowered for token in RESEARCH_LICENSES)


def assess_candidate(candidate: DatasetCandidate, problem: ProblemCard) -> Assessment:
    constraints: DatasetConstraints = problem.constraints
    checks: list[CheckResult] = []

    def add(name: str, passed: bool, weight: int, detail: str) -> None:
        checks.append(CheckResult(name=name, passed=passed, weight=weight, detail=detail))

    add(
        "time_series",
        candidate.time_series,
        25,
        "Multivariate or regularly sampled physiologic traces are required.",
    )
    add(
        "arterial_pressure",
        candidate.has_arterial_pressure,
        30,
        "MAP/ABP is required to label new hypotensive onsets.",
    )
    add(
        "subject_or_case_ids",
        candidate.has_subject_ids,
        15,
        "Patient or case identity is required for leakage-safe splits.",
    )
    add(
        "programmatic_access",
        candidate.programmatic_access,
        10,
        "Need a public API or file endpoint the rest of the repo can fetch.",
    )
    add(
        "research_license",
        _license_allows_research(candidate.license),
        10,
        f"License recorded as {candidate.license}.",
    )
    add(
        "uncredentialed" if not constraints.credentialed_access_ok else "access_policy",
        (not candidate.credentialed) or constraints.credentialed_access_ok,
        20,
        "Hackathon training needs data we can retrieve without a PhysioNet DUA wait.",
    )
    add(
        "sampling_rate",
        candidate.sampling_hz is None or candidate.sampling_hz >= constraints.minimum_sampling_hz,
        5,
        f"Sampling {candidate.sampling_hz} Hz versus minimum {constraints.minimum_sampling_hz} Hz.",
    )
    intraoperative = candidate.domain == constraints.preferred_domain
    add(
        "intraoperative_domain",
        intraoperative,
        20,
        f"Preferred domain is {constraints.preferred_domain}; candidate is {candidate.domain}.",
    )
    useful_hits = sum(
        1
        for signal in candidate.signals
        if any(
            token in signal.lower()
            for token in ("hr", "spo2", "etco2", "bis", "remi", "propofol", "co2")
        )
    )
    add(
        "supporting_or_signals",
        useful_hits >= 2,
        15 if useful_hits >= 2 else 0,
        f"{useful_hits} supporting OR channels beyond pressure.",
    )

    hard_names = {
        "time_series",
        "arterial_pressure",
        "subject_or_case_ids",
        "programmatic_access",
        "research_license",
        "uncredentialed",
        "intraoperative_domain",
    }
    rejection = [check.name for check in checks if check.name in hard_names and not check.passed]
    score = sum(check.weight for check in checks if check.passed)
    return Assessment(
        candidate=candidate,
        hard_pass=not rejection,
        score=score,
        checks=checks,
        rejection_reasons=rejection,
    )


def rank_assessments(assessments: list[Assessment]) -> list[Assessment]:
    return sorted(
        assessments,
        key=lambda item: (item.hard_pass, item.score, item.candidate.dataset_id == "vitaldb"),
        reverse=True,
    )
