"""Search → assess → select → retrieve/validate loop for the problem card."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sourcing.assess import Assessment, assess_candidate, rank_assessments
from sourcing.catalog import DatasetCandidate
from sourcing.problem import PROBLEM, ProblemCard
from sourcing.retrieve import RetrievalReport, UrlFetcher, validate_candidate
from sourcing.search import JsonFetcher, search_candidates


@dataclass
class RoundLog:
    excluded: list[str]
    considered: str
    score: int
    hard_pass: bool
    validation_ok: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "excluded": list(self.excluded),
            "considered": self.considered,
            "score": self.score,
            "hard_pass": self.hard_pass,
            "validation_ok": self.validation_ok,
            "detail": self.detail,
        }


@dataclass
class SourcingReport:
    problem: dict
    selected: dict | None
    validation: dict | None
    assessments: list[dict]
    rounds: list[dict]
    search_errors: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "problem": self.problem,
            "selected": self.selected,
            "validation": self.validation,
            "assessments": self.assessments,
            "rounds": self.rounds,
            "search_errors": list(self.search_errors),
        }


class DataSourcingAgent:
    """Tool-using loop: search, score, attempt retrieval, exclude failures, repeat."""

    def __init__(
        self,
        problem: ProblemCard = PROBLEM,
        *,
        json_fetcher: JsonFetcher | None = None,
        url_fetcher: UrlFetcher | None = None,
        include_huggingface: bool = True,
        max_rounds: int = 6,
        candidates: list[DatasetCandidate] | None = None,
    ) -> None:
        self.problem = problem
        self.json_fetcher = json_fetcher
        self.url_fetcher = url_fetcher
        self.include_huggingface = include_huggingface
        self.max_rounds = max_rounds
        self.candidates = candidates

    def run(self) -> SourcingReport:
        generated_at = datetime.now(timezone.utc).isoformat()
        search_errors: list[str] = []
        if self.candidates is not None:
            candidates = list(self.candidates)
        else:
            try:
                candidates = search_candidates(
                    self.problem,
                    fetcher=self.json_fetcher,
                    include_huggingface=self.include_huggingface,
                )
            except Exception as error:  # pragma: no cover - defensive live-run guard
                search_errors.append(str(error))
                candidates = search_candidates(
                    self.problem,
                    fetcher=self.json_fetcher,
                    include_huggingface=False,
                )

        assessments = rank_assessments(
            [assess_candidate(candidate, self.problem) for candidate in candidates]
        )
        excluded: set[str] = set()
        rounds: list[RoundLog] = []
        selected_assessment: Assessment | None = None
        validation: RetrievalReport | None = None

        for _ in range(self.max_rounds):
            remaining = [
                item
                for item in assessments
                if item.hard_pass and item.candidate.dataset_id not in excluded
            ]
            if not remaining:
                break
            top = remaining[0]
            report = validate_candidate(top.candidate, fetcher=self.url_fetcher)
            rounds.append(
                RoundLog(
                    excluded=sorted(excluded),
                    considered=top.candidate.dataset_id,
                    score=top.score,
                    hard_pass=top.hard_pass,
                    validation_ok=report.ok,
                    detail="; ".join(
                        f"{check.name}={'pass' if check.passed else 'fail'}"
                        for check in report.checks
                    ),
                )
            )
            if report.ok:
                selected_assessment = top
                validation = report
                break
            excluded.add(top.candidate.dataset_id)

        return SourcingReport(
            problem=self.problem.to_dict(),
            selected=selected_assessment.to_dict() if selected_assessment else None,
            validation=validation.to_dict() if validation else None,
            assessments=[item.to_dict() for item in assessments],
            rounds=[item.to_dict() for item in rounds],
            search_errors=search_errors,
            generated_at=generated_at,
        )


def run_pipeline(
    output_dir: Path,
    *,
    include_huggingface: bool = True,
    json_fetcher: JsonFetcher | None = None,
    url_fetcher: UrlFetcher | None = None,
) -> SourcingReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = DataSourcingAgent(
        json_fetcher=json_fetcher,
        url_fetcher=url_fetcher,
        include_huggingface=include_huggingface,
    ).run()
    payload = report.to_dict()
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    selection = {
        "dataset_id": (report.selected or {}).get("dataset_id"),
        "name": (report.selected or {}).get("name"),
        "landing_url": ((report.selected or {}).get("candidate") or {}).get("landing_url"),
        "validation_ok": bool(report.validation and report.validation.get("ok")),
        "target_user": report.problem["target_user"],
        "problem": report.problem["problem"],
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search, assess, select, and validate a dataset for the OR hypotension task."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data_sourcing"),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only the seed catalog (no Hugging Face Hub search).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_pipeline(args.output_dir, include_huggingface=not args.offline)
    selected = report.selected["dataset_id"] if report.selected else None
    if not selected:
        raise SystemExit("No dataset passed search, assessment, and retrieval checks.")
    print(f"Selected {selected}; wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
