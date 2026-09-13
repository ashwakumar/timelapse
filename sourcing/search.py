"""Live dataset search tools used by the sourcing agent."""

from __future__ import annotations

import json
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sourcing.catalog import DatasetCandidate, seed_candidates
from sourcing.problem import ProblemCard


JsonFetcher = Callable[[str], object]

USER_AGENT = "timelapse-dataset-sourcing/0.1 (+https://github.com/ashwakumar/timelapse)"
HF_DATASETS = "https://huggingface.co/api/datasets"
REQUEST_TIMEOUT_SEC = 20


def default_json_fetcher(url: str) -> object:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_list(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("datasets", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _tags(item: dict) -> list[str]:
    tags = item.get("tags") or []
    if isinstance(tags, list):
        return [str(tag).lower() for tag in tags]
    return []


def huggingface_candidates(
    queries: tuple[str, ...],
    *,
    fetcher: JsonFetcher = default_json_fetcher,
    limit: int = 8,
    max_total: int = 20,
) -> list[DatasetCandidate]:
    found: dict[str, DatasetCandidate] = {}
    for query in queries:
        if len(found) >= max_total:
            break
    found: dict[str, DatasetCandidate] = {}
    for query in queries:
        url = f"{HF_DATASETS}?{urlencode({'search': query, 'limit': str(limit)})}"
        try:
            payload = fetcher(url)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
            continue
        for item in _as_list(payload):
            dataset_id = str(item.get("id") or "").strip()
            if not dataset_id or dataset_id in found:
                continue
            tags = _tags(item)
            text = " ".join(
                [
                    dataset_id,
                    str(item.get("description") or ""),
                    " ".join(tags),
                ]
            ).lower()
            arterial = any(
                token in text
                for token in ("abp", "map", "arterial", "blood-pressure", "blood pressure", "vitaldb")
            )
            medical = any(
                token in text
                for token in ("medical", "health", "ehr", "icu", "anesthes", "vital", "physionet")
            )
            time_series = "time-series" in tags or "time_series" in text or "waveform" in text
            found[dataset_id] = DatasetCandidate(
                dataset_id=f"hf:{dataset_id}",
                name=dataset_id,
                source="huggingface",
                landing_url=f"https://huggingface.co/datasets/{dataset_id}",
                access_url=f"https://huggingface.co/api/datasets/{dataset_id}",
                license=str(item.get("license") or "unknown"),
                domain="medical" if medical else "unknown",
                description=str(item.get("description") or item.get("id") or ""),
                signals=(),
                has_arterial_pressure=arterial,
                has_subject_ids="patient" in text or "subject" in text or arterial,
                sampling_hz=None,
                programmatic_access=True,
                credentialed="gated" in tags or bool(item.get("gated")),
                time_series=time_series or arterial or medical,
                notes=f"Hugging Face Hub search for {query!r}.",
                discovered_by="huggingface_hub",
                extra={"downloads": item.get("downloads"), "tags": tags[:24]},
            )
    return list(found.values())


def search_candidates(
    problem: ProblemCard,
    *,
    fetcher: JsonFetcher | None = None,
    include_huggingface: bool = True,
) -> list[DatasetCandidate]:
    """Merge the seed catalog with optional live Hub hits; first id wins."""

    merged: dict[str, DatasetCandidate] = {}
    for candidate in seed_candidates():
        merged[candidate.dataset_id] = candidate
    if include_huggingface:
        live_fetcher = fetcher or default_json_fetcher
        for candidate in huggingface_candidates(
            problem.hub_search_queries or problem.search_queries,
            fetcher=live_fetcher,
        ):
            merged.setdefault(candidate.dataset_id, candidate)
    return list(merged.values())
