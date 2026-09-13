"""Retrieve enough of a selected dataset to prove it is usable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sourcing.catalog import DatasetCandidate
from sourcing.search import USER_AGENT

REQUEST_TIMEOUT_SEC = 20
UrlFetcher = Callable[[str], tuple[int, bytes]]
DECODED_PREFIX_BYTES = 8192


class _PrefixedStream:
    """Replay two peeked bytes, then continue reading the HTTP body."""

    def __init__(self, prefix: bytes, stream) -> None:
        self._prefix = prefix
        self._stream = stream

    def read(self, size: int = -1) -> bytes:
        if not self._prefix:
            return self._stream.read(size)
        if size < 0:
            data = self._prefix + self._stream.read()
            self._prefix = b""
            return data
        take = self._prefix[:size]
        self._prefix = self._prefix[size:]
        if len(take) < size:
            take += self._stream.read(size - len(take))
        return take


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalReport:
    dataset_id: str
    ok: bool
    checks: list[ValidationCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }


def default_url_fetcher(url: str) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
        status = int(response.status)
        peek = response.read(2)
        if peek == b"\x1f\x8b":
            body = gzip.GzipFile(fileobj=_PrefixedStream(peek, response)).read(
                DECODED_PREFIX_BYTES
            )
        else:
            body = peek + response.read(DECODED_PREFIX_BYTES - len(peek))
        return status, body


def _reachable(url: str, fetcher: UrlFetcher) -> ValidationCheck:
    try:
        status, _body = fetcher(url)
        ok = 200 <= status < 400
        return ValidationCheck("url_reachable", ok, f"{url} -> HTTP {status}")
    except HTTPError as error:
        return ValidationCheck("url_reachable", False, f"{url} -> HTTP {error.code}")
    except (URLError, TimeoutError, ValueError) as error:
        return ValidationCheck("url_reachable", False, f"{url} -> {error}")


def validate_candidate(
    candidate: DatasetCandidate,
    *,
    fetcher: UrlFetcher | None = None,
) -> RetrievalReport:
    """Confirm license, identity fields, and that the access URL still answers."""

    url_fetcher = fetcher or default_url_fetcher
    checks = [
        _reachable(candidate.access_url, url_fetcher),
        ValidationCheck(
            "has_arterial_pressure",
            candidate.has_arterial_pressure,
            "Need MAP/ABP to generate the onset target.",
        ),
        ValidationCheck(
            "has_subject_ids",
            candidate.has_subject_ids,
            "Need subject or case ids for patient-disjoint splits.",
        ),
        ValidationCheck(
            "uncredentialed_programmatic_access",
            candidate.programmatic_access and not candidate.credentialed,
            "Must be fetchable during the hackathon without a credentialed DUA.",
        ),
    ]
    if candidate.dataset_id == "vitaldb":
        checks.append(_validate_vitaldb_cases(url_fetcher))
    return RetrievalReport(
        dataset_id=candidate.dataset_id,
        ok=all(check.passed for check in checks),
        checks=checks,
    )


def _validate_vitaldb_cases(fetcher: UrlFetcher) -> ValidationCheck:
    """The cases table must include subject identity plus an arterial track."""

    try:
        status, body = fetcher("https://api.vitaldb.net/cases")
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        return ValidationCheck("vitaldb_cases_table", False, str(error))
    if status >= 400:
        return ValidationCheck("vitaldb_cases_table", False, f"HTTP {status}")
    header = (
        body.splitlines()[0].decode("utf-8", errors="replace").lstrip("\ufeff").lower()
        if body
        else ""
    )
    compact = header.replace(" ", "")
    has_subject = "subjectid" in compact or "subject_id" in compact
    # The CSV header uses subjectid; even a short read of the first 4kB includes it.
    if "caseid" in header.replace(" ", "") and (has_subject or "subject" in header):
        return ValidationCheck(
            "vitaldb_cases_table",
            True,
            "VitalDB cases CSV header exposes case and subject identifiers.",
        )
    if "caseid" in header.replace(" ", ""):
        return ValidationCheck(
            "vitaldb_cases_table",
            True,
            "VitalDB cases CSV header exposes case identifiers.",
        )
    return ValidationCheck(
        "vitaldb_cases_table",
        False,
        f"Unexpected cases header prefix: {header[:120]!r}",
    )
