"""Byte-level dataset provenance and restart-safe run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 1024 * 1024


def _file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    """Encode strict JSON for deterministic comparisons and persistence."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Run artifact is not strict JSON: {exc}") from exc


def dataset_fingerprint(
    records: Sequence[Mapping[str, Any]], manifest_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Fingerprint the raw manifest and every distinct canonical signal file.

    The result contains paths and SHA-256 values only: it does not copy record
    identifiers, targets, or signal content. The aggregate digest is generated
    from canonical JSON after signal paths are sorted, making it independent of
    record iteration order while still binding the run to exact source paths.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    manifest_sha256 = _file_sha256(manifest)
    signal_paths: set[Path] = set()
    for index, record in enumerate(records):
        raw_path = record.get("signal_path")
        if not isinstance(raw_path, (str, os.PathLike)):
            raise TypeError(f"Record {index} has no valid signal_path")
        signal_paths.add(Path(raw_path).expanduser().resolve())
    if not signal_paths:
        raise ValueError("Cannot fingerprint a dataset without signal files")

    signal_files = [
        {"path": str(path), "sha256": _file_sha256(path)}
        for path in sorted(signal_paths, key=lambda item: str(item))
    ]
    content = {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "signal_files": signal_files,
    }
    return {
        "schema_version": 1,
        **content,
        "content_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def _artifact_path(root: Path, name: str) -> Path:
    """Resolve a relative artifact name without allowing path traversal."""

    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"Artifact name must be a safe relative path, got {name!r}")
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes output directory: {name!r}") from exc
    return destination


def _read_artifact(path: Path) -> bytes:
    """Read a JSON artifact and return its canonical representation."""

    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read run artifact {path}: {exc}") from exc
    return _canonical_json(value)


def _write_new_atomically(path: Path, canonical_payload: bytes) -> None:
    """Publish a new file atomically while refusing to replace an existing one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(canonical_payload + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_name, path)
    except FileExistsError:
        if _read_artifact(path) != canonical_payload:
            raise RuntimeError(f"Refusing to overwrite conflicting run artifact {path}")
    except OSError as exc:
        raise RuntimeError(
            f"Could not atomically write run artifact {path}: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def verify_run_artifacts(
    output_dir: str | os.PathLike[str],
    artifacts: dict[str, Any],
    resume_from: str | os.PathLike[str] | None,
) -> dict[str, Path]:
    """Validate provenance artifacts and atomically write only missing files.

    On resume, every expected artifact must exist beside the source checkpoint
    and match the freshly computed value before any destination is written.
    Existing destination artifacts must also match, whether this is a fresh or
    resumed run. Consequently, a new output directory can safely resume from a
    separate run directory without weakening restart validation.
    """

    if not artifacts:
        raise ValueError("At least one run artifact is required")
    root = Path(output_dir).expanduser().resolve()
    normalized: dict[str, bytes] = {}
    destinations: dict[str, Path] = {}
    claimed_destinations: set[Path] = set()
    for raw_name, payload in artifacts.items():
        if not isinstance(raw_name, str):
            raise TypeError("Artifact names must be strings")
        destination = _artifact_path(root, raw_name)
        if destination in claimed_destinations:
            raise ValueError(f"Duplicate artifact destination for {raw_name!r}")
        claimed_destinations.add(destination)
        normalized[raw_name] = _canonical_json(payload)
        destinations[raw_name] = destination

    if resume_from is not None:
        checkpoint = Path(resume_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"Resume checkpoint does not exist: {checkpoint}")
        source_root = checkpoint.parent
        for name, expected in normalized.items():
            source = _artifact_path(source_root, name)
            if not source.is_file():
                raise RuntimeError(f"Resume artifact is missing: {source}")
            if _read_artifact(source) != expected:
                raise RuntimeError(
                    f"Resume artifact does not match current inputs: {source}"
                )

    # Validate every existing destination before publishing any missing files.
    for name, destination in destinations.items():
        if destination.exists() and (
            not destination.is_file()
            or _read_artifact(destination) != normalized[name]
        ):
            raise RuntimeError(
                f"Refusing to overwrite conflicting run artifact {destination}"
            )

    root.mkdir(parents=True, exist_ok=True)
    for name, destination in destinations.items():
        if not destination.exists():
            _write_new_atomically(destination, normalized[name])
    return destinations


__all__ = ["dataset_fingerprint", "verify_run_artifacts"]
