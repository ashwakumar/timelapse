"""Training-run folders the demo can switch between."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def list_runs(root: Path = ROOT) -> list[dict]:
    catalog = _read_json(root / "artifacts/runs/catalog.json") or {"runs": []}
    rows = []
    for item in catalog.get("runs") or []:
        folder = root / item["dir"]
        metrics = _read_json(folder / "metrics.json") or []
        if not isinstance(metrics, list):
            metrics = []
        last = metrics[-1] if metrics else {}
        first = metrics[0] if metrics else {}
        rows.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "subtitle": item.get("subtitle"),
                "folder": item.get("dir"),
                "ready": (folder / "metrics.json").is_file(),
                "has_model": (folder / "best.pt").is_file(),
                "passes": int(last.get("epoch") or 0) if last else 0,
                "train_start": first.get("train_loss"),
                "train_end": last.get("train_loss"),
                "check_start": first.get("validation_loss"),
                "check_end": last.get("validation_loss") or last.get("best_validation_loss"),
                "log": [
                    {
                        "pass": int(row.get("epoch") or 0),
                        "train": row.get("train_loss"),
                        "check": row.get("validation_loss"),
                    }
                    for row in metrics
                ],
            }
        )
    return rows


def resolve_run(run_id: str | None, root: Path = ROOT) -> dict:
    rows = list_runs(root)
    if not rows:
        raise RuntimeError("No training runs listed in artifacts/runs/catalog.json")
    if run_id:
        for row in rows:
            if row["id"] == run_id:
                return row
        raise RuntimeError(f"Unknown run id: {run_id}")
    for preferred in ("50-epochs", "50-epochs-b", "3-epochs"):
        for row in rows:
            if row["id"] == preferred and row.get("has_model"):
                return row
    for row in rows:
        if row.get("has_model"):
            return row
    return rows[0]
