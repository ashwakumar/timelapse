"""Lazy-import TimeNet connector classes so the id map can be tested without the SDK."""

from __future__ import annotations

from importlib import import_module

# Challenge 1 writes selection.json with dataset_id. Challenge 2 looks up the
# connector here so a new source is a new folder plus one registry line.
SOURCED_DATASET_CONNECTORS: dict[str, str] = {
    "vitaldb": "connectors.vitaldb.hypotension_windows:VitalDBHypotensionConnector",
}


def load_selection(path) -> dict:
    """Read challenge-1 selection.json and require a registered connector."""

    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/source_datasets.py` first "
            "(challenge 1: find a problem and source the data)."
        )
    import json

    selection = json.loads(path.read_text())
    if not selection.get("validation_ok"):
        raise RuntimeError(f"{path} did not validate a retrievable dataset")
    dataset_id = selection.get("dataset_id")
    if dataset_id not in SOURCED_DATASET_CONNECTORS:
        known = ", ".join(sorted(SOURCED_DATASET_CONNECTORS))
        raise RuntimeError(
            f"Selected {dataset_id!r} has no TimeNet connector. Registered: {known}."
        )
    return selection


def connector_class_for(dataset_id: str):
    """Load the TimeNet connector registered for a sourcing selection."""

    from timenet.connectors import BaseConnector

    target = SOURCED_DATASET_CONNECTORS.get(dataset_id)
    if target is None:
        known = ", ".join(sorted(SOURCED_DATASET_CONNECTORS))
        raise KeyError(
            f"No TimeNet connector is registered for sourced dataset {dataset_id!r}. "
            f"Known connectors: {known}."
        )
    module_name, _, class_name = target.partition(":")
    module = import_module(module_name)
    connector_cls = getattr(module, class_name)
    if not issubclass(connector_cls, BaseConnector):
        raise TypeError(f"{target} is not a TimeNet BaseConnector")
    return connector_cls
