"""TimeNet connector for the prepared VitalDB hypotension windows."""

from .targets import ANSWER_TEXT, QA_PROMPT

__all__ = [
    "ANSWER_TEXT",
    "CONNECTOR",
    "QA_PROMPT",
    "VitalDBHypotensionConnector",
]


def __getattr__(name: str):
    if name in {"CONNECTOR", "VitalDBHypotensionConnector"}:
        from .connector import CONNECTOR, VitalDBHypotensionConnector

        return CONNECTOR if name == "CONNECTOR" else VitalDBHypotensionConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
