"""Held-out evaluation of a TSLM against a leakage-safe classical baseline."""

from evaluation.compare import compare_models
from evaluation.leakage import audit_prepared_splits

__all__ = ["audit_prepared_splits", "compare_models"]
