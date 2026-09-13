"""Agentic discovery, assessment, selection, and validation of time-series datasets."""

from sourcing.pipeline import DataSourcingAgent, run_pipeline
from sourcing.problem import PROBLEM

__all__ = ["PROBLEM", "DataSourcingAgent", "run_pipeline"]
