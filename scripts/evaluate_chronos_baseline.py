"""Compatibility entry point for evaluating saved baseline models."""

from training.baseline_features import (
    extract_chronos_representations as extract_chronos_representations,
)
from training.evaluate_baselines import main

if __name__ == "__main__":
    main()
