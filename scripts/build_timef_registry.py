"""Build the workspace C-MAPSS connector through TimeNet's local engine."""

import argparse
from pathlib import Path

from timenet.client import TimeNet
from timenet.engine import run_pipeline

from aeroguard_connectors.cmapss import CMAPSSConnector


def build_and_verify(registry_dir: str = "artifacts/registry") -> Path:
    """Export the local connector directly, without upstream package discovery.

    Args:
        registry_dir: Destination local registry directory.

    Returns:
        Built dataset version directory after SDK verification.
    """
    version_dir = run_pipeline(CMAPSSConnector(), Path(registry_dir), force=True)
    TimeNet(registry=registry_dir).load("nasa/cmapss").describe()
    return version_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/registry")
    args = parser.parse_args()
    print(f"Built and verified: {build_and_verify(args.out)}")
