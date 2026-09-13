"""Generate a comparison summary from measured benchmark results."""

import argparse
import json
from pathlib import Path


def generate_summary(
    benchmark_path: str | Path = "artifacts/benchmark_results.json",
    out_path: str | Path = "artifacts/model_comparison_summary.json",
) -> dict:
    benchmark = json.loads(Path(benchmark_path).read_text())
    models = benchmark["models"]
    summary = {
        "metadata": benchmark["configuration"],
        "source": str(benchmark_path),
        "models": models,
        "aeroguard_advantages_vs_chronos": {},
    }
    ours = models.get("AeroGuard TSLM", {})
    chronos = models.get("Chronos + Ridge", {})
    for metric, label in [("RMSE", "rmse_reduction_pct"), ("MAE", "mae_reduction_pct")]:
        baseline, value = chronos.get(metric), ours.get(metric)
        if baseline is not None and baseline > 0 and value is not None:
            summary["aeroguard_advantages_vs_chronos"][label] = round(
                (1 - value / baseline) * 100, 2
            )
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, metrics in models.items():
        print(f"{name}: RMSE={metrics['RMSE']}, coverage={metrics['coverage']:.1%}")
    print(f"Saved measured benchmark summary to {destination}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", default="artifacts/benchmark_results.json")
    parser.add_argument("--out-path", default="artifacts/model_comparison_summary.json")
    generate_summary(**vars(parser.parse_args()))
