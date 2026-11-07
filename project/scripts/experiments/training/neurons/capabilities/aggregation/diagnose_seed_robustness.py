"""Measure sample-count effects on fresh validation seeds only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import diagnose_training_budget


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    runs = []
    for sample_count in config["sample_counts_per_family"]:
        for seed in config["seeds"]:
            variant = {
                key: value
                for key, value in config.items()
                if key not in {"sample_counts_per_family", "seeds"}
            }
            variant["samples_per_family"] = sample_count
            variant["seed"] = seed
            with tempfile.TemporaryDirectory(prefix="newera-validation-seed-") as directory:
                work = Path(directory)
                variant_path = work / "config.json"
                variant_path.write_text(json.dumps(variant, sort_keys=True))
                result = diagnose_training_budget.run(variant_path, work / "results.json")
            runs.append(
                {
                    "sample_count_per_family": sample_count,
                    "seed": seed,
                    "config_sha256": result["config_sha256"],
                    "candidate_results": result["candidates"],
                }
            )
            print(f"completed validation seed={seed}, samples_per_family={sample_count}", flush=True)

    summary = {}
    for sample_count in config["sample_counts_per_family"]:
        selected = [run for run in runs if run["sample_count_per_family"] == sample_count]
        candidate = config["candidates"][0]
        validation_metrics = [
            run["candidate_results"][candidate]["checkpoints"][-1]["validation"]["grouped_binding"]["mae"]
            for run in selected
        ]
        extrapolation_metrics = [
            run["candidate_results"][candidate]["checkpoints"][-1]["size_extrapolation_validation"]["grouped_binding"]["mae"]
            for run in selected
        ]
        extrapolation_by_size: dict[str, list[dict[str, float | int]]] = {}
        for run in selected:
            checkpoint = run["candidate_results"][candidate]["checkpoints"][-1]
            for size, metrics in checkpoint["validation_by_size"]["grouped_binding"].items():
                extrapolation_by_size.setdefault(size, []).append(metrics)
            for family, metrics in checkpoint["size_extrapolation_validation"].items():
                for size in config.get("extrapolation_sizes", {}).get(family, []):
                    extrapolation_by_size.setdefault(str(size), []).append(metrics)
        summary[str(sample_count)] = {
            "seed_count": len(selected),
            "validation_grouped_binding_mae_mean": statistics.mean(validation_metrics),
            "validation_grouped_binding_mae_max": max(validation_metrics),
            "extrapolation_grouped_binding_mae_mean": statistics.mean(extrapolation_metrics),
            "extrapolation_grouped_binding_mae_max": max(extrapolation_metrics),
            "validation_values_by_seed": validation_metrics,
            "extrapolation_values_by_seed": extrapolation_metrics,
            "extrapolation_by_size": {
                size: {
                    "mae_by_seed": [item["mae"] for item in metrics],
                    "mse_by_seed": [item["mse"] for item in metrics],
                    "max_absolute_error_by_seed": [item["max_absolute_error"] for item in metrics],
                    "mae_mean": statistics.mean(item["mae"] for item in metrics),
                    "mae_stdev": statistics.stdev(item["mae"] for item in metrics) if len(metrics) > 1 else 0.0,
                    "mae_max": max(item["mae"] for item in metrics),
                }
                for size, metrics in extrapolation_by_size.items()
            },
        }

    result = {
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(diagnose_training_budget.__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sealed_test_generated_or_read": False,
        "seeds": config["seeds"],
        "sample_counts_per_family": config["sample_counts_per_family"],
        "validation_sizes": config.get("validation_sizes", config["train_sizes"]),
        "extrapolation_sizes": config.get("extrapolation_sizes", {}),
        "summary": summary,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "run_count": len(result["runs"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
