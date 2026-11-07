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
import torch
from compare_message_aggregators import _device_runtime_metadata


def _write_progress(output: Path, result: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)


def _summarize(runs: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
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
    return summary


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    device_runtime = _device_runtime_metadata(config, torch.device(config["device"]))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    runner_sha256 = hashlib.sha256(Path(diagnose_training_budget.__file__).read_bytes()).hexdigest()
    comparison_path = Path(diagnose_training_budget.__file__).with_name("compare_message_aggregators.py")
    comparison_sha256 = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if output.exists():
        result = json.loads(output.read_text())
        for key, expected in (
            ("config_sha256", config_sha256),
            ("runner_sha256", runner_sha256),
            ("comparison_source_sha256", comparison_sha256),
            ("source_sha256", source_sha256),
        ):
            if result.get(key) != expected:
                raise RuntimeError(f"cannot resume validation run: {key} does not match frozen inputs")
        if result.get("run_state") != "in_progress":
            raise RuntimeError("refusing to overwrite a completed validation run")
    else:
        result = {
            "experiment_id": config["experiment_id"],
            "config_sha256": config_sha256,
            "runner_sha256": runner_sha256,
            "comparison_source_sha256": comparison_sha256,
            "source_sha256": source_sha256,
            "sealed_test_generated_or_read": False,
            "device": config["device"],
            "device_runtime": device_runtime,
            "require_rocm": bool(config.get("require_rocm", False)),
            "expected_accelerator_name": config.get("expected_accelerator_name"),
            "expected_torch_hip": config.get("expected_torch_hip"),
            "seeds": config["seeds"],
            "sample_counts_per_family": config["sample_counts_per_family"],
            "validation_sizes": config.get("validation_sizes", config["train_sizes"]),
            "extrapolation_sizes": config.get("extrapolation_sizes", {}),
            "run_state": "in_progress",
            "summary": {},
            "runs": [],
        }
        _write_progress(output, result)

    completed_pair_list = [
        (run["sample_count_per_family"], run["seed"]) for run in result["runs"]
    ]
    expected_pairs = {
        (sample_count, seed)
        for sample_count in config["sample_counts_per_family"]
        for seed in config["seeds"]
    }
    completed_pairs = set(completed_pair_list)
    if len(completed_pairs) != len(completed_pair_list) or not completed_pairs <= expected_pairs:
        raise RuntimeError("cannot resume validation run: progress file has duplicate or unknown seed entries")
    for sample_count in config["sample_counts_per_family"]:
        for seed in config["seeds"]:
            if (sample_count, seed) in completed_pairs:
                continue
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
                seed_result = diagnose_training_budget.run(variant_path, work / "results.json")
            result["runs"].append(
                {
                    "sample_count_per_family": sample_count,
                    "seed": seed,
                    "config_sha256": seed_result["config_sha256"],
                    "runner_sha256": seed_result["source_sha256"],
                    "comparison_source_sha256": seed_result["comparison_source_sha256"],
                    "device_runtime": seed_result["device_runtime"],
                    "candidate_results": seed_result["candidates"],
                }
            )
            completed_pairs.add((sample_count, seed))
            _write_progress(output, result)
            print(f"completed validation seed={seed}, samples_per_family={sample_count}", flush=True)

    if completed_pairs != expected_pairs or len(result["runs"]) != len(expected_pairs):
        raise RuntimeError("validation run is incomplete or has duplicate seeds")
    result["summary"] = _summarize(result["runs"], config)
    result["run_state"] = "completed"
    _write_progress(output, result)
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
