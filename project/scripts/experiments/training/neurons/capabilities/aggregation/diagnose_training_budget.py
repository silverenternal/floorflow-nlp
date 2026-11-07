"""Check aggregation learning curves on training and validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT / "src"))

from compare_message_aggregators import (
    SharedAggregationModel,
    _batch,
    _cost,
    _dataset,
    _evaluate,
    _normalized_training_loss,
    _permutation_delta,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    torch.use_deterministic_algorithms(bool(config.get("deterministic_algorithms", False)))
    device_name = config["device"]
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; fallback is disabled")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required; CPU fallback is disabled")
    seed = int(config["seed"])
    families = config["train_sizes"]
    train_samples = []
    validation: dict[str, list[dict[str, torch.Tensor]]] = {}
    validation_by_size: dict[str, dict[int, list[dict[str, torch.Tensor]]]] = {}
    for family_index, family in enumerate(families):
        train_samples.extend(
            _dataset(
                family,
                seed + family_index * 100_000,
                families[family],
                config["samples_per_family"],
            )
        )
        validation_sizes = config.get("validation_sizes", families)[family]
        validation[family] = _dataset(
            family,
            seed + 200_000 + family_index * 100_000,
            validation_sizes,
            config["validation_samples_per_family"],
        )
        validation_by_size[family] = {
            size: _dataset(
                family,
                seed + 250_000 + family_index * 100_000 + size * 31,
                [size],
                config["validation_samples_per_family"],
            )
            for size in validation_sizes
        }
    extrapolation = {
        family: _dataset(
            family,
            seed + 400_000 + list(families).index(family) * 100_000,
            sizes,
            config["validation_samples_per_family"],
        )
        for family, sizes in config.get("extrapolation_sizes", {}).items()
    }
    random.Random(seed).shuffle(train_samples)
    results: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "device": device_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_source_sha256": hashlib.sha256(Path(__file__).with_name("compare_message_aggregators.py").read_bytes()).hexdigest(),
        "seed": seed,
        "training_objective": config.get("training_objective", "normalized_mse"),
        "train_sample_count": len(train_samples),
        "validation_sample_count_per_family": config["validation_samples_per_family"],
        "sealed_test_generated_or_read": False,
        "candidates": {},
    }
    for name in config["candidates"]:
        torch.manual_seed(seed)
        model = SharedAggregationModel(hidden_size=config["hidden_size"], aggregator_kind=name).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        order_generator = torch.Generator().manual_seed(seed)
        candidate_result: dict[str, Any] = {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoints": [],
            "optimizer_steps": 0,
        }
        completed_epochs = 0
        started = time.perf_counter()
        for checkpoint_epoch in config["epoch_checkpoints"]:
            if checkpoint_epoch <= completed_epochs:
                raise ValueError("epoch_checkpoints must be strictly increasing")
            model.train()
            for _ in range(checkpoint_epoch - completed_epochs):
                order = torch.randperm(len(train_samples), generator=order_generator).tolist()
                for offset in range(0, len(order), config["batch_size"]):
                    batch_samples = [train_samples[index] for index in order[offset : offset + config["batch_size"]]]
                    batch = _batch(batch_samples, config["capacity"], device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    loss = _normalized_training_loss(
                        prediction,
                        batch["targets"],
                        batch["counts"],
                        objective=config.get("training_objective", "normalized_mse"),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite training loss for {name} at epoch {completed_epochs + 1}")
                    loss.backward()
                    optimizer.step()
                    candidate_result["optimizer_steps"] += 1
            completed_epochs = checkpoint_epoch
            _synchronize(device)
            elapsed = time.perf_counter() - started
            model.eval()
            train_metrics = {
                family: _evaluate(
                    model,
                    [sample for sample in train_samples if float(sample["family_id"]) == family_index],
                    capacity=config["capacity"],
                    batch_size=64,
                    device=device,
                )
                for family_index, family in enumerate(families)
            }
            validation_metrics = {
                family: _evaluate(
                    model,
                    validation[family],
                    capacity=config["capacity"],
                    batch_size=64,
                    device=device,
                )
                for family in families
            }
            permutation_metrics = {
                family: _permutation_delta(
                    model, validation[family], capacity=config["capacity"], device=device
                )
                for family in families
            }
            extrapolation_metrics = {
                family: _evaluate(
                    model,
                    samples,
                    capacity=max(config["capacity"], max((x["values"].numel() for x in samples), default=0)),
                    batch_size=64,
                    device=device,
                )
                for family, samples in extrapolation.items()
            }
            extrapolation_permutation_metrics = {
                family: _permutation_delta(
                    model,
                    samples,
                    capacity=max(
                        config["capacity"],
                        max((x["values"].numel() for x in samples), default=0),
                    ),
                    device=device,
                )
                for family, samples in extrapolation.items()
            }
            validation_size_metrics = {
                family: {
                    str(size): _evaluate(
                        model,
                        samples,
                        capacity=max(
                            config["capacity"],
                            max((x["values"].numel() for x in samples), default=0),
                        ),
                        batch_size=64,
                        device=device,
                    )
                    for size, samples in by_size.items()
                }
                for family, by_size in validation_by_size.items()
            }
            candidate_result["checkpoints"].append(
                {
                    "epoch": completed_epochs,
                    "optimizer_steps": candidate_result["optimizer_steps"],
                    "samples_seen": len(train_samples) * completed_epochs,
                    "elapsed_seconds": elapsed,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "validation_permutation_max_abs_delta": permutation_metrics,
                    "size_extrapolation_validation": extrapolation_metrics,
                    "size_extrapolation_permutation_max_abs_delta": extrapolation_permutation_metrics,
                    "validation_by_size": validation_size_metrics,
                }
            )
        _synchronize(device)
        candidate_result["total_training_seconds"] = time.perf_counter() - started
        if config.get("cost_widths"):
            candidate_result["forward_backward_cost"] = _cost(
                model, device=device, widths=config["cost_widths"]
            )
        results["candidates"][name] = candidate_result
        print(f"completed validation-only budget curve for {name}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "candidates": list(result["candidates"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
