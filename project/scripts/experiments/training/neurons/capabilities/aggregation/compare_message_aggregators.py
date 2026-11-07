"""Compare shared sum and PyG set attention on set and binding tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT / "src"))

from newera.capabilities.evaluation.functional_tasks import make_grouped_binding, make_set_sum
from newera.neuron.cells.aggregation import (
    ContextualSumResidualDeepSetsAggregator,
    ContextualSumResidualSetTransformerAggregator,
    DeepSetsMessageAggregator,
    GroupRoleDeepSetsAggregator,
    SetTransformerSetAggregator,
    SumResidualSetTransformerAggregator,
)
from newera.neuron.composition import ModelNeuron


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_runtime_metadata(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Fail closed when a configured accelerator does not match the experiment host."""
    if config.get("require_rocm") and device.type != "cuda":
        raise RuntimeError("ROCm CUDA device is required; CPU/MPS fallback is disabled")
    metadata: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP device is required; CPU fallback is disabled")
        accelerator_name = torch.cuda.get_device_name(device)
        expected_name = config.get("expected_accelerator_name")
        if expected_name and expected_name not in accelerator_name:
            raise RuntimeError(
                f"expected accelerator {expected_name!r}, got {accelerator_name!r}"
            )
        if config.get("require_rocm") and torch.version.hip is None:
            raise RuntimeError("ROCm PyTorch is required; CPU/CUDA fallback is disabled")
        expected_hip = config.get("expected_torch_hip")
        if expected_hip and torch.version.hip != expected_hip:
            raise RuntimeError(f"expected ROCm runtime {expected_hip!r}, got {torch.version.hip!r}")

        probe = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
        checksum = float((probe @ probe).sum().item())
        _synchronize(device)
        if not math.isfinite(checksum):
            raise RuntimeError("accelerator matrix-multiplication preflight returned non-finite output")
        metadata.update(
            {
                "accelerator_name": accelerator_name,
                "accelerator_preflight": "synchronized_matrix_multiplication",
                "accelerator_preflight_checksum": checksum,
            }
        )
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required; CPU fallback is disabled")
        metadata["accelerator_name"] = "Apple MPS"
    else:
        metadata["accelerator_name"] = "CPU"
    return metadata


def _select_task_output(values: Tensor, family_ids: Tensor) -> Tensor:
    """Decode one shared model's task-conditioned output channel."""
    if values.ndim != 2 or family_ids.shape != (values.shape[0],):
        raise ValueError("values and family_ids must have shapes [batch, outputs] and [batch]")
    channels = family_ids.to(device=values.device, dtype=torch.long)
    if torch.any(channels < 0) or torch.any(channels >= values.shape[1]):
        raise ValueError("task family ID must select an available output channel")
    channel_ids = torch.arange(values.shape[1], device=values.device)
    selected = channels.unsqueeze(1).eq(channel_ids.unsqueeze(0))
    return (values * selected.to(values.dtype)).sum(dim=1)


def _normalized_training_loss(
    prediction: Tensor,
    targets: Tensor,
    counts: Tensor,
    *,
    objective: str = "normalized_mse",
) -> Tensor:
    errors = (prediction - targets) / counts.sqrt()
    if objective == "normalized_mse":
        return errors.square().mean()
    if objective == "normalized_mae":
        return errors.abs().mean()
    raise ValueError(f"unsupported normalized training objective: {objective}")


class SharedAggregationModel(torch.nn.Module):
    def __init__(self, *, hidden_size: int, aggregator_kind: str) -> None:
        super().__init__()
        exact_input_sum_readout = aggregator_kind == "exact_raw_sum_bilinear_group_role_sum"
        isolated_sum_relation = aggregator_kind == "isolated_sum_relation_bilinear_group_role_sum"
        factorized_role_product = aggregator_kind == "factorized_role_product_group_sum"
        # Initialize the shared neuron core identically for every candidate;
        # experimental aggregator parameters must not shift core RNG draws.
        self.neuron = ModelNeuron(
            input_size=2,
            hidden_size=hidden_size,
            message_size=9,
            num_edge_types=2,
            exact_input_sum_readout=exact_input_sum_readout,
            exact_input_sum_output_channel=0 if (isolated_sum_relation or factorized_role_product) else None,
            aggregate_only_output_channels=(1,) if (isolated_sum_relation or factorized_role_product) else (),
            use_edge_type_embedding=not factorized_role_product,
        )
        if exact_input_sum_readout or isolated_sum_relation:
            aggregator_kind = "full_bilinear_group_role_sum"
        if aggregator_kind == "global_sum":
            aggregator = None
        elif aggregator_kind == "pyg_set_transformer":
            aggregator = SetTransformerSetAggregator(hidden_size, heads=4)
        elif aggregator_kind == "pyg_deep_sets":
            aggregator = DeepSetsMessageAggregator(hidden_size)
        elif aggregator_kind == "group_role_deep_sets":
            aggregator = GroupRoleDeepSetsAggregator(hidden_size, num_edge_types=2)
        elif aggregator_kind == "contextual_group_role_deep_sets":
            aggregator = GroupRoleDeepSetsAggregator(hidden_size, num_edge_types=2, context_size=2)
        elif aggregator_kind == "group_role_pairwise_deep_sets":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size, num_edge_types=2, pairwise_interactions=True
            )
        elif aggregator_kind == "sum_residual_group_role_pairwise":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size,
                num_edge_types=2,
                context_size=2,
                pairwise_interactions=True,
                sum_residual=True,
            )
        elif aggregator_kind == "sum_residual_group_role_bilinear":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size,
                num_edge_types=2,
                context_size=2,
                bilinear_interactions=True,
                sum_residual=True,
            )
        elif aggregator_kind == "exact_bilinear_group_role_sum":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size,
                num_edge_types=2,
                context_size=2,
                bilinear_interactions=True,
                exact_bilinear=True,
                sum_residual=True,
            )
        elif aggregator_kind == "full_bilinear_group_role_sum":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size,
                num_edge_types=2,
                context_size=2,
                bilinear_interactions=True,
                full_bilinear=True,
                full_bilinear_bias=not isolated_sum_relation,
                sum_residual=True,
            )
        elif aggregator_kind == "factorized_role_product_group_sum":
            aggregator = GroupRoleDeepSetsAggregator(
                hidden_size,
                num_edge_types=2,
                bilinear_interactions=True,
                exact_bilinear=True,
                group_network_bias=False,
            )
        elif aggregator_kind == "pyg_deep_sets_wide":
            aggregator = DeepSetsMessageAggregator(hidden_size, expansion=8)
        elif aggregator_kind == "contextual_sum_residual_deep_sets":
            aggregator = ContextualSumResidualDeepSetsAggregator(hidden_size, context_size=2)
        elif aggregator_kind == "contextual_sum_residual_deep_sets_wide":
            aggregator = ContextualSumResidualDeepSetsAggregator(hidden_size, context_size=2, expansion=8)
        elif aggregator_kind == "sum_residual":
            aggregator = SumResidualSetTransformerAggregator(hidden_size, heads=4)
        elif aggregator_kind == "contextual_sum_residual":
            aggregator = ContextualSumResidualSetTransformerAggregator(
                hidden_size, context_size=2, heads=4
            )
        else:
            raise ValueError(f"unsupported aggregator candidate: {aggregator_kind}")
        self.neuron.message_aggregator = aggregator

    def forward(self, sample: dict[str, Tensor]) -> Tensor:
        result = self.neuron.step(
            sample["context"],
            incoming_messages=sample["messages"],
            incoming_strengths=sample["strengths"],
            incoming_edge_types=sample["edge_types"],
            incoming_group_ids=sample["group_ids"],
        )
        return _select_task_output(result.value, sample["context"][:, 0])


def _make_sample(family: str, seed: int, size: int) -> dict[str, Tensor]:
    if family == "set_sum":
        sample = make_set_sum(size=size, seed=seed)
        groups = torch.arange(size)
        roles = torch.zeros(size, dtype=torch.long)
        family_id = 0.0
    else:
        sample = make_grouped_binding(group_count=size, seed=seed)
        relabel = torch.randperm(size, generator=torch.Generator().manual_seed(seed + 71_000))
        groups = relabel[sample.groups].float()
        roles = sample.roles
        family_id = 1.0
    label_generator = random.Random(seed + 91_000)
    opaque_labels = label_generator.sample(range(-(2**60), 2**60), len(groups.unique()))
    groups = torch.tensor([opaque_labels[int(group)] for group in groups], dtype=torch.long)
    values = sample.values.float()
    count = values.numel()
    return {
        "values": values,
        "group_ids": groups,
        "roles": roles,
        "target": sample.target.float().reshape(()),
        "family_id": torch.tensor(family_id),
        "count": torch.tensor(count),
    }


def _batch(samples: list[dict[str, Tensor]], capacity: int, device: torch.device) -> dict[str, Tensor]:
    batch = len(samples)
    messages = torch.zeros((batch, capacity, 9), device=device)
    strengths = torch.zeros((batch, capacity), device=device)
    edge_types = torch.zeros((batch, capacity), dtype=torch.long, device=device)
    group_ids = torch.zeros((batch, capacity), dtype=torch.long, device=device)
    contexts = torch.zeros((batch, 2), device=device)
    targets = torch.zeros(batch, device=device)
    counts = torch.zeros(batch, device=device)
    for row, sample in enumerate(samples):
        count = sample["values"].numel()
        messages[row, :count, 0] = sample["values"].to(device)
        group_ids[row, :count] = sample["group_ids"].to(device)
        strengths[row, :count] = 1.0
        edge_types[row, :count] = sample["roles"].to(device)
        contexts[row, 0] = sample["family_id"].to(device)
        contexts[row, 1] = math.log1p(count) / math.log1p(capacity)
        targets[row] = sample["target"].to(device)
        counts[row] = count
    return {
        "context": contexts,
        "messages": messages,
        "strengths": strengths,
        "edge_types": edge_types,
        "group_ids": group_ids,
        "targets": targets,
        "counts": counts,
    }


def _dataset(family: str, seed: int, sizes: list[int], count: int) -> list[dict[str, Tensor]]:
    rng = random.Random(seed)
    return [
        _make_sample(family, seed + index * 17, sizes[rng.randrange(len(sizes))])
        for index in range(count)
    ]


def _train(
    model: SharedAggregationModel,
    train_samples: list[dict[str, Tensor]],
    *,
    seed: int,
    capacity: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
    objective: str = "normalized_mse",
) -> dict[str, float | int]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    order_gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_samples), generator=order_gen).tolist()
    losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        if epoch:
            order = torch.randperm(len(train_samples), generator=order_gen).tolist()
        for offset in range(0, len(order), batch_size):
            samples = [train_samples[index] for index in order[offset : offset + batch_size]]
            batch = _batch(samples, capacity, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = _normalized_training_loss(
                prediction,
                batch["targets"],
                batch["counts"],
                objective=objective,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
    _synchronize(device)
    return {
        "training_seconds": time.perf_counter() - started,
        "samples_seen": len(train_samples) * epochs,
        "epochs": epochs,
        "optimizer_steps": math.ceil(len(train_samples) / batch_size) * epochs,
        "mean_batch_loss": statistics.mean(losses),
    }


def _evaluate(
    model: SharedAggregationModel,
    samples: list[dict[str, Tensor]],
    *,
    capacity: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    errors: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(samples), batch_size):
            batch = _batch(samples[offset : offset + batch_size], capacity, device)
            prediction = model(batch)
            errors.extend((prediction - batch["targets"]).abs().cpu().tolist())
            targets.extend(batch["targets"].abs().cpu().tolist())
    mae = statistics.mean(errors)
    target_mean = statistics.mean(targets)
    return {
        "sample_count": len(samples),
        "mae": mae,
        "normalized_mae": mae / target_mean if target_mean > 1e-12 else None,
        "mse": statistics.mean(error * error for error in errors),
        "max_absolute_error": max(errors),
        "zero_baseline_mae": target_mean,
        "oracle_mae": 0.0,
    }


def _permutation_delta(
    model: SharedAggregationModel,
    samples: list[dict[str, Tensor]],
    *,
    capacity: int,
    device: torch.device,
) -> float:
    model.eval()
    deltas = []
    with torch.no_grad():
        for index, sample in enumerate(samples[:64]):
            order = torch.randperm(sample["values"].numel(), generator=torch.Generator().manual_seed(index + 88000))
            permuted = dict(sample)
            for name in ("values", "group_ids", "roles"):
                permuted[name] = sample[name][order]
            left = model(_batch([sample], capacity, device))[0]
            right = model(_batch([permuted], capacity, device))[0]
            deltas.append(abs(float(left.item() - right.item())))
    return max(deltas, default=0.0)


def _cost(model: SharedAggregationModel, *, device: torch.device, widths: list[int]) -> dict[str, Any]:
    costs: dict[str, Any] = {"parameter_count": sum(p.numel() for p in model.parameters())}
    model.train()
    for width in widths:
        sample = _make_sample("grouped_binding", 99000 + width, max(1, width // 2))
        batch = _batch([sample], width, device)
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            model(batch).sum().backward()
        _synchronize(device)
        times = []
        for _ in range(15):
            model.zero_grad(set_to_none=True)
            _synchronize(device)
            started = time.perf_counter()
            model(batch).sum().backward()
            _synchronize(device)
            times.append((time.perf_counter() - started) * 1000)
        costs[str(width)] = {
            "median_forward_backward_ms": statistics.median(times),
            "p95_forward_backward_ms": sorted(times)[int(0.95 * (len(times) - 1))],
        }
    return costs


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_formal_progress(
    output: Path,
    *,
    config_sha256: str,
    source_sha256: str,
    device_runtime: dict[str, Any],
    expected_seeds: list[int],
    initial_results: dict[str, Any],
) -> dict[str, Any]:
    if not output.exists():
        return initial_results
    progress = json.loads(output.read_text())
    if progress.get("run_state") == "completed":
        raise RuntimeError("refusing to rerun a completed sealed confirmation")
    if progress.get("run_state") != "in_progress":
        raise RuntimeError("sealed confirmation progress has an invalid run state")
    if progress.get("config_sha256") != config_sha256 or progress.get("source_sha256") != source_sha256:
        raise RuntimeError("cannot resume sealed confirmation with different config or source hashes")
    if progress.get("device_runtime") != device_runtime:
        raise RuntimeError("cannot resume sealed confirmation on a different accelerator runtime")
    completed_seeds = [entry.get("seed") for entry in progress.get("seeds", [])]
    if len(completed_seeds) != len(set(completed_seeds)) or not set(completed_seeds) <= set(expected_seeds):
        raise RuntimeError("sealed confirmation progress contains duplicate or unexpected seeds")
    active_seed = progress.get("active_seed")
    if active_seed is not None:
        if active_seed not in expected_seeds:
            raise RuntimeError("sealed confirmation progress contains an unexpected active seed")
        if progress.get("sealed_access_started"):
            raise RuntimeError(
                "refusing to repeat a seed whose sealed evaluation started; preserve the partial run and stop"
            )
        if active_seed in completed_seeds:
            raise RuntimeError("sealed confirmation active seed is already marked complete")
        progress["active_seed"] = None
    elif progress.get("sealed_access_started"):
        raise RuntimeError("sealed confirmation progress has an access marker without an active seed")
    progress["sealed_access_started"] = False
    return progress


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    torch.use_deterministic_algorithms(bool(config.get("deterministic_algorithms", False)))
    device_name = config["device"]
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this experiment; fallback is disabled")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for this experiment; CPU fallback is disabled")
    runtime_device = _device_runtime_metadata(config, device)
    capacity = int(config["data"].get("capacity", 64))
    families = config["data"].get(
        "families", {"set_sum": [4, 16, 64], "grouped_binding": [2, 4, 8]}
    )
    held_out = config["data"].get("held_out", {"grouped_binding": 16})
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected_seeds = [config["data"]["seed_base"] + index for index in config["training"]["seeds"]]
    results: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "device": device_name,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "accelerator_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda"
            else "Apple MPS" if device.type == "mps"
            else "CPU"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "shared_model_contract": "one ModelNeuron and one weight set trained jointly on both families",
        "weight_sets_per_model": 1,
        "output_decoder": "observable task family selects a channel from the single ModelNeuron output_projection; no per-task model or extra decoder weights",
        "sealed_test_used": True,
        "training_objective": config["training"].get("objective", "normalized_mse"),
        "device_runtime": runtime_device,
        "run_state": "in_progress",
        "active_seed": None,
        "sealed_access_started": False,
        "seeds": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    results = _load_formal_progress(
        output,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        device_runtime=runtime_device,
        expected_seeds=expected_seeds,
        initial_results=results,
    )
    completed_seeds = {entry["seed"] for entry in results["seeds"]}
    _atomic_write_json(output, results)
    for seed_index in config["training"]["seeds"]:
        seed = config["data"]["seed_base"] + seed_index
        if seed in completed_seeds:
            continue
        results["active_seed"] = seed
        results["sealed_access_started"] = False
        _atomic_write_json(output, results)
        train_samples = []
        validation: dict[str, list[dict[str, Tensor]]] = {}
        extrapolation: dict[str, list[dict[str, Tensor]]] = {}
        for family_index, family in enumerate(families):
            train_samples.extend(
                _dataset(
                    family,
                    seed + family_index * 100_000,
                    families[family],
                    config["data"]["train_samples_per_family"],
                )
            )
            validation[family] = _dataset(
                family,
                seed + 200_000 + family_index * 100_000,
                families[family],
                config["data"]["validation_samples_per_family"],
            )
            if family in held_out:
                extrapolation[family] = _dataset(
                    family,
                    seed + 400_000 + family_index * 100_000,
                    [held_out[family]],
                    config["data"]["validation_samples_per_family"],
                )
        random.Random(seed).shuffle(train_samples)
        seed_result: dict[str, Any] = {"seed": seed, "candidates": {}}
        models: dict[str, SharedAggregationModel] = {}
        for name in config["candidates"]:
            torch.manual_seed(seed)
            model = SharedAggregationModel(
                hidden_size=config["model"]["hidden_size"],
                aggregator_kind=name,
            ).to(device)
            train = _train(
                model,
                train_samples,
                seed=seed,
                capacity=capacity,
                batch_size=config["training"]["batch_size"],
                epochs=config["training"]["epochs"],
                device=device,
                objective=config["training"].get("objective", "normalized_mse"),
            )
            candidate: dict[str, Any] = {
                "training": train,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "validation": {},
                "sealed_test": {},
                "size_extrapolation_validation": {},
            }
            for family in families:
                candidate["validation"][family] = _evaluate(
                    model, validation[family], capacity=capacity, batch_size=64, device=device
                )
                if family in extrapolation:
                    candidate["size_extrapolation_validation"][family] = _evaluate(
                        model,
                        extrapolation[family],
                        capacity=max(capacity, max(x["values"].numel() for x in extrapolation[family])),
                        batch_size=64,
                        device=device,
                    )
                    candidate["size_extrapolation_validation"][family]["permutation_max_abs_delta"] = _permutation_delta(
                        model,
                        extrapolation[family],
                        capacity=max(capacity, max(x["values"].numel() for x in extrapolation[family])),
                        device=device,
                    )
            candidate["cost"] = _cost(
                model,
                device=device,
                widths=config.get("cost_widths", [4, 16, 64, 256]),
            )
            seed_result["candidates"][name] = candidate
            models[name] = model

        # Persist the once-only access marker before generating or evaluating any sealed examples.
        results["active_seed"] = seed
        results["active_seed_result"] = seed_result
        results["sealed_access_started"] = True
        _atomic_write_json(output, results)
        test: dict[str, list[dict[str, Tensor]]] = {}
        for family_index, family in enumerate(families):
            test[family] = _dataset(
                family,
                seed + 300_000 + family_index * 100_000,
                families[family],
                config["data"]["sealed_test_samples_per_family"],
            )
        for name, model in models.items():
            candidate = seed_result["candidates"][name]
            for family in families:
                candidate["sealed_test"][family] = _evaluate(
                    model, test[family], capacity=capacity, batch_size=64, device=device
                )
                candidate["sealed_test"][family]["permutation_max_abs_delta"] = _permutation_delta(
                    model, test[family], capacity=capacity, device=device
                )
        results["seeds"].append(seed_result)
        results.pop("active_seed_result", None)
        completed_seeds.add(seed)
        results["active_seed"] = None
        results["sealed_access_started"] = False
        _atomic_write_json(output, results)
        print(f"completed seed {seed}; graph candidates evaluated", flush=True)
    if completed_seeds != set(expected_seeds):
        raise RuntimeError("sealed confirmation is incomplete or contains duplicate seeds")
    summary: dict[str, Any] = {}
    for name in config["candidates"]:
        summary[name] = {}
        for split in ("validation", "sealed_test", "size_extrapolation_validation"):
            summary[name][split] = {}
            split_families = held_out if split == "size_extrapolation_validation" else families
            for family in split_families:
                summary[name][split][family] = {
                    "mae_mean": statistics.mean(
                        seed["candidates"][name][split][family]["mae"] for seed in results["seeds"]
                    ),
                    "normalized_mae_mean": statistics.mean(
                        seed["candidates"][name][split][family]["normalized_mae"] for seed in results["seeds"]
                    ),
                    "mse_mean": statistics.mean(
                        seed["candidates"][name][split][family]["mse"] for seed in results["seeds"]
                    ),
                    "permutation_max_abs_delta_mean": (
                        statistics.mean(
                            seed["candidates"][name][split][family].get("permutation_max_abs_delta", 0.0)
                            for seed in results["seeds"]
                        )
                        if split == "sealed_test"
                        else None
                    ),
                }
    if "global_sum" in config["candidates"] and "grouped_binding" in families:
        paired = []
        for seed in results["seeds"]:
            baseline = seed["candidates"]["global_sum"]["sealed_test"]["grouped_binding"]["mse"]
            for name in config["candidates"]:
                if name == "global_sum":
                    continue
                candidate_mse = seed["candidates"][name]["sealed_test"]["grouped_binding"]["mse"]
                paired.append((name, 1.0 - candidate_mse / baseline))
        paired_summary = {}
        for name in config["candidates"]:
            values = [effect for candidate, effect in paired if candidate == name]
            if values:
                mean = statistics.mean(values)
                sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
                critical_value = 2.571 if len(values) == 6 else 1.96
                paired_summary[name] = {
                    "relative_binding_mse_improvement_mean": mean,
                    "relative_binding_mse_improvement_95ci": [
                        mean - critical_value * sem,
                        mean + critical_value * sem,
                    ],
                    "seed_count": len(values),
                }
        results["paired_binding_mse_effect"] = paired_summary
    results["summary"] = summary
    results["run_state"] = "completed"
    results["active_seed"] = None
    results["sealed_access_started"] = False
    _atomic_write_json(output, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "seeds": len(result["seeds"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
