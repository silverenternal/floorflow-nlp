"""Train and evaluate a shared-weight model neuron over sparse graph identities."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
GRAPH_SCRIPTS = ROOT / "project/scripts/experiments/training/neurons/graph"
sys.path.insert(0, str(GRAPH_SCRIPTS))
import run_neuron_graph_sparse_evolution as graph_proxy

from newera.capabilities.contracts import Capability, TaskAdapter
from newera.capabilities.evaluation import (
    execute_task,
    load_checkpoint,
    sample_indices,
    save_checkpoint,
    train_with_accumulation,
    write_json_atomic,
)
from newera.neuron import ModelNeuron, ModelNeuronGraph
from newera.neuron.graph import (
    LinkProposalPolicy,
    NeuronGraphTopology,
    NeuronIdentityRegistry,
    TopologyController,
)

DEFAULT_CONFIG = ROOT / "project/configs/models_training/learning/neuron_comparisons/topology_evolution/modelized_neuron_training_baseline.json"


class SharedWeightNeuronTask(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, edge_types: int) -> None:
        super().__init__()
        self.neuron_graph = ModelNeuronGraph(
            ModelNeuron(feature_size, hidden_size, hidden_size, num_edge_types=edge_types)
        )
        self.link_policy = LinkProposalPolicy(
            hidden_size, edge_types, initial_keep_bias=0.0, predict_strengths=True
        )
        self.readout = nn.Linear(hidden_size, 1)
        self.hard_action_active = False
        self.edge_types = edge_types

    def forward(
        self,
        data: dict[str, Any],
        *,
        hard: bool = False,
        threshold: float = 0.5,
        state: Any = None,
    ) -> dict[str, Any]:
        features, node_ids = data["features"], data["node_ids"]
        if features.shape[0] != 1:
            raise ValueError("the v9 runner currently advances one topology graph per call")
        node_count = features.shape[1]
        order = torch.argsort(node_ids, dim=1, stable=True)
        memory_count = int(data["memory_count"])
        source_slots = order[:, -memory_count - 1 : -1]
        readout_slots = order[:, -1:]
        candidates = torch.stack(
            (source_slots.reshape(-1), readout_slots.expand_as(source_slots).reshape(-1))
        )
        topology = NeuronGraphTopology(
            1,
            node_count,
            data["initial_edge_index"],
            data["initial_edge_type"],
            data["initial_strength"],
            node_ids,
        )
        first = self.neuron_graph(features, topology, state=state)
        logits = self.link_policy(
            first.values, candidates, topology, causal=False, check_values=False
        )
        strengths = self.link_policy.predict_strengths(
            first.values, candidates, topology, causal=False, check_values=False
        )
        if hard:
            mutations = self.link_policy.propose(
                logits,
                candidates,
                topology,
                threshold=threshold,
                candidate_strengths=strengths,
                strength_update_threshold=data.get("strength_update_threshold", 0.05),
            )
            next_topology = TopologyController(
                max_edges=data.get("max_edges", node_count * node_count), causal=False
            ).apply(topology, mutations, num_edge_types=self.edge_types)
            second = self.neuron_graph(features, next_topology, state=first.state)
            candidate_weights = None
        else:
            candidate_weights = self.link_policy.expected_type_weights(
                logits,
                candidates,
                topology,
                candidate_strengths=strengths,
                straight_through=self.training and self.hard_action_active,
            )
            mutations = []
            next_topology = topology
            second = self.neuron_graph(
                features,
                topology,
                state=first.state,
                candidate_edges=candidates,
                candidate_type_weights=candidate_weights.unsqueeze(0),
            )
        readout_index = node_ids.argmax(dim=1).view(1, 1, 1)
        readout_value = second.values.gather(
            1, readout_index.expand(-1, -1, second.values.shape[-1])
        ).squeeze(1)
        return {
            "prediction": self.readout(readout_value).squeeze(-1),
            "logits": logits,
            "strengths": strengths,
            "candidate_edges": candidates,
            "candidate_weights": candidate_weights,
            "topology": next_topology,
            "mutations": mutations,
            "state": second.state,
        }


class FlatControl(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.key_space = 0
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features.flatten(start_dim=1)).squeeze(-1)


class _ModuleCapability:
    """Expose a trainable module through the project's small capability contract."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def execute(self, inputs: Any) -> Any:
        return self.model(inputs)


class _GraphTaskAdapter:
    def encode(self, task_input: dict[str, Any]) -> dict[str, Any]:
        return task_input

    def decode(self, capability_output: dict[str, Any]) -> Tensor:
        return capability_output["prediction"].reshape(-1)


class _FlatTaskAdapter:
    def __init__(self, key_space: int) -> None:
        self.key_space = key_space

    def encode(self, task_input: dict[str, Any]) -> Tensor:
        return graph_proxy._canonical_features(
            task_input["features"], task_input["node_ids"], self.key_space
        )

    def decode(self, capability_output: Tensor) -> Tensor:
        return capability_output


def _data(
    seed: int,
    count: int,
    *,
    memory_count: int,
    task: str,
    config: dict[str, Any],
    device: torch.device,
    identities: dict[int, int],
    readout_identity: int,
    memory_id_pool: list[int] | None = None,
) -> dict[str, Any]:
    raw = graph_proxy._data(
        seed,
        count,
        memory_count=memory_count,
        task=task,
        key_space=config["task"]["key_space"],
        device=device,
        identity_ids=identities,
        readout_identity=readout_identity,
        memory_id_pool=memory_id_pool,
    )
    old_nodes = raw["features"].shape[1]
    nodes = memory_count + 1
    valid = raw["node_ids"].ge(0)
    # Stable sort by validity compacts active neurons while preserving the random slot order.
    order = torch.argsort(valid.to(torch.long), dim=1, stable=True)[:, -nodes:]
    features = raw["features"].gather(1, order.unsqueeze(-1).expand(-1, -1, raw["features"].shape[-1]))
    node_ids = raw["node_ids"].gather(1, order)
    inverse = torch.full_like(raw["node_ids"], -1)
    ranks = torch.arange(nodes, device=device).expand(count, -1)
    inverse.scatter_(1, order, ranks)
    edge_index = raw["initial_edge_index"]
    batches = edge_index[0] // old_nodes
    local_sources = edge_index[0] % old_nodes
    local_targets = edge_index[1] % old_nodes
    remapped_sources = batches * nodes + inverse[batches, local_sources]
    remapped_targets = batches * nodes + inverse[batches, local_targets]
    return {
        **raw,
        "features": features,
        "node_ids": node_ids,
        "initial_edge_index": torch.stack((remapped_sources, remapped_targets)),
        "memory_count": memory_count,
        "key_space": config["task"]["key_space"],
        "strength_update_threshold": config["model"]["strength_update_threshold"],
        "max_edges": config["model"]["max_edges"],
    }


def _oracle_targets(data: dict[str, Any], key_space: int) -> Tensor:
    """Recompute labels from identity-bound source values and the declared target edges."""
    bit_width = math.ceil(math.log2(key_space))
    predictions: list[float] = []
    node_ids = data["node_ids"].detach().cpu().tolist()
    features = data["features"].detach().cpu().tolist()
    for sample_index, edges in enumerate(data["oracle_edges"]):
        identity_to_slot = {identity: slot for slot, identity in enumerate(node_ids[sample_index])}
        value = 0.0
        for source_identity, _edge_type, strength in edges:
            source_slot = identity_to_slot[source_identity]
            value += strength * features[sample_index][source_slot][bit_width]
        predictions.append(value)
    return torch.tensor(predictions, dtype=data["labels"].dtype, device=data["labels"].device)


def _topology(data: dict[str, Any]) -> NeuronGraphTopology:
    return NeuronGraphTopology(
        1,
        data["features"].shape[1],
        data["initial_edge_index"],
        data["initial_edge_type"],
        data["initial_strength"],
        data["node_ids"],
    )


def _ops_expected(data: dict[str, Any], topology: NeuronGraphTopology) -> dict[tuple[int, int], str]:
    if "expected_operations" in data:
        return data["expected_operations"][0]
    ids = data["node_ids"][0].detach().cpu().tolist()
    current: dict[tuple[int, int], tuple[int, float]] = {}
    for index, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist()):
        current[(ids[source], ids[target])] = (
            int(topology.edge_type[index]), float(topology.strength[index])
        )
    readout_id = max(ids)
    desired = {
        (source_id, readout_id): (edge_type, strength)
        for source_id, edge_type, strength in data["oracle_edges"][0]
    }
    result = {}
    for edge, (edge_type, strength) in desired.items():
        present = current.get(edge)
        if present is None:
            result[edge] = "add"
        elif present[0] != edge_type:
            result[edge] = "retype"
        elif abs(present[1] - strength) >= data["strength_update_threshold"]:
            result[edge] = "set_strength"
    for edge in current.keys() - desired.keys():
        result[edge] = "remove"
    return result


def _metrics(
    model: SharedWeightNeuronTask,
    data: dict[str, Any],
    *,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    records = []
    with torch.no_grad():
        for index in range(data["features"].shape[0]):
            sample = graph_proxy._subset(data, torch.tensor([index]))
            output = model(sample, hard=True, threshold=threshold)
            prediction = _GraphTaskAdapter().decode(output)
            topology = output["topology"]
            ids = sample["node_ids"][0].detach().cpu().tolist()
            actual: dict[tuple[int, int, int], float] = {}
            for edge_index, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist()):
                actual[(ids[source], ids[target], int(topology.edge_type[edge_index]))] = float(topology.strength[edge_index])
            expected = {
                (source_id, max(ids), edge_type): strength
                for source_id, edge_type, strength in sample["oracle_edges"][0]
            }
            tp_edges = actual.keys() & expected.keys()
            precision = len(tp_edges) / len(actual) if actual else None
            recall = len(tp_edges) / len(expected) if expected else None
            f1 = 1.0 if not actual and not expected else (
                2 * len(tp_edges) / (len(actual) + len(expected))
            )
            strength_errors = [abs(actual[edge] - expected[edge]) for edge in tp_edges]
            predicted_ops: dict[tuple[int, int], str] = {}
            for mutation in output["mutations"]:
                predicted_ops[(ids[mutation.source], ids[mutation.target])] = mutation.operation
            expected_ops = _ops_expected(sample, _topology(sample))
            op_metrics = {}
            for operation in ("add", "remove", "retype", "set_strength"):
                expected_items = {edge for edge, op in expected_ops.items() if op == operation}
                predicted_items = {edge for edge, op in predicted_ops.items() if op == operation}
                correct = len(expected_items & predicted_items)
                op_metrics[operation] = {
                    "expected": len(expected_items),
                    "predicted": len(predicted_items),
                    "correct": correct,
                }
            records.append({
                "mse": float(F.mse_loss(output["prediction"], sample["labels"]).item()),
                "mae": float(F.l1_loss(output["prediction"], sample["labels"]).item()),
                "prediction": float(prediction.reshape(-1)[0].item()),
                "label": float(sample["labels"].reshape(-1)[0].item()),
                "predicted_actions": [
                    {
                        "operation": mutation.operation,
                        "source": ids[mutation.source],
                        "target": ids[mutation.target],
                        "edge_type": mutation.edge_type,
                        "strength": mutation.strength,
                    }
                    for mutation in output["mutations"]
                ],
                "edge_f1": f1,
                "edge_precision": precision,
                "edge_recall": recall,
                "strength_errors": strength_errors,
                "operations": op_metrics,
                "edges": set(actual),
                "topology": topology,
            })
    operation_summary = {}
    for operation in ("add", "remove", "retype", "set_strength"):
        expected = sum(row["operations"][operation]["expected"] for row in records)
        predicted = sum(row["operations"][operation]["predicted"] for row in records)
        correct = sum(row["operations"][operation]["correct"] for row in records)
        operation_summary[operation] = {
            "expected_count": expected,
            "predicted_count": predicted,
            "correct_count": correct,
            "precision": correct / predicted if predicted else None,
            "recall": correct / expected if expected else None,
        }
    predicted_edge_count = sum(len(row["edges"]) for row in records)
    expected_edge_count = sum(
        len({(source_id, max(data["node_ids"][index].detach().cpu().tolist()), edge_type)
             for source_id, edge_type, _strength in data["oracle_edges"][index]})
        for index in range(len(records))
    )
    correct_edge_count = sum(
        len(row["edges"] & {
            (source_id, max(data["node_ids"][index].detach().cpu().tolist()), edge_type)
            for source_id, edge_type, _strength in data["oracle_edges"][index]
        })
        for index, row in enumerate(records)
    )
    return {
        "mse": statistics.mean(row["mse"] for row in records),
        "mae": statistics.mean(row["mae"] for row in records),
        "edge_f1": statistics.mean(row["edge_f1"] for row in records),
        "empty_edge_graph_count": sum(not row["edges"] for row in records),
        "evaluated_graph_count": len(records),
        "edge_precision": correct_edge_count / predicted_edge_count if predicted_edge_count else None,
        "edge_precision_numerator": correct_edge_count,
        "edge_precision_denominator": predicted_edge_count,
        "edge_recall": correct_edge_count / expected_edge_count if expected_edge_count else None,
        "edge_recall_numerator": correct_edge_count,
        "edge_recall_denominator": expected_edge_count,
        "strength_mae": (
            statistics.mean(error for row in records for error in row["strength_errors"])
            if any(row["strength_errors"] for row in records)
            else None
        ),
        "operation_scores": operation_summary,
        "_records": records,
    }


def _topology_edge_f1(data: dict[str, Any], topology: NeuronGraphTopology) -> float:
    ids = data["node_ids"][0].detach().cpu().tolist()
    actual = {
        (ids[source], ids[target], int(topology.edge_type[index]))
        for index, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist())
    }
    expected = {
        (source_id, max(ids), edge_type)
        for source_id, edge_type, _strength in data["oracle_edges"][0]
    }
    tp = len(actual & expected)
    if not actual and not expected:
        return 1.0
    return 2 * tp / (len(actual) + len(expected))


def _select_threshold(model: SharedWeightNeuronTask, validation: dict[str, Any], config: dict[str, Any], seed: int) -> tuple[float, list[dict[str, Any]]]:
    trials = []
    for threshold in config["validation_selection"]["thresholds"]:
        metrics = _metrics(model, validation, threshold=threshold, seed=seed)
        active = [
            row["recall"]
            for row in metrics["operation_scores"].values()
            if row["expected_count"] > 0
        ]
        action_recall = statistics.mean(active) if active else 0.0
        score = config["validation_selection"]["edge_f1_weight"] * metrics["edge_f1"] + config["validation_selection"]["operation_recall_weight"] * action_recall
        trials.append({"threshold": threshold, "selection_score": score, "edge_f1": metrics["edge_f1"], "operation_macro_recall": action_recall, "mse": metrics["mse"]})
    best = max(trials, key=lambda row: (row["selection_score"], -row["mse"], row["threshold"]))
    return float(best["threshold"]), trials


def _paired_ci(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
    half = crit.get(n - 1, 1.96) * sd / math.sqrt(n)
    return {"mean": mean, "sample_std": sd, "ci95": [mean - half, mean + half]}


def evaluate_saved_checkpoint(
    config: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    *,
    task: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one selected shared-neuron checkpoint without retraining."""
    if task not in config["task"]["tasks"]:
        raise ValueError(f"task is not in the experiment config: {task}")
    if seed not in config["training"]["seeds"]:
        raise ValueError(f"seed is not in the experiment config: {seed}")
    requested_device = config["training"]["device"]
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the config requires Apple MPS but MPS is unavailable")
    device = torch.device(requested_device)
    key_space = config["task"]["key_space"]
    feature_size = 2 * math.ceil(math.log2(key_space)) + 2
    registry = NeuronIdentityRegistry(config["task"]["identity_namespace"])
    identities = {key: registry.register(f"memory/{key}") for key in range(key_space)}
    readout_identity = registry.register("readout")
    model = SharedWeightNeuronTask(
        feature_size,
        config["model"]["hidden_size"],
        config["model"]["edge_types"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    metadata = {
        "task": task,
        "seed": seed,
        "model": "modelized_neuron_graph",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    restored = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        sample_generator=torch.Generator(device="cpu"),
        map_location=device,
        expected_metadata=metadata,
    )
    validation = _data(
        seed + 200,
        config["task"]["validation_examples"],
        memory_count=config["task"]["training_memory_neurons"],
        task=task,
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    )
    test = _data(
        seed + 300,
        config["task"]["test_examples"],
        memory_count=config["task"]["training_memory_neurons"],
        task=task,
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    )
    threshold, threshold_trials = _select_threshold(model, validation, config, seed)
    validation_metrics = _metrics(model, validation, threshold=threshold, seed=seed)
    test_metrics = _metrics(model, test, threshold=threshold, seed=seed + 1)
    return {
        "schema_version": "newera-checkpoint-evaluation-v1",
        "run_id": config["run_id"],
        "task": task,
        "seed": seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "training_step": restored["step"],
        "samples_seen": restored["samples_seen"],
        "validation_selected_threshold": threshold,
        "threshold_trials": threshold_trials,
        "validation_metrics": {
            key: value for key, value in validation_metrics.items() if not key.startswith("_")
        },
        "test_metrics": {
            key: value for key, value in test_metrics.items() if not key.startswith("_")
        },
        "test_predictions": [
            {"prediction": row["prediction"], "label": row["label"]}
            for row in test_metrics["_records"]
        ],
        "test_predicted_actions": [
            row["predicted_actions"] for row in test_metrics["_records"]
        ],
        "gradient_coverage": (restored.get("caller_state") or {}).get("gradient_coverage"),
    }


def _train(
    model: nn.Module,
    train_data: dict[str, Any],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    hard_action_fraction: float,
    sparsity_weight: float = 0.0,
    sample_interval_us: int = 2_000,
    checkpoint_path: Path | None = None,
    checkpoint_interval_steps: int = 0,
    checkpoint_metadata: dict[str, Any] | None = None,
    resume: bool = False,
    identity_registry: dict[str, Any] | None = None,
    overfit_probe_steps: int = 0,
    progress_interval_steps: int = 100,
) -> tuple[float, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    completed_steps = 0
    samples_seen = 0
    resumed_caller_state: dict[str, Any] = {}
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("resume requires an existing checkpoint_path")
        restored = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            sample_generator=generator,
            map_location=device,
            expected_metadata=checkpoint_metadata,
        )
        completed_steps = restored["step"]
        samples_seen = restored["samples_seen"]
        resumed_caller_state = restored.get("caller_state") or {}
    gradient_coverage: dict[str, Any] | None = resumed_caller_state.get("gradient_coverage")
    overfit_probe: dict[str, Any] | None = resumed_caller_state.get("single_sample_overfit_probe")
    if batch_size <= 0 or steps < completed_steps:
        raise ValueError("batch_size must be positive and steps cannot precede restored optimizer steps")
    if steps == completed_steps:
        return 0.0, {
            "optimizer_steps": completed_steps,
            "samples_seen": samples_seen,
            "requested_batch_size": batch_size,
            "effective_batch_size": batch_size,
            "effective_batch_mode": "per_sample_gradient_accumulation",
            "resumed_complete_checkpoint": True,
            "gradient_coverage": resumed_caller_state.get("gradient_coverage"),
            "single_sample_overfit_probe": overfit_probe,
            "checkpoint": str(checkpoint_path),
        }
    started = time.perf_counter()
    interval = sample_interval_us / 1_000_000
    checkpoint_every = checkpoint_interval_steps or steps
    progress_every = max(1, batch_size * progress_interval_steps)

    if overfit_probe is None and overfit_probe_steps > 0:
        probe_model = copy.deepcopy(model).to(device)
        probe_sample = graph_proxy._subset(
            train_data, torch.zeros(1, dtype=torch.long, device="cpu")
        )

        def probe_loss() -> Tensor:
            if isinstance(probe_model, SharedWeightNeuronTask):
                return F.mse_loss(
                    probe_model(probe_sample)["prediction"], probe_sample["labels"]
                )
            probe_features = graph_proxy._canonical_features(
                probe_sample["features"], probe_sample["node_ids"], train_data["key_space"]
            )
            return F.mse_loss(probe_model(probe_features), probe_sample["labels"])

        probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=lr)
        probe_model.train()
        with torch.no_grad():
            initial_probe_loss = float(probe_loss().item())
        for _ in range(overfit_probe_steps):
            probe_optimizer.zero_grad(set_to_none=True)
            current_loss = probe_loss()
            if not torch.isfinite(current_loss):
                break
            current_loss.backward()
            probe_optimizer.step()
        probe_model.eval()
        with torch.no_grad():
            final_probe_loss = float(probe_loss().item())
        overfit_probe = {
            "single_sample": True,
            "optimizer_steps": overfit_probe_steps,
            "initial_mse": initial_probe_loss,
            "final_mse": final_probe_loss,
            "loss_decreased": final_probe_loss < initial_probe_loss,
        }

    class SampleBatches:
        total_samples = (steps - completed_steps) * batch_size

        def __iter__(self):
            for index in sample_indices(
                train_data["features"].shape[0], self.total_samples, generator
            ):
                yield graph_proxy._subset(
                    train_data,
                    torch.tensor([index], dtype=torch.long, device="cpu"),
                )

    def loss_for(active_model: nn.Module, batch: dict[str, Any], optimizer_step: int) -> Tensor:
        if isinstance(active_model, SharedWeightNeuronTask):
            active_model.hard_action_active = optimizer_step >= int(steps * (1.0 - hard_action_fraction))
            capability: Capability[Any, Any] = _ModuleCapability(active_model)
            adapter: TaskAdapter[Any, Any, Any, Any] = _GraphTaskAdapter()
            output, prediction = execute_task(
                capability, adapter, batch, return_raw=True
            )
            loss = F.mse_loss(prediction, batch["labels"])
            if sparsity_weight:
                # This penalty is per example, matching the task loss reduction.
                loss = loss + sparsity_weight * output["candidate_weights"].abs().sum()
            return loss
        capability = _ModuleCapability(active_model)
        adapter = _FlatTaskAdapter(train_data["key_space"])
        prediction = execute_task(capability, adapter, batch)
        return F.mse_loss(prediction, batch["labels"])

    def checkpoint(step: int, consumed: int) -> None:
        if checkpoint_path is None or step % checkpoint_every != 0 and step != steps:
            return
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=step,
            samples_seen=samples_seen + consumed,
            sample_generator=generator,
            caller_state={
                "identity_registry": identity_registry,
                "node_state": None,
                "gradient_coverage": gradient_coverage,
                "single_sample_overfit_probe": overfit_probe,
            },
            metadata=checkpoint_metadata,
            rng_device=device,
        )

    def report(consumed: int, total: int, loss_value: float) -> None:
        if consumed % progress_every == 0 or consumed == total:
            print(
                f"train {checkpoint_metadata.get('task', 'control') if checkpoint_metadata else 'run'}: "
                f"samples {samples_seen + consumed}/{samples_seen + total}, loss={loss_value:.6g}",
                flush=True,
            )

    with graph_proxy._MemorySampler(device, interval) as memory:
        model.train()
        summary = train_with_accumulation(
            model,
            SampleBatches(),
            optimizer=optimizer,
            loss_for=loss_for,
            accumulation_steps=batch_size,
            start_step=completed_steps,
            progress=report,
            on_step=checkpoint,
        )
        gradient_coverage = summary.gradient_coverage
        checkpoint(steps, summary.samples_seen)
        if device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, {
        **memory.result(),
        "optimizer_steps": completed_steps + summary.optimizer_steps,
        "samples_seen": samples_seen + summary.samples_seen,
        "requested_batch_size": batch_size,
        "effective_batch_size": batch_size,
        "effective_batch_mode": "per_sample_gradient_accumulation",
        "sampling_seed": seed,
        "mean_training_loss": summary.mean_loss,
        "gradient_coverage": summary.gradient_coverage,
        "single_sample_overfit_probe": overfit_probe,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def _linear_macs(model: SharedWeightNeuronTask, data: dict[str, Any]) -> int:
    total = 0
    handles = []
    def hook(module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook))
    sample = graph_proxy._subset(data, torch.tensor([0]))
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    node_count = sample["features"].shape[1]
    hidden = model.neuron_graph.neuron.hidden_size
    # GRUCell uses fused input/recurrent matrix multiplies, not nn.Linear children.
    return total + 2 * node_count * 6 * hidden * hidden


def _permute_sample(data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    node_count = data["features"].shape[1]
    permutation = torch.tensor(rng.sample(range(node_count), node_count), device=data["features"].device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(node_count, device=permutation.device)
    edge_index = data["initial_edge_index"].clone()
    edge_index[0] = inverse[edge_index[0]]
    edge_index[1] = inverse[edge_index[1]]
    return {
        **data,
        "features": data["features"].index_select(1, permutation),
        "node_ids": data["node_ids"].index_select(1, permutation),
        "initial_edge_index": edge_index,
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _relative_error(reference: float, candidate: float, epsilon: float) -> float | None:
    if reference <= epsilon:
        return None
    return abs(reference - candidate) / reference


def _rollout(
    model: SharedWeightNeuronTask,
    *,
    task: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    identities: dict[int, int],
    readout_identity: int,
    threshold: float,
) -> dict[str, Any]:
    sequence_count = config["evolution"]["sequences_per_seed_task"]
    sequence_length = config["evolution"]["steps_per_sequence"]
    pool_size = config["task"]["training_memory_neurons"]
    rows = []
    churn = []
    for sequence in range(sequence_count):
        pool_rng = random.Random(config["evolution"]["seed_base"] + seed * 100_000 + sequence)
        fixed_pool = sorted(pool_rng.sample(range(config["task"]["key_space"]), pool_size))
        topology = None
        state = None
        previous: set[tuple[int, int, int, float]] = set()
        for step in range(sequence_length):
            frame = _data(
                config["evolution"]["seed_base"] + seed * 100_000 + sequence * 1000 + step,
                1,
                memory_count=pool_size,
                task=task,
                config=config,
                device=device,
                identities=identities,
                readout_identity=readout_identity,
                memory_id_pool=fixed_pool,
            )
            frame = _permute_sample(frame, seed + sequence * 1000 + step)
            if topology is None:
                topology = _topology(frame)
            else:
                topology = topology.remap_node_ids(frame["node_ids"], causal=False)
                state = model.neuron_graph.remap_state(state, frame["node_ids"])
            frame["initial_edge_index"] = topology.edge_index
            frame["initial_edge_type"] = topology.edge_type
            frame["initial_strength"] = topology.strength
            frame["oracle_edges"] = frame["oracle_edges"]
            frame.pop("expected_operations", None)
            frame["expected_operations"] = [_ops_expected(frame, topology)]
            output = model(frame, hard=True, threshold=threshold, state=state)
            topology = output["topology"]
            state = output["state"]
            ids = frame["node_ids"][0].detach().cpu().tolist()
            current = {
                (ids[source], ids[target], int(topology.edge_type[i]), float(topology.strength[i]))
                for i, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist())
            }
            churn.append(len(current.symmetric_difference(previous)))
            previous = current
            rows.append({
                "mse": float(F.mse_loss(output["prediction"], frame["labels"]).item()),
                "edge_count": topology.edge_count,
                "edge_f1": _topology_edge_f1(frame, topology),
            })
    return {
        "sequence_count": sequence_count,
        "steps_per_sequence": sequence_length,
        "total_transitions": len(rows),
        "mean_step_mse": statistics.mean(row["mse"] for row in rows),
        "mean_edge_f1": statistics.mean(row["edge_f1"] for row in rows),
        "mean_hard_edge_count": statistics.mean(row["edge_count"] for row in rows),
        "mean_identity_edge_state_churn": statistics.mean(churn),
    }


def run(
    config: dict[str, Any],
    config_path: Path,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if config["training"]["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the config requires Apple MPS but MPS is unavailable")
    device = torch.device(config["training"]["device"])
    torch.set_num_threads(config["training"]["threads"])
    key_space = config["task"]["key_space"]
    bit_width = math.ceil(math.log2(key_space))
    feature_size = 2 * bit_width + 2
    registry = NeuronIdentityRegistry(config["task"]["identity_namespace"])
    identities = {key: registry.register(f"memory/{key}") for key in range(key_space)}
    readout_identity = registry.register("readout")
    hidden_size = config["model"]["hidden_size"]
    provisional = SharedWeightNeuronTask(feature_size, hidden_size, config["model"]["edge_types"]).to(device)
    train_probe = _data(
        config["training"]["seeds"][0] + 100,
        1,
        memory_count=config["task"]["training_memory_neurons"],
        task=config["task"]["tasks"][0],
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    )
    model_parameters = sum(parameter.numel() for parameter in provisional.parameters())
    mac_count = _linear_macs(provisional, train_probe)
    input_size = (key_space + 1) * feature_size
    param_width = graph_proxy._closest_width(model_parameters, input_size, parameters=True)
    mac_width = graph_proxy._closest_width(mac_count, input_size, parameters=False)
    cases = []
    train_times = {"modelized_neuron_graph": [], "parameter_matched_mlp": [], "linear_mac_matched_mlp": []}
    for task in config["task"]["tasks"]:
        for seed in config["training"]["seeds"]:
            print(f"prepare task={task} seed={seed}: train/validation/test splits", flush=True)
            train = _data(seed + 100, config["task"]["training_examples"], memory_count=config["task"]["training_memory_neurons"], task=task, config=config, device=device, identities=identities, readout_identity=readout_identity)
            validation = _data(seed + 200, config["task"]["validation_examples"], memory_count=config["task"]["training_memory_neurons"], task=task, config=config, device=device, identities=identities, readout_identity=readout_identity)
            test = _data(seed + 300, config["task"]["test_examples"], memory_count=config["task"]["training_memory_neurons"], task=task, config=config, device=device, identities=identities, readout_identity=readout_identity)
            torch.manual_seed(seed)
            models: dict[str, nn.Module] = {
                "modelized_neuron_graph": SharedWeightNeuronTask(feature_size, hidden_size, config["model"]["edge_types"]).to(device),
                "parameter_matched_mlp": FlatControl(input_size, param_width).to(device),
                "linear_mac_matched_mlp": FlatControl(input_size, mac_width).to(device),
            }
            result: dict[str, Any] = {"task": task, "seed": seed, "models": {}}
            train_mean = train["labels"].mean()
            zero_prediction = torch.zeros_like(test["labels"])
            mean_prediction = train_mean.expand_as(test["labels"])
            oracle_errors = {}
            for split_name, split in (("train", train), ("validation", validation), ("test", test)):
                oracle = _oracle_targets(split, key_space)
                oracle_errors[split_name] = float((oracle - split["labels"]).abs().max().item())
            max_oracle_error = max(oracle_errors.values())
            if max_oracle_error > config["decision_rules"]["maximum_oracle_label_error"]:
                raise ValueError(
                    f"analytic oracle does not reproduce task labels: max error={max_oracle_error}"
                )
            oracle_prediction = _oracle_targets(test, key_space)
            oracle_max_error = float((oracle_prediction - test["labels"]).abs().max().item())
            result["models"]["zero_predictor"] = {
                "test": {
                    "mse": float(F.mse_loss(zero_prediction, test["labels"]).item()),
                    "mae": float(F.l1_loss(zero_prediction, test["labels"]).item()),
                },
                "kind": "constant_control",
            }
            result["models"]["train_mean_predictor"] = {
                "test": {
                    "mse": float(F.mse_loss(mean_prediction, test["labels"]).item()),
                    "mae": float(F.l1_loss(mean_prediction, test["labels"]).item()),
                },
                "prediction": float(train_mean.item()),
                "kind": "constant_control_fit_on_train_split",
            }
            result["models"]["generator_oracle"] = {
                "test": {
                    "mse": float(F.mse_loss(oracle_prediction, test["labels"]).item()),
                    "mae": float(F.l1_loss(oracle_prediction, test["labels"]).item()),
                    "max_absolute_label_error": oracle_max_error,
                    "max_absolute_label_error_by_split": oracle_errors,
                },
                "kind": "identity_edge_reference_solution_not_an_inference_adapter",
            }
            for name, model in models.items():
                print(f"train start task={task} seed={seed} model={name}", flush=True)
                checkpoint_path = (
                    checkpoint_dir / task / f"seed_{seed}" / f"{name}.pt"
                    if checkpoint_dir is not None
                    else None
                )
                checkpoint_metadata = {
                    "task": task,
                    "seed": seed,
                    "model": name,
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                }
                elapsed, memory = _train(
                    model,
                    train,
                    steps=config["training"]["steps"],
                    batch_size=config["training"]["batch_size"],
                    lr=config["training"]["learning_rate"],
                    seed=seed,
                    device=device,
                    hard_action_fraction=config["training"]["hard_action_training_fraction"],
                    sparsity_weight=config["training"]["edge_sparsity_weight"] if name == "modelized_neuron_graph" else 0.0,
                    sample_interval_us=config["cost_accounting"]["memory_sampling_interval_microseconds"],
                    checkpoint_path=checkpoint_path,
                    checkpoint_interval_steps=config["training"].get("checkpoint_interval_steps", 100),
                    checkpoint_metadata=checkpoint_metadata,
                    resume=resume,
                    identity_registry=registry.state_dict(),
                    overfit_probe_steps=config.get("diagnostics", {}).get("single_sample_overfit_steps", 0),
                    progress_interval_steps=config["training"].get("progress_interval_steps", 100),
                )
                train_times[name].append(elapsed)
                if name == "modelized_neuron_graph":
                    print(f"validation start task={task} seed={seed}: select threshold", flush=True)
                    threshold, trials = _select_threshold(model, validation, config, seed)
                    valid_metrics = _metrics(model, validation, threshold=threshold, seed=seed)
                    test_metrics = _metrics(model, test, threshold=threshold, seed=seed + 1)
                    print(f"test complete task={task} seed={seed}: frozen threshold", flush=True)
                    pool_generalization = {}
                    for pool in config["task"]["evaluation_memory_neurons"]:
                        pool_data = _data(
                            seed + 10_000 + pool,
                            min(config["task"]["test_examples"], 32),
                            memory_count=pool,
                            task=task,
                            config=config,
                            device=device,
                            identities=identities,
                            readout_identity=readout_identity,
                        )
                        pool_metrics = _metrics(model, pool_data, threshold=threshold, seed=seed)
                        pool_generalization[str(pool)] = {
                            key: value for key, value in pool_metrics.items() if not key.startswith("_")
                        }
                    permuted = [_permute_sample(graph_proxy._subset(test, torch.tensor([i])), seed + 10_000 + i) for i in range(test["features"].shape[0])]
                    paired_jaccard = []
                    empty_graph_pairs = []
                    empty_graph_count = 0
                    paired_drift = []
                    with torch.no_grad():
                        for i, sample in enumerate(permuted):
                            original = graph_proxy._subset(test, torch.tensor([i]))
                            out_a = model(original, hard=True, threshold=threshold)
                            out_b = model(sample, hard=True, threshold=threshold)
                            ids_a = original["node_ids"][0].detach().cpu().tolist()
                            ids_b = sample["node_ids"][0].detach().cpu().tolist()
                            edges_a = {(ids_a[s], ids_a[t], int(out_a["topology"].edge_type[j])) for j, (s, t) in enumerate(out_a["topology"].edge_index.t().detach().cpu().tolist())}
                            edges_b = {(ids_b[s], ids_b[t], int(out_b["topology"].edge_type[j])) for j, (s, t) in enumerate(out_b["topology"].edge_index.t().detach().cpu().tolist())}
                            paired_jaccard.append(_jaccard(edges_a, edges_b))
                            empty_graph_pairs.append(not edges_a and not edges_b)
                            empty_graph_count += int(not edges_a) + int(not edges_b)
                            reference_mse = float(F.mse_loss(out_a["prediction"], original["labels"]).item())
                            candidate_mse = float(F.mse_loss(out_b["prediction"], sample["labels"]).item())
                            epsilon = config["decision_rules"].get("relative_error_denominator_epsilon", 1e-12)
                            paired_drift.append(_relative_error(reference_mse, candidate_mse, epsilon))
                    result["models"][name] = {
                        "validation": {key: value for key, value in valid_metrics.items() if not key.startswith("_")},
                        "test": {key: value for key, value in test_metrics.items() if not key.startswith("_")},
                        "pool_generalization": pool_generalization,
                        "threshold": threshold,
                        "threshold_trials": trials,
                        "permutation_identity_edge_jaccard": statistics.mean(paired_jaccard),
                        "permutation_both_empty_graph_rate": statistics.mean(empty_graph_pairs),
                        "permutation_empty_graph_fraction": empty_graph_count / max(1, 2 * len(empty_graph_pairs)),
                        "permutation_relative_mse_drift": (
                            statistics.mean(value for value in paired_drift if value is not None)
                            if any(value is not None for value in paired_drift)
                            else None
                        ),
                        "permutation_relative_mse_drift_defined_pairs": sum(
                            value is not None for value in paired_drift
                        ),
                        "evolution": _rollout(model, task=task, seed=seed, config=config, device=device, identities=identities, readout_identity=readout_identity, threshold=threshold),
                        "train_seconds": elapsed,
                        "training_memory": memory,
                        "single_sample_overfit_probe": memory.get("single_sample_overfit_probe"),
                        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    }
                else:
                    with torch.no_grad():
                        features = graph_proxy._canonical_features(test["features"], test["node_ids"], key_space)
                        predictions = model(features)
                        mse = float(F.mse_loss(predictions, test["labels"]).item())
                        mae = float(F.l1_loss(predictions, test["labels"]).item())
                    result["models"][name] = {"test": {"mse": mse, "mae": mae}, "train_seconds": elapsed, "training_memory": memory, "single_sample_overfit_probe": memory.get("single_sample_overfit_probe"), "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
            cases.append(result)
    summaries = {}
    for task in config["task"]["tasks"]:
        rows = [row for row in cases if row["task"] == task]
        graph_mse = [row["models"]["modelized_neuron_graph"]["test"]["mse"] for row in rows]
        task_summary = {"seed_count": len(rows)}
        for baseline in (
            "zero_predictor",
            "train_mean_predictor",
            "parameter_matched_mlp",
            "linear_mac_matched_mlp",
        ):
            deltas = [
                (row["models"]["modelized_neuron_graph"]["test"]["mse"] - row["models"][baseline]["test"]["mse"])
                / max(row["models"][baseline]["test"]["mse"], 1e-12)
                for row in rows
            ]
            ci = _paired_ci(deltas)
            ci["criterion_met"] = ci["mean"] <= -config["decision_rules"]["minimum_relative_mse_improvement"] and ci["ci95"][1] < 0
            task_summary[baseline] = ci
            task_summary["edge_f1_mean"] = statistics.mean(row["models"]["modelized_neuron_graph"]["test"]["edge_f1"] for row in rows)
        task_summary["permutation_edge_jaccard_mean"] = statistics.mean(row["models"]["modelized_neuron_graph"]["permutation_identity_edge_jaccard"] for row in rows)
        task_summary["permutation_both_empty_graph_rate"] = statistics.mean(
            row["models"]["modelized_neuron_graph"]["permutation_both_empty_graph_rate"] for row in rows
        )
        task_summary["permutation_empty_graph_fraction"] = statistics.mean(
            row["models"]["modelized_neuron_graph"]["permutation_empty_graph_fraction"] for row in rows
        )
        defined_drifts = [
            row["models"]["modelized_neuron_graph"]["permutation_relative_mse_drift"]
            for row in rows
            if row["models"]["modelized_neuron_graph"]["permutation_relative_mse_drift"] is not None
        ]
        task_summary["permutation_mse_drift_mean"] = statistics.mean(defined_drifts) if defined_drifts else None
        task_summary["permutation_mse_drift_defined_seeds"] = len(defined_drifts)
        operation_names = ("add", "remove", "retype", "set_strength")
        operation_recall = {}
        operation_expected_counts = {}
        for operation in operation_names:
            expected = sum(
                row["models"]["modelized_neuron_graph"]["test"]["operation_scores"][operation]["expected_count"]
                for row in rows
            )
            correct = sum(
                row["models"]["modelized_neuron_graph"]["test"]["operation_scores"][operation]["correct_count"]
                for row in rows
            )
            operation_expected_counts[operation] = expected
            operation_recall[operation] = correct / expected if expected else None
        task_summary["operation_recall"] = operation_recall
        task_summary["operation_expected_counts"] = operation_expected_counts
        strength_values = [
            row["models"]["modelized_neuron_graph"]["test"]["strength_mae"]
            for row in rows
            if row["models"]["modelized_neuron_graph"]["test"]["strength_mae"] is not None
        ]
        task_summary["strength_mae_mean"] = statistics.mean(strength_values) if strength_values else None
        task_summary["structure_criterion_met"] = (
            task_summary["edge_f1_mean"] >= config["decision_rules"]["minimum_edge_f1"]
            and task_summary["permutation_edge_jaccard_mean"] >= config["decision_rules"]["minimum_permutation_identity_edge_jaccard"]
            and task_summary["permutation_empty_graph_fraction"] <= config["decision_rules"]["maximum_permutation_empty_graph_fraction"]
            and task_summary["permutation_mse_drift_mean"] is not None
            and task_summary["permutation_mse_drift_mean"] <= config["decision_rules"]["maximum_permutation_relative_mse_drift"]
            and all(
                recall is None or recall >= config["decision_rules"]["minimum_operation_recall"]
                for recall in operation_recall.values()
            )
            and task_summary["strength_mae_mean"] is not None
            and task_summary["strength_mae_mean"] <= config["decision_rules"]["maximum_strength_mae"]
        )
        task_summary["mean_test_mse"] = statistics.mean(graph_mse)
        pool_summary = {}
        for pool in config["task"]["evaluation_memory_neurons"]:
            pool_rows = [
                row["models"]["modelized_neuron_graph"]["pool_generalization"][str(pool)]
                for row in rows
            ]
            pool_summary[str(pool)] = {
                "mse_mean": statistics.mean(item["mse"] for item in pool_rows),
                "edge_f1_mean": statistics.mean(item["edge_f1"] for item in pool_rows),
                "strength_mae_mean": statistics.mean(
                    item["strength_mae"] for item in pool_rows if item["strength_mae"] is not None
                ) if any(item["strength_mae"] is not None for item in pool_rows) else None,
            }
        task_summary["pool_generalization"] = pool_summary
        summaries[task] = task_summary

    # Use the last trained model only for throughput; no weights are selected from scale outcomes.
    scale_rows = []
    for pool in config["task"]["evaluation_memory_neurons"]:
        data = _data(8_000_000 + pool, 1, memory_count=pool, task="two_link_composition", config=config, device=device, identities=identities, readout_identity=readout_identity)
        model = models["modelized_neuron_graph"].eval()
        times = []
        with torch.no_grad():
            for _ in range(config["cost_accounting"]["scale_warmup_calls"]):
                model(data)
            if device.type == "mps": torch.mps.synchronize()
            for _ in range(config["cost_accounting"]["scale_measured_calls"]):
                started = time.perf_counter(); model(data)
                if device.type == "mps": torch.mps.synchronize()
                times.append((time.perf_counter() - started) * 1000)
        scale_rows.append({"memory_neurons": pool, "parameter_count": sum(p.numel() for p in model.parameters()), "median_forward_ms": statistics.median(times), "p95_forward_ms": max(times)})

    state = registry.state_dict()
    return {
        "schema_version": "newera-modelized-neuron-graph-v1",
        "run_id": config["run_id"],
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in config["accounting"]["source_files"]},
        "task_contract": config["task"],
        "model_contract": {
            "architecture": provisional.neuron_graph.neuron.architecture_signature,
            "neuron_weight_sets": 1,
            "parameter_count": model_parameters,
            "linear_and_gru_macs_per_graph_step": mac_count,
            "multiply_add_flops_subtotal": 2 * mac_count,
            "total_flops_exactly_counted": False,
            "profiler_supported_activities": [str(item) for item in torch.profiler.supported_activities()],
            "identity_registry": state,
        },
        "training_budget": {
            "optimizer_steps_per_model": config["training"]["steps"],
            "requested_batch_size": config["training"]["batch_size"],
            "effective_batch_size": config["training"]["batch_size"],
            "batch_implementation": "per_sample_gradient_accumulation; ModelNeuronGraph currently processes one graph per forward",
            "samples_per_model": config["training"]["steps"] * config["training"]["batch_size"],
            "samples_per_task_seed_across_three_trained_models": 3 * config["training"]["steps"] * config["training"]["batch_size"],
        },
        "decision_rules": config["decision_rules"],
        "training_time_mean_seconds": {name: statistics.mean(values) for name, values in train_times.items()},
        "training_cases": cases,
        "summary": summaries,
        "scaling": scale_rows,
        "limitations": config["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints beside --output")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="evaluate one saved modelized-neuron checkpoint without training")
    parser.add_argument("--task", choices=("single_link_retrieval", "two_link_composition"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.evaluate_checkpoint is not None:
        if args.task is None or args.seed is None:
            parser.error("--evaluate-checkpoint requires --task and --seed")
        result = evaluate_saved_checkpoint(
            config,
            args.config,
            args.evaluate_checkpoint,
            task=args.task,
            seed=args.seed,
        )
        write_json_atomic(args.output, result)
        print(json.dumps({"task": result["task"], "seed": result["seed"], "checkpoint": str(args.evaluate_checkpoint), "output": str(args.output)}, ensure_ascii=False, indent=2))
        return
    checkpoint_dir = args.output.with_name(f"{args.output.stem}_checkpoints")
    result = run(config, args.config, checkpoint_dir=checkpoint_dir, resume=args.resume)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = result["model_contract"].pop("identity_registry")
    registry = NeuronIdentityRegistry.from_state_dict(state)
    registry_path = args.output.with_name("identity_registry.json")
    registry.save(registry_path)
    restored = NeuronIdentityRegistry.load(registry_path)
    next_id = restored.register("post-restart/model-neuron")
    restored.save(registry_path)
    result["model_contract"]["identity_registry_persistence"] = {
        "disk_round_trip_ok": NeuronIdentityRegistry.load(registry_path).resolve("post-restart/model-neuron") == next_id,
        "count_after_restart": len(restored.state_dict()["entries"]),
        "path": registry_path.name,
        "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    write_json_atomic(args.output, result)
    print(json.dumps({"run_id": result["run_id"], "device": result["device"], "cases": len(result["training_cases"]), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
