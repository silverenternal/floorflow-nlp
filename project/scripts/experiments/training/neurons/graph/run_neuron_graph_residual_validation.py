"""Evaluate permutation-invariant neuron routing and task-trained topology evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Self

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from newera.neuron.graph import (
    DynamicNeuronGraphLayer,
    LinkMutation,
    LinkProposalPolicy,
    NeuronGraphTopology,
    NeuronIdentityRegistry,
    TopologyController,
)

ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "project/configs/models_training/learning/neuron_comparisons/topology_evolution/topology_evolution_identity.json"


class _MemorySampler:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval = interval_seconds
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self.start_device = _device_memory(device)
        self.peak_device = dict(self.start_device)
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            for key, value in _device_memory(self.device).items():
                self.peak_device[key] = max(self.peak_device.get(key, 0), value)
            self.stop.wait(self.interval)

    def __enter__(self) -> Self:
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        for key, value in _device_memory(self.device).items():
            self.peak_device[key] = max(self.peak_device.get(key, 0), value)

    def result(self) -> dict[str, Any]:
        return {
            "host_rss_start_bytes": self.start_rss,
            "host_rss_sampled_peak_delta_bytes": max(0, self.peak_rss - self.start_rss),
            "device_start_bytes": self.start_device,
            "device_sampled_peak_delta_bytes": {
                key: value - self.start_device.get(key, 0) for key, value in self.peak_device.items()
            },
            "sample_interval_us": int(self.interval * 1_000_000),
        }


def _device_memory(device: torch.device) -> dict[str, int]:
    if device.type == "mps":
        return {
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
    if device.type == "cuda":
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
        }
    return {}


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class IdentityGraphModel(nn.Module):
    def __init__(
        self, input_features: int, hidden_size: int, key_space: int, readout_identity: int
    ) -> None:
        super().__init__()
        self.key_space = key_space
        self.readout_identity = readout_identity
        self.hard_action_active = False
        self.max_nodes = key_space + 1
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_size), nn.Tanh())
        self.policy = LinkProposalPolicy(
            hidden_size, 2, initial_keep_bias=0.0, predict_strengths=True
        )
        self.graph = DynamicNeuronGraphLayer(hidden_size, 2, causal=False)
        with torch.no_grad():
            self.graph.residual_gate.fill_(1.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(
        self,
        features: torch.Tensor,
        node_ids: torch.Tensor,
        initial_edge_index: torch.Tensor,
        initial_edge_type: torch.Tensor,
        initial_strength: torch.Tensor,
        *,
        hard: bool = False,
        threshold: float = 0.5,
        strength_update_threshold: float = 0.05,
    ) -> dict[str, Any]:
        batch, node_count, _ = features.shape
        if node_count != self.max_nodes or node_ids.shape != (batch, node_count):
            raise ValueError("features/node_ids must match the fixed padded identity pool")
        topology = NeuronGraphTopology(
            batch,
            node_count,
            initial_edge_index,
            initial_edge_type,
            initial_strength,
            node_ids,
        )
        readout = features[:, :, -1].gt(0.5)
        if not torch.all(readout.sum(dim=1) == 1):
            raise ValueError("every graph must contain exactly one readout neuron")
        memory = node_ids.ge(0) & ~readout
        memory_counts = memory.sum(dim=1)
        if not torch.all(memory_counts == memory_counts[0]):
            raise ValueError("batch graphs must use one pool size per forward pass")
        source_slots = memory.nonzero(as_tuple=False)[:, 1].reshape(batch, -1)
        target_slots = readout.nonzero(as_tuple=False)[:, 1].reshape(batch, 1)
        offsets = torch.arange(batch, device=features.device)[:, None] * node_count
        candidate_edges = torch.stack(
            (
                (source_slots + offsets).reshape(-1),
                (target_slots.expand_as(source_slots) + offsets).reshape(-1),
            )
        )
        encoded = self.encoder(features)
        logits = self.policy(encoded, candidate_edges, topology, causal=False)
        strengths = self.policy.predict_strengths(encoded, candidate_edges, topology, causal=False)
        mutations: list[LinkMutation] = []
        if hard:
            mutations = self.policy.propose(
                logits,
                candidate_edges,
                topology,
                threshold=threshold,
                candidate_strengths=strengths,
                strength_update_threshold=strength_update_threshold,
            )
            topology = TopologyController(
                max_edges=batch * node_count, causal=False
            ).apply(topology, mutations, num_edge_types=2)
            graph_result = self.graph(encoded, topology)
        else:
            weights = self.policy.expected_type_weights(
                logits,
                candidate_edges,
                topology,
                candidate_strengths=strengths,
                straight_through=self.training and self.hard_action_active,
            )
            graph_result = self.graph(
                encoded,
                topology,
                candidate_edges=candidate_edges,
                candidate_type_weights=weights,
            )
        readout_slots = readout.nonzero(as_tuple=False)[:, 1]
        rows = torch.arange(batch, device=features.device)
        prediction = self.head(graph_result.hidden_states[rows, readout_slots]).squeeze(-1)
        return {
            "prediction": prediction,
            "logits": logits,
            "strengths": strengths,
            "topology": topology,
            "mutations": mutations,
        }



def _canonical_features(features: torch.Tensor, node_ids: torch.Tensor, key_space: int) -> torch.Tensor:
    """Arrange padded neurons by stable ID so flat controls share the same slot invariance."""
    canonical = features.new_zeros(features.shape[0], key_space + 1, features.shape[-1])
    valid = node_ids.ge(0)
    bit_width = math.ceil(math.log2(key_space))
    powers = (2 ** torch.arange(bit_width, device=features.device, dtype=torch.long))
    keys = (features[..., :bit_width].round().long() * powers).sum(dim=-1)
    destinations = torch.where(features[..., -1] > 0.5, key_space, keys).clamp(0, key_space)
    canonical.scatter_add_(1, destinations.unsqueeze(-1).expand_as(features), features * valid.unsqueeze(-1))
    return canonical


class FlatControl(nn.Module):
    def __init__(self, input_features: int, node_count: int, hidden_size: int, key_space: int) -> None:
        super().__init__()
        self.key_space = key_space
        self.network = nn.Sequential(
            nn.Linear(input_features * node_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.flatten(start_dim=1)).squeeze(-1)


def _bits(value: int, width: int) -> list[float]:
    return [float((value >> index) & 1) for index in range(width)]


def _data(
    seed: int,
    count: int,
    *,
    memory_count: int,
    task: str,
    key_space: int,
    device: torch.device,
    identity_ids: dict[int, int],
    readout_identity: int,
) -> dict[str, Any]:
    if memory_count < 3 or memory_count > key_space:
        raise ValueError("memory_count must be between 3 and key_space")
    rng = random.Random(seed)
    bit_width = math.ceil(math.log2(key_space))
    feature_size = 2 * bit_width + 2
    max_nodes = key_space + 1
    features = torch.zeros(count, max_nodes, feature_size)
    node_ids = torch.full((count, max_nodes), -1, dtype=torch.long)
    labels = torch.zeros(count)
    initial_sources: list[int] = []
    initial_targets: list[int] = []
    initial_types: list[int] = []
    initial_strengths: list[float] = []
    oracle_edges: list[list[tuple[int, int, float]]] = []
    expected_operations: list[dict[tuple[int, int], str]] = []

    for sample in range(count):
        query = rng.randrange(key_space)
        decoy = (query + 2) % key_space
        successor = (query + 1) % key_space
        required = [query, decoy]
        if task == "two_link_composition":
            required.append(successor)
        keys = list(required)
        candidates = [key for key in range(key_space) if key not in keys]
        rng.shuffle(candidates)
        keys.extend(candidates[: memory_count - len(keys)])
        values = {key: rng.uniform(-1.0, 1.0) for key in keys}
        q_type = rng.randrange(2)
        decoy_type = rng.randrange(2)
        q_strength = 0.95
        decoy_strength = 0.85
        oracle = [(query, 0, 0.60)]
        label = 0.60 * values[query]
        if task == "two_link_composition":
            oracle.append((successor, 1, 0.35))
            label += 0.35 * values[successor]
        elif task != "single_link_retrieval":
            raise ValueError(f"unknown task: {task}")
        readout_id = key_space
        rows: list[tuple[int, list[float], int]] = []
        query_bits = _bits(query, bit_width)
        for key in keys:
            row = _bits(key, bit_width) + [values[key]] + [0.0] * bit_width + [0.0]
            rows.append((key, row, key))
        readout = [0.0] * bit_width + [0.0] + query_bits + [1.0]
        rows.append((readout_id, readout, readout_id))
        valid_slots = rng.sample(range(max_nodes), memory_count + 1)
        rng.shuffle(valid_slots)
        slot_for_identity: dict[int, int] = {}
        for (identity, row, _), slot in zip(rows, valid_slots, strict=True):
            features[sample, slot] = torch.tensor(row)
            node_ids[sample, slot] = readout_identity if identity == readout_id else identity_ids[identity]
            slot_for_identity[identity] = slot
        target_slot = slot_for_identity[readout_id]
        for source_identity, edge_type, strength in ((query, q_type, q_strength), (decoy, decoy_type, decoy_strength)):
            initial_sources.append(sample * max_nodes + slot_for_identity[source_identity])
            initial_targets.append(sample * max_nodes + target_slot)
            initial_types.append(edge_type)
            initial_strengths.append(strength)
        true_ops: dict[tuple[int, int], str] = {}
        true_ops[(query, readout_id)] = "set_strength" if q_type == 0 else "retype"
        true_ops[(decoy, readout_id)] = "remove"
        if task == "two_link_composition":
            true_ops[(successor, readout_id)] = "add"
        labels[sample] = label
        oracle_edges.append([(identity_ids[source], edge_type, strength) for source, edge_type, strength in oracle])
        expected_operations.append({(identity_ids[source], readout_identity): op for (source, _target), op in true_ops.items()})

    return {
        "features": features.to(device),
        "node_ids": node_ids.to(device),
        "labels": labels.to(device),
        "initial_edge_index": torch.tensor([initial_sources, initial_targets], dtype=torch.long, device=device),
        "initial_edge_type": torch.tensor(initial_types, dtype=torch.long, device=device),
        "initial_strength": torch.tensor(initial_strengths, dtype=torch.float32, device=device),
        "oracle_edges": oracle_edges,
        "expected_operations": expected_operations,
        "task": task,
        "memory_count": memory_count,
    }


def _subset(data: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    selected = indices.detach().cpu().tolist()
    node_count = data["features"].shape[1]
    edge_batches = data["initial_edge_index"][0] // node_count
    edge_mask = torch.zeros(data["features"].shape[0], dtype=torch.bool, device=edge_batches.device)
    edge_mask[selected] = True
    edge_ids = edge_mask[edge_batches]
    edge_index = data["initial_edge_index"][:, edge_ids].clone()
    edge_types = data["initial_edge_type"][edge_ids]
    edge_strengths = data["initial_strength"][edge_ids]
    old_to_new = {old: new for new, old in enumerate(selected)}
    for edge_col in range(edge_index.shape[1]):
        old_batch = int(edge_index[0, edge_col] // node_count)
        edge_index[:, edge_col] -= old_batch * node_count
        edge_index[:, edge_col] += old_to_new[old_batch] * node_count
    return {
        "features": data["features"].index_select(0, indices),
        "node_ids": data["node_ids"].index_select(0, indices),
        "labels": data["labels"].index_select(0, indices),
        "initial_edge_index": edge_index,
        "initial_edge_type": edge_types,
        "initial_strength": edge_strengths,
        "oracle_edges": [data["oracle_edges"][index] for index in selected],
        "expected_operations": [data["expected_operations"][index] for index in selected],
        "task": data["task"],
        "memory_count": data["memory_count"],
    }


def _permute_data(data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    features = data["features"].clone()
    node_ids = data["node_ids"].clone()
    node_count = features.shape[1]
    edge_index = data["initial_edge_index"].clone()
    edge_batches = edge_index[0] // node_count
    for batch_index in range(features.shape[0]):
        permutation = torch.tensor(rng.sample(range(node_count), node_count), device=features.device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(node_count, device=features.device)
        features[batch_index] = data["features"][batch_index].index_select(0, permutation)
        node_ids[batch_index] = data["node_ids"][batch_index].index_select(0, permutation)
        cols = edge_batches == batch_index
        edge_index[0, cols] = batch_index * node_count + inverse[edge_index[0, cols] % node_count]
        edge_index[1, cols] = batch_index * node_count + inverse[edge_index[1, cols] % node_count]
    return {
        **data,
        "features": features,
        "node_ids": node_ids,
        "initial_edge_index": edge_index,
    }


def _linear_macs(module: nn.Module, data: dict[str, Any]) -> int:
    total = 0
    handles = []
    def count(layer: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features
    for child in module.modules():
        if isinstance(child, nn.Linear):
            handles.append(child.register_forward_hook(count))
    one = _subset(data, torch.tensor([0], device=data["features"].device))
    with torch.no_grad():
        if isinstance(module, IdentityGraphModel):
            module(one["features"], one["node_ids"], one["initial_edge_index"], one["initial_edge_type"], one["initial_strength"])
        else:
            module(_canonical_features(one["features"], one["node_ids"], module.key_space))
    for handle in handles:
        handle.remove()
    return total


def _closest_width(target: int, input_size: int, *, parameters: bool) -> int:
    def cost(width: int) -> int:
        return width * (input_size + 2) + 1 if parameters else width * (input_size + 1)
    upper = max(3, (target // max(input_size, 1)) * 2 + 2)
    return min(range(1, upper), key=lambda width: abs(cost(width) - target))


def _metrics(model: IdentityGraphModel | FlatControl, data: dict[str, Any], *, threshold: float, seed: int) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        if isinstance(model, IdentityGraphModel):
            soft = model(data["features"], data["node_ids"], data["initial_edge_index"], data["initial_edge_type"], data["initial_strength"])
            hard = model(data["features"], data["node_ids"], data["initial_edge_index"], data["initial_edge_type"], data["initial_strength"], hard=True, threshold=threshold)
            prediction = hard["prediction"]
            soft_mse = F.mse_loss(soft["prediction"], data["labels"]).item()
            actual_sets: list[set[tuple[int, int]]] = [set() for _ in range(data["features"].shape[0])]
            actual_strengths: list[dict[tuple[int, int], float]] = [{} for _ in actual_sets]
            topology = hard["topology"]
            n = topology.nodes_per_graph
            for edge_id, (source, _target) in enumerate(topology.edge_index.t().tolist()):
                batch_index = source // n
                source_identity = int(data["node_ids"][batch_index, source % n])
                edge = (source_identity, int(topology.edge_type[edge_id]))
                actual_sets[batch_index].add(edge)
                actual_strengths[batch_index][edge] = float(topology.strength[edge_id])
            expected_sets = [
                {(source, edge_type) for source, edge_type, _strength in rows}
                for rows in data["oracle_edges"]
            ]
            tp = sum(len(actual & expected) for actual, expected in zip(actual_sets, expected_sets, strict=True))
            fp = sum(len(actual - expected) for actual, expected in zip(actual_sets, expected_sets, strict=True))
            fn = sum(len(expected - actual) for actual, expected in zip(actual_sets, expected_sets, strict=True))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            matching_strengths = [
                abs(actual_strengths[i][(source, edge_type)] - expected_strength)
                for i, rows in enumerate(data["oracle_edges"])
                for source, edge_type, expected_strength in rows
                if (source, edge_type) in actual_strengths[i]
            ]
            strength_mae = statistics.mean(matching_strengths) if matching_strengths else None
            op_counts = {name: 0 for name in ("add", "remove", "retype", "set_strength")}
            predicted_ops: list[dict[tuple[int, int], str]] = [{} for _ in actual_sets]
            for mutation in hard["mutations"]:
                op_counts[mutation.operation] += 1
                batch_index = mutation.batch_index
                src_slot = mutation.source
                dst_slot = mutation.target
                src_id = int(data["node_ids"][batch_index, src_slot])
                dst_id = int(data["node_ids"][batch_index, dst_slot])
                predicted_ops[batch_index][(src_id, dst_id)] = mutation.operation
            operation_scores: dict[str, dict[str, float | int]] = {}
            for op in op_counts:
                expected_items = [
                    (i, edge) for i, mapping in enumerate(data["expected_operations"])
                    for edge, expected_op in mapping.items() if expected_op == op
                ]
                predicted_items = [
                    (i, edge) for i, mapping in enumerate(predicted_ops)
                    for edge, predicted_op in mapping.items() if predicted_op == op
                ]
                true_positive = sum(
                    data["expected_operations"][i].get(edge) == op
                    and predicted_ops[i].get(edge) == op
                    for i, edge in expected_items
                )
                operation_scores[op] = {
                    "expected_count": len(expected_items),
                    "predicted_count": len(predicted_items),
                    "correct_count": true_positive,
                    "precision": true_positive / max(1, len(predicted_items)),
                    "recall": true_positive / max(1, len(expected_items)),
                }
            changed_pairs = []
            for left in range(len(actual_sets)):
                for right in range(left + 1, min(left + 9, len(actual_sets))):
                    if expected_sets[left] != expected_sets[right]:
                        changed_pairs.append(actual_sets[left] != actual_sets[right])
            dynamic_change = sum(changed_pairs) / len(changed_pairs) if changed_pairs else None
            random_mse = _random_matched_mse(model, data, actual_sets, actual_strengths, seed)
            candidate_count = int(soft["logits"].shape[0])
            edge_sets = actual_sets
            edge_op_counts = op_counts
        else:
            prediction = model(_canonical_features(data["features"], data["node_ids"], model.key_space))
            soft_mse = None
            precision = recall = f1 = strength_mae = dynamic_change = random_mse = None
            operation_scores = {}
            candidate_count = 0
            edge_sets = []
            edge_op_counts = {}
        mse = F.mse_loss(prediction, data["labels"]).item()
        mae = F.l1_loss(prediction, data["labels"]).item()
    return {
        "mse": mse,
        "soft_mse": soft_mse,
        "random_matched_mse": random_mse,
        "mae": mae,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "strength_mae_on_correct_edges": strength_mae,
        "query_conditioned_link_change_rate": dynamic_change,
        "query_pair_count": len(changed_pairs) if isinstance(model, IdentityGraphModel) else 0,
        "operation_scores": operation_scores,
        "operation_counts": edge_op_counts,
        "candidate_count": candidate_count,
        "_edge_sets": edge_sets,
        "_topology": hard["topology"] if isinstance(model, IdentityGraphModel) else None,
    }


def _random_matched_mse(
    model: IdentityGraphModel,
    data: dict[str, Any],
    edge_sets: list[set[tuple[int, int]]],
    strengths: list[dict[tuple[int, int], float]],
    seed: int,
) -> float:
    rng = random.Random(seed)
    features, ids = data["features"], data["node_ids"]
    batch, n, _ = features.shape
    srcs: list[int] = []
    tgts: list[int] = []
    kinds: list[int] = []
    weights: list[float] = []
    for b, selected in enumerate(edge_sets):
        valid_ids = [int(value) for value in ids[b].tolist() if value >= 0 and int(value) != model.readout_identity]
        shuffled = valid_ids[:]
        rng.shuffle(shuffled)
        random_source_ids = shuffled[: len(selected)]
        readout_slot = int(features[b, :, -1].argmax())
        identity_to_slot = {int(identity): slot for slot, identity in enumerate(ids[b].tolist()) if identity >= 0}
        existing_attrs = sorted(
            ((kind, strengths[b][(source, kind)]) for source, kind in selected),
            key=lambda item: (item[0], item[1]),
        )
        for source_identity, (kind, strength) in zip(random_source_ids, existing_attrs, strict=True):
            srcs.append(b * n + identity_to_slot[source_identity])
            tgts.append(b * n + readout_slot)
            kinds.append(kind)
            weights.append(strength)
    topology = NeuronGraphTopology(
        batch,
        n,
        torch.tensor([srcs, tgts], dtype=torch.long, device=features.device),
        torch.tensor(kinds, dtype=torch.long, device=features.device),
        torch.tensor(weights, dtype=torch.float32, device=features.device),
        ids,
    )
    encoded = model.encoder(features)
    output = model.graph(encoded, topology)
    readout_slots = features[:, :, -1].gt(0.5).nonzero(as_tuple=False)[:, 1]
    rows = torch.arange(batch, device=features.device)
    prediction = model.head(output.hidden_states[rows, readout_slots]).squeeze(-1)
    return F.mse_loss(prediction, data["labels"]).item()


def _train_one(
    model: nn.Module,
    data: dict[str, Any],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    sample_interval_us: int,
    hard_action_fraction: float,
) -> tuple[float, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(seed)
    _sync(device)
    started = time.perf_counter()
    with _MemorySampler(device, sample_interval_us / 1_000_000) as memory:
        model.train()
        for step in range(steps):
            if isinstance(model, IdentityGraphModel):
                model.hard_action_active = step >= int(steps * (1.0 - hard_action_fraction))
            generator = torch.Generator(device="cpu").manual_seed(seed * 100_000 + step)
            indices = torch.randperm(data["features"].shape[0], generator=generator)[:batch_size].to(device)
            batch = _subset(data, indices)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, IdentityGraphModel):
                prediction = model(
                    batch["features"], batch["node_ids"], batch["initial_edge_index"],
                    batch["initial_edge_type"], batch["initial_strength"],
                )["prediction"]
            else:
                prediction = model(_canonical_features(batch["features"], batch["node_ids"], model.key_space))
            loss = F.mse_loss(prediction, batch["labels"])
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - started
    return elapsed, memory.result()


def _macro_summary(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    rules = config["decision_rules"]
    for task in config["task"]["tasks"]:
        rows = [case for case in cases if case["task"] == task]
        task_out: dict[str, Any] = {"seed_count": len(rows)}
        for baseline in ("parameter_matched_mlp", "linear_mac_matched_mlp"):
            rel = [
                (case["graph"]["test"]["mse"] - case[baseline]["test"]["mse"])
                / max(case[baseline]["test"]["mse"], 1e-12)
                for case in rows
            ]
            n = len(rel)
            mean = statistics.mean(rel) if rel else None
            sd = statistics.stdev(rel) if n > 1 else None
            half = crit.get(n - 1, 1.96) * sd / math.sqrt(n) if sd is not None else None
            upper = mean + half if mean is not None and half is not None else None
            task_out[baseline] = {
                "paired_relative_mse_delta": mean,
                "sample_std": sd,
                "ci95": [mean - half, upper] if mean is not None and half is not None else None,
                "criterion_met": bool(mean is not None and upper is not None and mean <= -rules["minimum_relative_mse_improvement"] and upper < 0),
            }
        task_out["edge_f1_mean"] = statistics.mean(c["graph"]["test"]["edge_f1"] for c in rows)
        task_out["strength_mae_mean"] = statistics.mean(c["graph"]["test"]["strength_mae_on_correct_edges"] for c in rows)
        task_out["permutation_mean_mse_drift"] = statistics.mean(c["graph"]["permutation"]["relative_mse_drift"] for c in rows)
        task_out["permutation_edge_jaccard_mean"] = statistics.mean(c["graph"]["permutation"]["stable_identity_edge_jaccard"] for c in rows)
        op_summary = {}
        for op in ("add", "remove", "retype", "set_strength"):
            expected = sum(c["graph"]["test"]["operation_scores"][op]["expected_count"] for c in rows)
            correct = sum(c["graph"]["test"]["operation_scores"][op]["correct_count"] for c in rows)
            predicted = sum(c["graph"]["test"]["operation_scores"][op]["predicted_count"] for c in rows)
            op_summary[op] = {
                "expected_count": expected,
                "predicted_count": predicted,
                "precision": correct / max(1, predicted),
                "recall": correct / max(1, expected),
            }
        task_out["operation_summary"] = op_summary
        task_out["structure_criterion_met"] = (
            task_out["edge_f1_mean"] >= rules["minimum_edge_f1"]
            and task_out["strength_mae_mean"] <= rules["maximum_strength_mae"]
            and task_out["permutation_mean_mse_drift"] <= rules["maximum_permutation_relative_mse_drift"]
            and task_out["permutation_edge_jaccard_mean"] >= rules["minimum_permutation_identity_edge_jaccard"]
            and all(
                op_summary[op]["expected_count"] == 0
                or op_summary[op]["recall"] >= rules["minimum_operation_recall"]
                for op in ("add", "remove", "retype", "set_strength")
            )
        )
        summary[task] = task_out
    return summary


def _identity_edge_jaccard(left: list[set[tuple[int, int]]], right: list[set[tuple[int, int]]]) -> float:
    values = []
    for first, second in zip(left, right, strict=True):
        values.append(len(first & second) / max(1, len(first | second)))
    return statistics.mean(values) if values else 1.0


def _mean_or_none(values: Any) -> float | None:
    selected = [value for value in values if value is not None]
    return statistics.mean(selected) if selected else None


def _select_threshold(
    model: IdentityGraphModel, validation: dict[str, Any], config: dict[str, Any], seed: int
) -> tuple[float, list[dict[str, Any]]]:
    trials = []
    for threshold in config["validation_selection"]["action_confidence_thresholds"]:
        metrics = _metrics(model, validation, threshold=threshold, seed=seed)
        active_recalls = [
            metrics["operation_scores"][operation]["recall"]
            for operation in ("add", "remove", "retype", "set_strength")
            if metrics["operation_scores"][operation]["expected_count"] > 0
        ]
        operation_recall = statistics.mean(active_recalls) if active_recalls else 0.0
        score = (
            config["validation_selection"]["edge_f1_weight"] * metrics["edge_f1"]
            + config["validation_selection"]["operation_recall_weight"] * operation_recall
        )
        trials.append({
            "threshold": threshold,
            "selection_score": score,
            "edge_f1": metrics["edge_f1"],
            "active_operation_macro_recall": operation_recall,
            "validation_mse": metrics["mse"],
        })
    selected = max(
        trials,
        key=lambda trial: (
            trial["selection_score"],
            -trial["validation_mse"],
            trial["threshold"],
        ),
    )
    return float(selected["threshold"]), trials


def _expected_transition_operations(
    data: dict[str, Any], topology: NeuronGraphTopology, *, strength_threshold: float
) -> dict[tuple[int, int], str]:
    node_ids = data["node_ids"][0]
    current: dict[tuple[int, int], tuple[int, float]] = {}
    for index, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist()):
        source_id = int(node_ids[source])
        target_id = int(node_ids[target])
        if source_id >= 0 and target_id >= 0:
            current[(source_id, target_id)] = (
                int(topology.edge_type[index]), float(topology.strength[index])
            )
    desired = {
        (source, data["readout_identity"]): (edge_type, strength)
        for source, edge_type, strength in data["oracle_edges"][0]
    }
    operations: dict[tuple[int, int], str] = {}
    for edge, (edge_type, strength) in desired.items():
        existing = current.get(edge)
        if existing is None:
            operations[edge] = "add"
        elif existing[0] != edge_type:
            operations[edge] = "retype"
        elif abs(existing[1] - strength) >= strength_threshold:
            operations[edge] = "set_strength"
    for edge in current.keys() - desired.keys():
        operations[edge] = "remove"
    return operations


def _run_topology_evolution(
    model: IdentityGraphModel,
    *,
    task: str,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    identity_ids: dict[int, int],
    readout_identity: int,
    threshold: float,
) -> dict[str, Any]:
    sequence_count = config["evolution"]["sequences_per_seed_task"]
    sequence_length = config["evolution"]["steps_per_sequence"]
    pool_size = config["task"]["training_memory_neurons"]
    all_steps = []
    topology: NeuronGraphTopology | None = None
    churn = []
    for sequence in range(sequence_count):
        topology = None
        previous_edges: set[tuple[int, int, int]] = set()
        for step in range(sequence_length):
            frame_seed = config["evolution"]["seed_base"] + seed * 100_000 + sequence * 1_000 + step
            frame = _data(
                frame_seed,
                1,
                memory_count=pool_size,
                task=task,
                key_space=config["task"]["key_space"],
                device=device,
                identity_ids=identity_ids,
                readout_identity=readout_identity,
            )
            if topology is None:
                topology = NeuronGraphTopology(
                    1,
                    frame["features"].shape[1],
                    frame["initial_edge_index"],
                    frame["initial_edge_type"],
                    frame["initial_strength"],
                    frame["node_ids"],
                )
            else:
                topology = topology.remap_node_ids(frame["node_ids"], causal=False)
            frame["readout_identity"] = readout_identity
            frame["expected_operations"] = [
                _expected_transition_operations(
                    frame,
                    topology,
                    strength_threshold=config["model"]["strength_update_threshold"],
                )
            ]
            frame["initial_edge_index"] = topology.edge_index
            frame["initial_edge_type"] = topology.edge_type
            frame["initial_strength"] = topology.strength
            metrics = _metrics(model, frame, threshold=threshold, seed=frame_seed)
            metrics = _metrics(model, frame, threshold=threshold, seed=frame_seed)
            next_topology = metrics["_topology"]
            edges = {
                (source, edge_type, float(strength))
                for (source, edge_type), strength in _topology_identity_edges(next_topology, frame["node_ids"]).items()
            }
            churn.append(len(previous_edges.symmetric_difference(edges)))
            previous_edges = edges
            topology = next_topology
            step_result = {key: value for key, value in metrics.items() if not key.startswith("_")}
            step_result["hard_topology_edge_count"] = topology.edge_count
            all_steps.append(step_result)
    operation_recalls = {}
    for operation in ("add", "remove", "retype", "set_strength"):
        values = [
            step["operation_scores"][operation]["recall"]
            for step in all_steps
            if step["operation_scores"][operation]["expected_count"] > 0
        ]
        operation_recalls[operation] = statistics.mean(values) if values else None
    return {
        "sequence_count": sequence_count,
        "steps_per_sequence": sequence_length,
        "total_transitions": len(all_steps),
        "mean_step_mse": statistics.mean(row["mse"] for row in all_steps),
        "mean_edge_f1": statistics.mean(row["edge_f1"] for row in all_steps),
        "mean_hard_topology_edge_count": statistics.mean(row["hard_topology_edge_count"] for row in all_steps),
        "mean_identity_edge_state_churn": statistics.mean(churn),
        "operation_recall_by_type": operation_recalls,
        "operations_selected_threshold": threshold,
    }


def _topology_identity_edges(
    topology: NeuronGraphTopology, node_ids: torch.Tensor
) -> dict[tuple[int, int], float]:
    result = {}
    for index, (source, target) in enumerate(topology.edge_index.t().detach().cpu().tolist()):
        source_id = int(node_ids[0, source])
        target_id = int(node_ids[0, target])
        if source_id >= 0 and target_id >= 0:
            result[(source_id, int(topology.edge_type[index]))] = float(topology.strength[index])
    return result


def _scale_benchmark(model: IdentityGraphModel, config: dict[str, Any], device: torch.device, identity_ids: dict[int, int], readout_identity: int) -> list[dict[str, Any]]:
    results = []
    for count in config["task"]["evaluation_memory_neurons"]:
        data = _data(900 + count, 1, memory_count=count, task="two_link_composition", key_space=config["task"]["key_space"], device=device, identity_ids=identity_ids, readout_identity=readout_identity)
        for _ in range(config["cost_accounting"]["scale_warmup_calls"]):
            model(data["features"], data["node_ids"], data["initial_edge_index"], data["initial_edge_type"], data["initial_strength"])
        _sync(device)
        times = []
        with _MemorySampler(device, config["cost_accounting"]["memory_sampling_interval_microseconds"] / 1_000_000) as memory:
            for _ in range(config["cost_accounting"]["scale_measured_calls"]):
                started = time.perf_counter()
                model(data["features"], data["node_ids"], data["initial_edge_index"], data["initial_edge_type"], data["initial_strength"])
                _sync(device)
                times.append((time.perf_counter() - started) * 1000)
        results.append({
            "memory_neurons": count,
            "valid_node_count": count + 1,
            "candidate_edges": count,
            "median_forward_ms": statistics.median(times),
            "p95_forward_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
            "sampled_memory": memory.result(),
        })
    return results


def run(config: dict[str, Any], config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    device = torch.device(config["training"]["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("v6 config requires Apple MPS; refusing to fall back")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    torch.set_num_threads(config["training"]["threads"])
    key_space = config["task"]["key_space"]
    max_nodes = key_space + 1
    bit_width = math.ceil(math.log2(key_space))
    feature_size = bit_width * 2 + 2
    training_pool = config["task"]["training_memory_neurons"]
    registry = NeuronIdentityRegistry(config["task"]["identity_namespace"])
    for key in range(key_space):
        registry.register(f"memory/{key}")
    registry.register("readout")
    restored_registry = NeuronIdentityRegistry.from_state_dict(json.loads(json.dumps(registry.state_dict())))
    if restored_registry.state_dict() != registry.state_dict():
        raise RuntimeError("identity registry failed its serialization round trip")
    registry = restored_registry
    identity_ids = {key: registry.resolve(f"memory/{key}") for key in range(key_space)}
    readout_identity = registry.resolve("readout")
    if any(value is None for value in identity_ids.values()) or readout_identity is None:
        raise RuntimeError("restored registry omitted a required stable identity")
    torch.manual_seed(config["training"]["seeds"][0])
    probe_data = _data(10, 2, memory_count=training_pool, task=config["task"]["tasks"][0], key_space=key_space, device=device, identity_ids=identity_ids, readout_identity=readout_identity)
    probe = IdentityGraphModel(feature_size, config["model"]["hidden_size"], key_space, readout_identity).to(device)
    graph_params = sum(p.numel() for p in probe.parameters())
    graph_macs = _linear_macs(probe, probe_data)
    flat_input = feature_size * max_nodes
    param_width = _closest_width(graph_params, flat_input, parameters=True)
    mac_width = _closest_width(graph_macs, flat_input, parameters=False)
    param_probe = FlatControl(feature_size, max_nodes, param_width, key_space).to(device)
    mac_probe = FlatControl(feature_size, max_nodes, mac_width, key_space).to(device)
    param_macs = _linear_macs(param_probe, probe_data)
    mac_macs = _linear_macs(mac_probe, probe_data)
    control_costs = {
        "graph": {"parameters": graph_params, "linear_macs_per_example": graph_macs},
        "parameter_matched_mlp": {"hidden_size": param_width, "parameters": sum(p.numel() for p in param_probe.parameters()), "linear_macs_per_example": param_macs},
        "linear_mac_matched_mlp": {"hidden_size": mac_width, "parameters": sum(p.numel() for p in mac_probe.parameters()), "linear_macs_per_example": mac_macs},
        "parameter_relative_error": abs(sum(p.numel() for p in param_probe.parameters()) - graph_params) / graph_params,
        "linear_mac_relative_error": abs(mac_macs - graph_macs) / graph_macs,
        "training_time_mean_seconds": {},
        "linear_mac_proxy_contract": config["controls"]["linear_mac_contract"],
        "multiply_add_flops_subtotal_per_example": graph_macs * 2,
        "total_flops_exactly_counted": False,
        "profiler_supported_activities": sorted(
            activity.name for activity in torch.profiler.supported_activities()
        ),
        "profiler_flops_note": config["cost_accounting"]["flops_scope"],
    }
    cases = []
    all_train_times: dict[str, list[float]] = {"graph": [], "parameter_matched_mlp": [], "linear_mac_matched_mlp": []}
    for task in config["task"]["tasks"]:
        for seed in config["training"]["seeds"]:
            train = _data(seed + 100, config["task"]["training_examples"], memory_count=training_pool, task=task, key_space=key_space, device=device, identity_ids=identity_ids, readout_identity=readout_identity)
            validation = _data(seed + 200, config["task"]["validation_examples"], memory_count=training_pool, task=task, key_space=key_space, device=device, identity_ids=identity_ids, readout_identity=readout_identity)
            test = _data(seed + 300, config["task"]["test_examples"], memory_count=training_pool, task=task, key_space=key_space, device=device, identity_ids=identity_ids, readout_identity=readout_identity)
            torch.manual_seed(seed)
            graph_model = IdentityGraphModel(feature_size, config["model"]["hidden_size"], key_space, readout_identity).to(device)
            torch.manual_seed(seed + 1)
            param_model = FlatControl(feature_size, max_nodes, param_width, key_space).to(device)
            torch.manual_seed(seed + 2)
            mac_model = FlatControl(feature_size, max_nodes, mac_width, key_space).to(device)
            model_results = {}
            for name, model in (("graph", graph_model), ("parameter_matched_mlp", param_model), ("linear_mac_matched_mlp", mac_model)):
                elapsed, memory = _train_one(
                    model, train, seed=seed, steps=config["training"]["steps"],
                    batch_size=config["training"]["batch_size"], lr=config["training"]["learning_rate"],
                    device=device, sample_interval_us=config["cost_accounting"]["memory_sampling_interval_microseconds"],
                    hard_action_fraction=config["training"]["hard_action_training_fraction"],
                )
                all_train_times[name].append(elapsed)
                if isinstance(model, IdentityGraphModel):
                    selected_threshold, threshold_trials = _select_threshold(
                        model, validation, config, seed
                    )
                    initial = _metrics(model, validation, threshold=selected_threshold, seed=seed)
                    graph_test = _metrics(model, test, threshold=selected_threshold, seed=seed + 1)
                    permuted_test = _metrics(model, _permute_data(test, seed + 2), threshold=selected_threshold, seed=seed + 3)
                    drift = abs(permuted_test["mse"] - graph_test["mse"]) / max(graph_test["mse"], 1e-12)
                    evolution = _run_topology_evolution(
                        model,
                        task=task,
                        seed=seed,
                        config=config,
                        device=device,
                        identity_ids=identity_ids,
                        readout_identity=readout_identity,
                        threshold=selected_threshold,
                    )
                    model_results[name] = {
                        "test": {k: v for k, v in graph_test.items() if not k.startswith("_")},
                        "validation": {k: v for k, v in initial.items() if not k.startswith("_")},
                        "permutation": {
                            "relative_mse_drift": drift,
                            "stable_identity_edge_jaccard": _identity_edge_jaccard(graph_test["_edge_sets"], permuted_test["_edge_sets"]),
                            "test_pair_count": int(test["features"].shape[0]),
                        },
                        "validation_threshold_selection": {
                            "selected_threshold": selected_threshold,
                            "selection_rule": config["validation_selection"]["rule"],
                            "trials": threshold_trials,
                        },
                        "continuous_evolution": evolution,
                        "train_seconds": elapsed,
                        "training_memory": memory,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                    }
                else:
                    metric = _metrics(model, test, threshold=config["model"]["hard_link_threshold"], seed=seed)
                    model_results[name] = {
                        "test": {k: v for k, v in metric.items() if not k.startswith("_")},
                        "train_seconds": elapsed,
                        "training_memory": memory,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                    }
            variable = {}
            for memory_count in config["task"]["evaluation_memory_neurons"]:
                eval_data = _data(seed + 500 + memory_count, config["task"]["test_examples"], memory_count=memory_count, task=task, key_space=key_space, device=device, identity_ids=identity_ids, readout_identity=readout_identity)
                selected_threshold = model_results["graph"]["validation_threshold_selection"]["selected_threshold"]
                gm = _metrics(graph_model, eval_data, threshold=selected_threshold, seed=seed + memory_count)
                pm = _metrics(param_model, eval_data, threshold=config["model"]["hard_link_threshold"], seed=seed + memory_count)
                cm = _metrics(mac_model, eval_data, threshold=config["model"]["hard_link_threshold"], seed=seed + memory_count)
                variable[str(memory_count)] = {
                    "graph": {k: v for k, v in gm.items() if not k.startswith("_")},
                    "parameter_matched_mlp_mse": pm["mse"],
                    "linear_mac_matched_mlp_mse": cm["mse"],
                }
            cases.append({"task": task, "seed": seed, **model_results, "variable_pool_evaluation": variable})
    control_costs["training_time_mean_seconds"] = {
        name: statistics.mean(values) for name, values in all_train_times.items()
    }
    summaries = _macro_summary(cases, config)
    scale_model = IdentityGraphModel(feature_size, config["model"]["hidden_size"], key_space, readout_identity).to(device)
    scaling = _scale_benchmark(scale_model, config, device, identity_ids, readout_identity)
    size_summary = {}
    for task in config["task"]["tasks"]:
        task_cases = [case for case in cases if case["task"] == task]
        size_summary[task] = {
            str(size): {
                "graph_edge_f1_mean": statistics.mean(case["variable_pool_evaluation"][str(size)]["graph"]["edge_f1"] for case in task_cases),
                "graph_strength_mae_mean": _mean_or_none(
                    case["variable_pool_evaluation"][str(size)]["graph"]["strength_mae_on_correct_edges"]
                    for case in task_cases
                ),
            }
            for size in config["task"]["evaluation_memory_neurons"]
        }
    source_files = [
        "project/scripts/experiments/training/neurons/graph/run_neuron_graph_residual_validation.py",
        "src/newera/neuron/graph/policy.py",
        "src/newera/neuron/graph/topology.py",
        "src/newera/neuron/graph/__init__.py",
        "src/newera/neuron/graph/layer.py",
        "src/newera/nlp_flow/model/components/architecture/foundation/config.py",
        "src/newera/nlp_flow/model/components/architecture/foundation/model.py",
        "pyproject.toml",
        "uv.lock",
    ]
    return {
        "schema_version": "newera-neuron-topology-evolution-v6",
        "run_id": config["run_id"],
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
        "task_contract": config["task"],
        "identity_registry": {
            "schema": registry.SCHEMA,
            "namespace": registry.namespace,
            "count": len(registry.state_dict()["entries"]),
            "state_round_trip_ok": True,
        },
        "_identity_registry_state": registry.state_dict(),
        "decision_rules": config["decision_rules"],
        "cost_alignment": control_costs,
        "training_cases": cases,
        "summary": summaries,
        "variable_pool_summary": size_summary,
        "scaling": scaling,
        "limitations": config["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry_state = result.pop("_identity_registry_state")
    registry_path = args.output.with_name("identity_registry.json")
    NeuronIdentityRegistry.from_state_dict(registry_state).save(registry_path)
    restored_registry = NeuronIdentityRegistry.load(registry_path)
    post_restart_id = restored_registry.register("post-restart/new-neuron")
    restored_registry.save(registry_path)
    reloaded_registry = NeuronIdentityRegistry.load(registry_path)
    result["identity_registry"]["saved_path"] = registry_path.name
    result["identity_registry"]["first_id_after_restart"] = post_restart_id
    result["identity_registry"]["count_after_restart"] = len(reloaded_registry.state_dict()["entries"])
    result["identity_registry"]["disk_round_trip_ok"] = (
        reloaded_registry.state_dict() == restored_registry.state_dict()
        and reloaded_registry.resolve("post-restart/new-neuron") == post_restart_id
    )
    result["identity_registry"]["sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"run_id": result["run_id"], "device": result["device"], "cases": len(result["training_cases"]), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
