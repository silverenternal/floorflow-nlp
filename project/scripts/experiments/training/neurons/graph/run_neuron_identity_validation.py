"""Train task driven links between persistent, non-token neuron identities."""
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
    TopologyController,
)

ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "project/configs/models_training/learning/neuron_comparisons/graph/graph_validation_learned_identity_links.json"


class _MemorySampler:
    def __init__(self, device: torch.device, interval_seconds: float) -> None:
        self.device = device
        self.interval = interval_seconds
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak_rss = self.process.memory_info().rss
        self.start_rss = self.peak_rss
        self.peak_device = _device_memory(device)
        self.start_device = dict(self.peak_device)
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            current = _device_memory(self.device)
            for name, value in current.items():
                self.peak_device[name] = max(self.peak_device.get(name, 0), value)
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
        for name, value in _device_memory(self.device).items():
            self.peak_device[name] = max(self.peak_device.get(name, 0), value)

    def result(self) -> dict[str, Any]:
        return {
            "host_rss_start_bytes": self.start_rss,
            "host_rss_sampled_peak_delta_bytes": max(0, self.peak_rss - self.start_rss),
            "device_start_bytes": self.start_device,
            "device_sampled_peak_delta_bytes": {
                name: value - self.start_device.get(name, 0)
                for name, value in self.peak_device.items()
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
    def __init__(self, node_count: int, input_features: int, hidden_size: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_size), nn.Tanh())
        self.policy = LinkProposalPolicy(hidden_size, 1, initial_keep_bias=0.0)
        self.graph = DynamicNeuronGraphLayer(hidden_size, 1, causal=False)
        with torch.no_grad():
            self.graph.residual_gate.fill_(1.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        sources = torch.arange(node_count - 1, dtype=torch.long)
        target = torch.full_like(sources, node_count - 1)
        self.register_buffer("candidate_local", torch.stack((sources, target)))

    def forward(self, features: torch.Tensor, *, hard: bool = False, threshold: float = 0.5) -> dict[str, Any]:
        batch_size = features.shape[0]
        ids = torch.arange(self.node_count, device=features.device).expand(batch_size, -1)
        topology = NeuronGraphTopology.empty(
            batch_size, self.node_count, device=features.device, node_ids=ids
        )
        offsets = torch.arange(batch_size, device=features.device)[:, None] * self.node_count
        candidates = (self.candidate_local.to(features.device)[None, :, :] + offsets[:, None, :]).permute(1, 0, 2).reshape(2, -1)
        encoded = self.encoder(features)
        logits = self.policy(encoded, candidates, topology, causal=False)
        probabilities = logits.softmax(dim=-1)
        mutations = []
        if hard:
            mutations = self.policy.propose(logits, candidates, topology, threshold=threshold)
            topology = TopologyController(causal=False).apply(
                topology, mutations, num_edge_types=1
            )
            output = self.graph(encoded, topology)
        else:
            weights = self.policy.expected_type_weights(logits, candidates, topology)
            output = self.graph(encoded, topology, candidate_edges=candidates, candidate_type_weights=weights)
        prediction = self.head(output.hidden_states[:, -1]).squeeze(-1)
        return {
            "prediction": prediction,
            "logits": logits,
            "probabilities": probabilities,
            "topology": topology,
            "mutations": mutations,
        }


class FlatControl(nn.Module):
    def __init__(self, node_count: int, input_features: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(node_count * input_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.flatten(start_dim=1)).squeeze(-1)


def _data(seed: int, count: int, *, key_count: int, task: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand((count, key_count), generator=generator) * 2.0 - 1.0
    query = torch.randint(key_count, (count,), generator=generator)
    node_count = key_count + 1
    feature_count = 2 * key_count + 2
    features = torch.zeros(count, node_count, feature_count)
    # Each persistent memory neuron has a fixed key and a variable value.
    features[:, :key_count, :key_count] = torch.eye(key_count).unsqueeze(0).expand(count, -1, -1)
    features[:, :key_count, key_count] = values
    readout_start = key_count + 1
    features[torch.arange(count), key_count, readout_start + query] = 1.0
    features[:, key_count, -1] = 1.0
    if task == "single_link_retrieval":
        labels = values.gather(1, query[:, None]).squeeze(1)
        oracle = query[:, None]
    elif task == "two_link_composition":
        second = (query + 1) % key_count
        labels = values.gather(1, query[:, None]).squeeze(1) + values.gather(1, second[:, None]).squeeze(1)
        oracle = torch.stack((query, second), dim=1)
    else:
        raise ValueError(f"unknown task: {task}")
    return features.to(device), labels.to(device), oracle.to(device)


def _linear_macs(module: nn.Module, *args: torch.Tensor) -> int:
    total = 0
    handles = []
    def count(layer: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features
    for child in module.modules():
        if isinstance(child, nn.Linear):
            handles.append(child.register_forward_hook(count))
    with torch.no_grad():
        module(*args)
    for handle in handles:
        handle.remove()
    return total


def _closest_width(target: int, input_features: int, *, parameter_count: bool) -> int:
    def cost(width: int) -> int:
        return width * (input_features + 2) + 1 if parameter_count else width * (input_features + 1)
    return min(range(1, max(2, target // max(1, input_features) * 2 + 2)), key=lambda width: abs(cost(width) - target))


def _metrics(model: IdentityGraphModel | FlatControl, features: torch.Tensor, labels: torch.Tensor, oracle: torch.Tensor, *, threshold: float, random_seed: int = 0) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        if isinstance(model, IdentityGraphModel):
            soft = model(features)
            hard = model(features, hard=True, threshold=threshold)
            prediction = hard["prediction"]
            soft_mse = F.mse_loss(soft["prediction"], labels).item()
            candidate_count = soft["logits"].shape[0]
            predicted_edges: list[set[int]] = []
            node_count = model.node_count
            for batch_index in range(features.shape[0]):
                selected = set()
                for source, target in hard["topology"].edge_index.t().tolist():
                    if target // node_count == batch_index:
                        selected.add(source % node_count)
                predicted_edges.append(selected)
            true_edges = [set(row.tolist()) for row in oracle]
            tp = sum(len(actual & expected) for actual, expected in zip(predicted_edges, true_edges, strict=True))
            fp = sum(len(actual - expected) for actual, expected in zip(predicted_edges, true_edges, strict=True))
            fn = sum(len(expected - actual) for actual, expected in zip(predicted_edges, true_edges, strict=True))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            correct_query_changes = []
            for left in range(len(predicted_edges)):
                for right in range(left + 1, min(left + 9, len(predicted_edges))):
                    if not torch.equal(oracle[left], oracle[right]):
                        correct_query_changes.append(predicted_edges[left] != predicted_edges[right])
            mutation_rate = sum(bool(edges) for edges in predicted_edges) / max(1, len(predicted_edges))
            entropy = -(soft["probabilities"].mean(dim=0) * soft["probabilities"].mean(dim=0).clamp_min(1e-12).log()).sum()
            generator = random.Random(random_seed)
            random_mutations = []
            for batch_index, selected in enumerate(predicted_edges):
                random_sources = generator.sample(range(model.node_count - 1), len(selected))
                random_mutations.extend(
                    LinkMutation("add", batch_index, source, model.node_count - 1, 0, 1.0)
                    for source in random_sources
                )
            random_topology = NeuronGraphTopology.empty(
                features.shape[0], model.node_count, device=features.device,
                node_ids=hard["topology"].node_ids,
            )
            random_topology = TopologyController(causal=False).apply(
                random_topology, random_mutations, num_edge_types=1
            )
            random_graph_output = model.graph(model.encoder(features), random_topology)
            random_prediction = model.head(random_graph_output.hidden_states[:, -1]).squeeze(-1)
            random_matched_mse = F.mse_loss(random_prediction, labels).item()
        else:
            prediction = model(features)
            precision = recall = f1 = mutation_rate = None
            correct_query_changes = []
            entropy = None
            candidate_count = 0
            soft_mse = None
            random_matched_mse = None
        mse = F.mse_loss(prediction, labels).item()
        mae = F.l1_loss(prediction, labels).item()
    return {
        "mse": mse,
        "soft_mse": soft_mse,
        "random_matched_mse": random_matched_mse,
        "mae": mae,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "per_example_mutation_rate": mutation_rate,
        "changed_query_pair_count": len(correct_query_changes),
        "changed_query_pair_link_change_rate": (
            sum(correct_query_changes) / len(correct_query_changes) if correct_query_changes else None
        ),
        "action_entropy": float(entropy.item()) if entropy is not None else None,
        "candidate_count": candidate_count,
    }


def _train(model: nn.Module, task: str, seed: int, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    key_count = int(config["task"]["memory_neurons"])
    train_x, train_y, _ = _data(seed + 100, int(config["task"]["training_examples"]), key_count=key_count, task=task, device=device)
    validation_x, validation_y, validation_oracle = _data(seed + 200, int(config["task"]["validation_examples"]), key_count=key_count, task=task, device=device)
    test_x, test_y, test_oracle = _data(seed + 300, int(config["task"]["test_examples"]), key_count=key_count, task=task, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    batch = int(config["training"]["batch_size"])
    steps = int(config["training"]["steps"])
    _sync(device)
    started = time.perf_counter()
    with _MemorySampler(device, int(config["memory_sampling_interval_microseconds"]) / 1_000_000) as sampler:
        model.train()
        for step in range(steps):
            indices = torch.randint(train_x.shape[0], (batch,), generator=torch.Generator(device="cpu").manual_seed(seed * 10000 + step)).to(device)
            x, y = train_x.index_select(0, indices), train_y.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, IdentityGraphModel):
                prediction = model(x)["prediction"]
            else:
                prediction = model(x)
            loss = F.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - started
    memory = sampler.result()
    metrics = _metrics(model, test_x, test_y, test_oracle, threshold=float(config["decision_rules"]["hard_link_threshold"]), random_seed=seed)
    validation = _metrics(model, validation_x, validation_y, validation_oracle, threshold=float(config["decision_rules"]["hard_link_threshold"]), random_seed=seed + 1)
    return {
        "test": metrics,
        "validation": validation,
        "train_seconds": elapsed,
        "training_memory": memory,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "train_data_sha256": _tensor_hash(train_x, train_y),
        "validation_data_sha256": _tensor_hash(validation_x, validation_y),
        "test_data_sha256": _tensor_hash(test_x, test_y),
    }


def _tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _scaling(model: IdentityGraphModel, config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    hidden = model.encoder[0].out_features
    policy = model.policy
    graph = model.graph
    for node_count in config["scaling"]["node_counts"]:
        nodes = torch.randn(1, node_count, hidden, device=device)
        ids = torch.arange(node_count, device=device).unsqueeze(0)
        topology = NeuronGraphTopology.empty(1, node_count, device=device, node_ids=ids)
        source = torch.arange(node_count, device=device).repeat_interleave(node_count)
        target = torch.arange(node_count, device=device).repeat(node_count)
        keep = source != target
        candidates = torch.stack((source[keep], target[keep]))
        for _ in range(int(config["scaling"]["warmup_calls"])):
            logits = policy(nodes, candidates, topology, causal=False)
            weights = policy.expected_type_weights(logits, candidates, topology)
            graph(nodes, topology, candidate_edges=candidates, candidate_type_weights=weights)
        _sync(device)
        times = []
        with _MemorySampler(device, int(config["memory_sampling_interval_microseconds"]) / 1_000_000) as sampler:
            for _ in range(int(config["scaling"]["measured_calls"])):
                start = time.perf_counter()
                logits = policy(nodes, candidates, topology, causal=False)
                weights = policy.expected_type_weights(logits, candidates, topology)
                graph(nodes, topology, candidate_edges=candidates, candidate_type_weights=weights)
                _sync(device)
                times.append((time.perf_counter() - start) * 1000)
        rows.append({
            "nodes": node_count,
            "candidate_edges": int(candidates.shape[1]),
            "median_forward_ms": statistics.median(times),
            "p95_forward_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
            "sampled_memory": sampler.result(),
        })
    return rows


def _paired_summary(cases: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    critical_by_df = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    rules = config["decision_rules"]
    summary: dict[str, Any] = {}
    for task in config["task"]["tasks"]:
        task_cases = [case for case in cases if case["task"] == task]
        graph_rows = [case["graph"]["test"] for case in task_cases]
        edge_f1 = [row["edge_f1"] for row in graph_rows if row["edge_f1"] is not None]
        edge_precision = [row["edge_precision"] for row in graph_rows if row["edge_precision"] is not None]
        edge_recall = [row["edge_recall"] for row in graph_rows if row["edge_recall"] is not None]
        query_change = [row["changed_query_pair_link_change_rate"] for row in graph_rows if row["changed_query_pair_link_change_rate"] is not None]
        quality: dict[str, Any] = {}
        for baseline in ("parameter_matched_mlp", "compute_matched_mlp"):
            deltas = [case["graph"]["test"]["mse"] - case[baseline]["test"]["mse"] for case in task_cases]
            relative_deltas = [
                delta / max(case[baseline]["test"]["mse"], 1e-12)
                for delta, case in zip(deltas, task_cases, strict=True)
            ]
            n = len(relative_deltas)
            relative_mean = statistics.mean(relative_deltas) if relative_deltas else None
            relative_std = statistics.stdev(relative_deltas) if n > 1 else None
            absolute_mean = statistics.mean(deltas) if deltas else None
            critical = critical_by_df.get(n - 1, 1.96)
            half_width = critical * relative_std / math.sqrt(n) if relative_std is not None else None
            upper = relative_mean + half_width if half_width is not None and relative_mean is not None else None
            minimum_relative = float(rules["minimum_relative_mse_improvement"])
            quality[baseline] = {
                "paired_mean_mse_delta_graph_minus_control": absolute_mean,
                "paired_mean_relative_mse_delta": relative_mean,
                "relative_sample_std": relative_std,
                "relative_delta_ci95": [relative_mean - half_width, upper] if half_width is not None and relative_mean is not None else None,
                "minimum_relative_improvement": minimum_relative,
                "criterion_met": bool(relative_mean is not None and upper is not None and relative_mean <= -minimum_relative and upper < 0),
            }
        random_deltas = [row["mse"] - row["random_matched_mse"] for row in graph_rows if row["random_matched_mse"] is not None]
        summary[task] = {
            "seed_count": len(task_cases),
            "graph_test_mse_mean": statistics.mean(row["mse"] for row in graph_rows) if graph_rows else None,
            "graph_minus_random_matched_mse_mean": statistics.mean(random_deltas) if random_deltas else None,
            "graph_test_edge_f1_mean": statistics.mean(edge_f1) if edge_f1 else None,
            "graph_test_edge_precision_mean": statistics.mean(edge_precision) if edge_precision else None,
            "graph_test_edge_recall_mean": statistics.mean(edge_recall) if edge_recall else None,
            "query_conditioned_link_change_rate_mean": statistics.mean(query_change) if query_change else None,
            "structure_criterion_met": bool(edge_f1 and edge_precision and edge_recall and query_change and
                statistics.mean(edge_precision) >= float(rules["minimum_edge_precision_and_recall"]) and
                statistics.mean(edge_recall) >= float(rules["minimum_edge_precision_and_recall"]) and
                statistics.mean(query_change) >= float(rules["minimum_query_conditioned_link_change_rate"])),
            "quality_comparisons": quality,
        }
    return summary


def run(config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(config["training"]["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("v4 config requires Apple MPS; refusing to fall back")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    torch.set_num_threads(int(config["training"]["threads"]))
    key_count = int(config["task"]["memory_neurons"])
    node_count = key_count + 1
    feature_count = 2 * key_count + 2
    torch.manual_seed(int(config["training"]["seeds"][0]))
    graph_probe = IdentityGraphModel(node_count, feature_count, int(config["model"]["hidden_size"])).to(device)
    sample_x, _, _ = _data(17, 1, key_count=key_count, task="single_link_retrieval", device=device)
    graph_params = sum(p.numel() for p in graph_probe.parameters())
    graph_macs = _linear_macs(graph_probe, sample_x)
    flat_input_size = node_count * feature_count
    param_width = _closest_width(graph_params, flat_input_size, parameter_count=True)
    mac_width = _closest_width(graph_macs, flat_input_size, parameter_count=False)
    param_probe = FlatControl(node_count, feature_count, param_width).to(device)
    mac_probe = FlatControl(node_count, feature_count, mac_width).to(device)
    costs = {
        "graph": {"parameters": graph_params, "linear_macs_per_example": graph_macs},
        "parameter_matched_mlp": {"hidden_size": param_width, "parameters": sum(p.numel() for p in param_probe.parameters()), "linear_macs_per_example": _linear_macs(param_probe, sample_x)},
        "compute_matched_mlp": {"hidden_size": mac_width, "parameters": sum(p.numel() for p in mac_probe.parameters()), "linear_macs_per_example": _linear_macs(mac_probe, sample_x)},
        "matching_rule": "nearest integer hidden width; report actual ratios; linear MAC proxy excludes scatter/reduction, nonlinear, optimizer and memory costs",
    }
    results = []
    for task_index, task in enumerate(config["task"]["tasks"]):
        for seed in config["training"]["seeds"]:
            torch.manual_seed(int(seed) + task_index * 10000)
            graph_model = IdentityGraphModel(node_count, feature_count, int(config["model"]["hidden_size"])).to(device)
            torch.manual_seed(int(seed) + task_index * 10000 + 1)
            param_model = FlatControl(node_count, feature_count, param_width).to(device)
            torch.manual_seed(int(seed) + task_index * 10000 + 2)
            mac_model = FlatControl(node_count, feature_count, mac_width).to(device)
            graph_result = _train(graph_model, task, int(seed), config, device)
            param_result = _train(param_model, task, int(seed), config, device)
            mac_result = _train(mac_model, task, int(seed), config, device)
            results.append({
                "task": task,
                "seed": int(seed),
                "graph": graph_result,
                "parameter_matched_mlp": param_result,
                "compute_matched_mlp": mac_result,
            })
    scale_model = IdentityGraphModel(node_count, feature_count, int(config["model"]["hidden_size"])).to(device)
    scaling = _scaling(scale_model, config, device)
    costs["parameter_count_relative_error"] = abs(costs["parameter_matched_mlp"]["parameters"] - graph_params) / graph_params
    costs["linear_mac_relative_error"] = abs(costs["compute_matched_mlp"]["linear_macs_per_example"] - graph_macs) / graph_macs
    costs["training_time_mean_seconds"] = {
        name: statistics.mean(case[name]["train_seconds"] for case in results)
        for name in ("graph", "parameter_matched_mlp", "compute_matched_mlp")
    }
    source_files = [
        "project/scripts/experiments/training/neurons/graph/run_neuron_identity_validation.py",
        "src/newera/neuron/graph/diagnostics/metrics.py",
        "src/newera/neuron/graph/policy.py",
        "src/newera/neuron/graph/layer.py",
        "src/newera/neuron/graph/topology.py",
        "src/newera/nlp_flow/model/components/architecture/foundation/config.py",
        "src/newera/nlp_flow/model/components/architecture/foundation/model.py",
        "pyproject.toml",
        "uv.lock",
    ]
    return {
        "schema_version": "newera-neuron-identity-validation-v4",
        "run_id": config["run_id"],
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
        "task_contract": config["task"],
        "decision_rules": config["decision_rules"],
        "cost_alignment": costs,
        "training_cases": results,
        "summary": _paired_summary(results, config),
        "scaling": scaling,
        "limitations": config["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "run_id": result["run_id"],
        "device": result["device"],
        "case_count": len(result["training_cases"]),
        "scaling_points": len(result["scaling"]),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
