"""Screen task quality and candidate-search cost for the sparse neuron graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self

import psutil
import torch
import torch.nn.functional as F

from newera.neuron.graph import (
    NeuronGraphTopology,
    action_distribution,
    drop_topology_edges,
    normalized_effective_rank,
    relative_dirichlet_energy,
    rewire_topology_matched,
    sampled_causal_gradient_transport,
    topology_metrics,
    weighted_dirichlet_energy,
)
from newera.nlp_flow.components.research.training.objectives.training_losses import batch_loss
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "project/configs/models_training/learning/neuron_comparisons/graph/graph_validation_matched_controls.json"


class _RSSSampler:
    """Sample this process's resident set size while a model phase is running."""

    def __init__(self, interval_seconds: float = 0.002) -> None:
        if interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.start_rss_bytes = 0
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> Self:
        self.start_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)

    def as_dict(self) -> dict[str, int]:
        return {
            "start_rss_bytes": self.start_rss_bytes,
            "sampled_peak_rss_bytes": self.peak_rss_bytes,
            "sampled_peak_delta_bytes": max(self.peak_rss_bytes - self.start_rss_bytes, 0),
            "sampling_interval_microseconds": int(self.interval_seconds * 1_000_000),
        }


class _MPSMemorySampler:
    """Sample PyTorch MPS and Metal driver live allocation counters."""

    def __init__(self, interval_seconds: float = 0.002) -> None:
        if interval_seconds <= 0:
            raise ValueError("MPS memory sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.start = {
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
        self.peak = dict(self.start)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            current = {
                "allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            }
            for key, value in current.items():
                self.peak[key] = max(self.peak[key], value)
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> Self:
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        current = {
            "allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        }
        for key, value in current.items():
            self.peak[key] = max(self.peak[key], value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampling_interval_microseconds": int(self.interval_seconds * 1_000_000),
            "start": self.start,
            "sampled_peak": self.peak,
            "sampled_peak_delta_bytes": {
                key: value - self.start[key] for key, value in self.peak.items()
            },
        }


def _data(seed: int, *, count: int, length: int, vocab_size: int) -> torch.Tensor:
    return _data_and_segments(seed, count=count, length=length, vocab_size=vocab_size, segment_count=1)[0]


def _data_and_segments(
    seed: int, *, count: int, length: int, vocab_size: int, segment_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment_count <= 0 or length % segment_count:
        raise ValueError("segment_count must be positive and divide sequence length")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.empty((count, length), dtype=torch.long)
    segment_length = length // segment_count
    segment_ids = torch.arange(length, dtype=torch.long).div(segment_length, rounding_mode="floor")
    segment_ids = segment_ids.unsqueeze(0).expand(count, -1).clone()
    for start in range(0, length, segment_length):
        values[:, start : start + 2] = torch.randint(vocab_size, (count, 2), generator=generator)
        for index in range(start + 2, start + segment_length):
            values[:, index] = (values[:, index - 1] + values[:, index - 2]) % vocab_size
    return values, segment_ids


def _config(
    *,
    graph: bool,
    sequence_length: int,
    candidate_budget: int | None = None,
    batch_size: int = 64,
    intermediate_size: int = 64,
) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        max_seq_len=sequence_length,
        parameter_cap=2_000_000,
        enable_neuron_graph=graph,
        neuron_graph_candidate_budget=candidate_budget,
        neuron_graph_max_edges=(sequence_length + candidate_budget) * batch_size
        if candidate_budget is not None
        else None,
    )


def _metric(
    model: CompactNLPDecoder,
    values: torch.Tensor,
    *,
    topology: NeuronGraphTopology | None = None,
    segment_ids: torch.Tensor | None = None,
    disable_graph_message: bool = False,
) -> tuple[float, float, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        saved_gate = None
        if disable_graph_message:
            if model.neuron_graph is None:
                raise ValueError("graph-message ablation requires a graph-enabled model")
            saved_gate = model.neuron_graph.residual_gate.detach().clone()
            model.neuron_graph.residual_gate.zero_()
        node_ids = None
        if topology is not None:
            node_ids = topology.node_ids
            if node_ids is None:
                raise ValueError("custom topology evaluation requires stable node_ids")
        node_ids = (
            torch.arange(values.numel(), dtype=torch.long, device=values.device).reshape_as(values)
            if model.neuron_graph is not None
            else None
        )
        representations: dict[str, torch.Tensor] = {}
        hook = None
        if model.neuron_graph is not None:
            def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], result: Any) -> None:
                representations["graph_input"] = args[0]
                representations["graph_output"] = result.hidden_states

            hook = model.neuron_graph.register_forward_hook(capture)
        try:
            output = model(
                values,
                neuron_topology=topology,
                update_neuron_topology=(topology is None and model.neuron_graph is not None),
                segment_ids=segment_ids,
                neuron_candidate_edges=(
                    torch.empty((2, 0), dtype=torch.long, device=values.device)
                    if topology is not None
                    else None
                ),
                neuron_node_ids=topology.node_ids if topology is not None else node_ids,
            )
        finally:
            if hook is not None:
                hook.remove()
            if saved_gate is not None:
                model.neuron_graph.residual_gate.copy_(saved_gate)
        logits = output.token_logits[:, :-1]
        labels = values[:, 1:]
        valid_targets = torch.ones_like(labels, dtype=torch.bool)
        if segment_ids is not None:
            valid_targets &= segment_ids[:, :-1].eq(segment_ids[:, 1:])
            valid_targets &= segment_ids[:, 1:].ge(0)
        loss = F.cross_entropy(logits[valid_targets], labels[valid_targets])
        accuracy = (logits.argmax(dim=-1)[valid_targets] == labels[valid_targets]).float().mean()
        representations["final_decoder"] = output.hidden_states
        representation_diagnostics = {
            name: {"normalized_effective_rank": normalized_effective_rank(states)}
            for name, states in representations.items()
        }
        diagnostics: dict[str, Any] = {
            "normalized_effective_rank": normalized_effective_rank(output.hidden_states),
            "representation_diagnostics": representation_diagnostics,
        }
        if output.neuron_topology is not None:
            diagnostics["_topology"] = output.neuron_topology
            diagnostics["topology"] = topology_metrics(
                output.neuron_topology, segment_ids=segment_ids
            )
            for name, states in representations.items():
                representation_diagnostics[name]["weighted_dirichlet_energy"] = weighted_dirichlet_energy(
                    states,
                    output.neuron_topology.edge_index,
                    output.neuron_topology.strength,
                )
                representation_diagnostics[name]["relative_dirichlet_energy"] = relative_dirichlet_energy(
                    states,
                    output.neuron_topology.edge_index,
                    output.neuron_topology.strength,
                )
            diagnostics["dirichlet_energy"] = representation_diagnostics["final_decoder"][
                "weighted_dirichlet_energy"
            ]
        if output.neuron_link_logits is not None:
            actions = action_distribution(output.neuron_link_logits)
            if output.neuron_candidate_edges is not None:
                actions["candidate_actions"] = {
                    f"{source}:{target}": int(action)
                    for (source, target), action in zip(
                        output.neuron_candidate_edges.t().detach().cpu().tolist(),
                        output.neuron_link_logits.argmax(dim=-1).detach().cpu().tolist(),
                        strict=True,
                    )
                }
            diagnostics["link_actions"] = actions
        if output.neuron_graph_diagnostics is not None:
            diagnostics["graph_layer"] = {
                key: float(value.item()) if isinstance(value, torch.Tensor) else value
                for key, value in output.neuron_graph_diagnostics.items()
            }
    return float(loss), float(accuracy), diagnostics


def _policy_action_stability(
    earlier: dict[str, Any], later: dict[str, Any]
) -> dict[str, float | int]:
    early_actions = earlier.get("link_actions", {}).get("candidate_actions", {})
    late_actions = later.get("link_actions", {}).get("candidate_actions", {})
    shared = set(early_actions) & set(late_actions)
    return {
        "shared_candidate_count": len(shared),
        "action_agreement": (
            sum(early_actions[key] == late_actions[key] for key in shared) / len(shared)
            if shared
            else 1.0
        ),
    }


def _serializable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    result = dict(diagnostics)
    result.pop("_topology", None)
    if "link_actions" in result:
        result["link_actions"] = {
            key: value
            for key, value in result["link_actions"].items()
            if key != "candidate_actions"
        }
    return result


def _graph_gradient_diagnostic(
    model: CompactNLPDecoder, values: torch.Tensor, segment_ids: torch.Tensor
) -> dict[str, Any]:
    if model.neuron_graph is None:
        return {"sample_count": 0, "mean_gradient_norm": 0.0, "median_gradient_norm": 0.0}
    captured: dict[str, torch.Tensor] = {}

    def capture_layer_input(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["nodes"] = args[0]

    handle = model.neuron_graph.register_forward_pre_hook(capture_layer_input)
    model.eval()
    try:
        node_ids = torch.arange(values.numel(), dtype=torch.long, device=values.device).reshape_as(values)
        output = model(values, segment_ids=segment_ids, neuron_node_ids=node_ids)
        nodes = captured["nodes"]
        supervised_position = values.shape[1] - 1
        target_position = supervised_position - 1
        target_loss = F.cross_entropy(
            output.token_logits[0, target_position], values[0, supervised_position]
        )
        source_positions = (
            segment_ids[0, :target_position]
            .eq(segment_ids[0, target_position])
            .nonzero(as_tuple=True)[0]
        )
        sources = source_positions
        targets = torch.full_like(sources, target_position)
        return sampled_causal_gradient_transport(
            nodes,
            target_loss,
            source_indices=sources,
            target_indices=targets,
            nodes_per_graph=values.shape[1],
        )
    finally:
        handle.remove()


def _train_case(
    *,
    seed: int,
    steps: int,
    candidate_budget: int | None,
    segment_count: int,
    parameter_matched_intermediate_size: int,
    topology_perturbations: dict[str, Any],
    memory_sampling_interval_microseconds: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    training, training_segments = _data_and_segments(
        seed + 100, count=32, length=32, vocab_size=32, segment_count=segment_count
    )
    validation, validation_segments = _data_and_segments(
        seed + 200, count=64, length=32, vocab_size=32, segment_count=segment_count
    )
    training_digest = hashlib.sha256(
        training.numpy().tobytes() + training_segments.numpy().tobytes()
    ).hexdigest()
    validation_digest = hashlib.sha256(
        validation.numpy().tobytes() + validation_segments.numpy().tobytes()
    ).hexdigest()
    control = CompactNLPDecoder(_config(graph=False, sequence_length=32))
    matched_control = CompactNLPDecoder(
        _config(
            graph=False,
            sequence_length=32,
            intermediate_size=parameter_matched_intermediate_size,
        )
    )
    graph = CompactNLPDecoder(
        _config(graph=True, sequence_length=32, candidate_budget=candidate_budget)
    )
    if sum(parameter.numel() for parameter in matched_control.parameters()) != sum(
        parameter.numel() for parameter in graph.parameters()
    ):
        raise ValueError("parameter-matched control does not match graph model parameter count")
    shared_state = {
        key: value
        for key, value in control.state_dict().items()
        if key in graph.state_dict() and graph.state_dict()[key].shape == value.shape
    }
    graph.load_state_dict(shared_state, strict=False)
    matched_shared_state = {
        key: value
        for key, value in shared_state.items()
        if key in matched_control.state_dict()
        and matched_control.state_dict()[key].shape == value.shape
    }
    matched_control.load_state_dict(matched_shared_state, strict=False)
    models = (
        ("transformer_control", control),
        ("parameter_matched_control", matched_control),
        (f"graph_budget_{candidate_budget}", graph),
    )
    rows = []
    for name, model in models:
        initial_val_loss, initial_val_accuracy, initial_diagnostics = _metric(
            model, validation, segment_ids=validation_segments
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        with _RSSSampler(memory_sampling_interval_microseconds / 1_000_000) as rss_sampler:
            started = time.perf_counter()
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss, *_ = batch_loss(
                    model,
                    {
                        "input_ids": training,
                        "labels": training,
                        "segment_ids": training_segments,
                    },
                    device=torch.device("cpu"),
                    autocast=nullcontext(),
                )
                loss.backward()
                optimizer.step()
            train_seconds = time.perf_counter() - started
        training_memory = rss_sampler.as_dict()
        validation_loss, validation_accuracy, validation_diagnostics = _metric(
            model, validation, segment_ids=validation_segments
        )
        gradient_transport = (
            _graph_gradient_diagnostic(model, validation[:1], validation_segments[:1])
            if model.neuron_graph
            else None
        )
        ablations = {}
        if model.neuron_graph is not None:
            trained_topology = validation_diagnostics["_topology"]
            perturbations = {}
            if topology_perturbations["matched_rewire"]:
                perturbations["matched_rewire"] = rewire_topology_matched(
                    trained_topology,
                    segment_ids=validation_segments,
                    seed=seed,
                )
            for fraction in topology_perturbations["drop_fractions"]:
                label = round(float(fraction) * 100)
                perturbations[f"drop_{label}_percent"] = drop_topology_edges(
                    trained_topology, fraction=float(fraction), seed=seed
                )
            no_message_loss, no_message_accuracy, _ = _metric(
                model,
                validation,
                segment_ids=validation_segments,
                disable_graph_message=True,
            )
            ablations = {
                "no_graph_message_validation_cross_entropy": no_message_loss,
                "no_graph_message_validation_next_token_accuracy": no_message_accuracy,
            }
            for perturbation_name, perturbed_topology in perturbations.items():
                perturbation_loss, perturbation_accuracy, perturbation_metrics = _metric(
                    model,
                    validation,
                    topology=perturbed_topology,
                    segment_ids=validation_segments,
                )
                ablations[f"{perturbation_name}_validation_cross_entropy"] = perturbation_loss
                ablations[f"{perturbation_name}_validation_next_token_accuracy"] = perturbation_accuracy
                ablations[f"{perturbation_name}_metrics"] = perturbation_metrics["topology"]
        rows.append(
            {
                "seed": seed,
                "condition": name,
                "steps": steps,
                "initial_validation_cross_entropy": initial_val_loss,
                "initial_validation_next_token_accuracy": initial_val_accuracy,
                "validation_cross_entropy": validation_loss,
                "validation_next_token_accuracy": validation_accuracy,
                "initial_diagnostics": _serializable_diagnostics(initial_diagnostics),
                "validation_diagnostics": _serializable_diagnostics(validation_diagnostics),
                "sampled_causal_gradient_transport": gradient_transport,
                "policy_action_stability": _policy_action_stability(
                    initial_diagnostics, validation_diagnostics
                )
                if model.neuron_graph is not None
                else None,
                **ablations,
                "train_wall_seconds": train_seconds,
                "training_memory": training_memory,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "parameter_bytes": sum(
                    parameter.numel() * parameter.element_size() for parameter in model.parameters()
                ),
                "training_data_sha256": training_digest,
                "validation_data_sha256": validation_digest,
            }
        )
    return {row["condition"]: row for row in rows}


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_device(device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the v3 config requests MPS but Apple MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")


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
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {}


def _candidate_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    benchmark = config["search_benchmark"]
    device = torch.device(benchmark["device"])
    _validate_device(device)
    for length in benchmark["sequence_lengths"]:
        for budget in benchmark["candidate_budgets"]:
            def call(sequence_length: int = length, candidate_budget: int = budget):
                return NeuronGraphTopology.causal_candidates(
                    1,
                    sequence_length,
                    device=device,
                    max_candidates_per_graph=candidate_budget,
                    randomize=False,
                )
            for _ in range(benchmark["warmup_calls"]):
                call()
            _synchronize(device)
            samples = []
            candidate_count = 0
            for _ in range(benchmark["measured_calls"]):
                start = time.perf_counter()
                candidates = call()
                _synchronize(device)
                samples.append((time.perf_counter() - start) * 1000.0)
                candidate_count = candidates.shape[1]
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as memory_profile:
                call()
                _synchronize(device)
            profiled_positive_allocations = sum(
                max(event.self_cpu_memory_usage, 0)
                for event in memory_profile.key_averages()
            )
            results.append(
                {
                    "sequence_length": length,
                    "device": str(device),
                    "budget": budget,
                    "candidate_count": candidate_count,
                    "median_search_ms": statistics.median(samples),
                    "candidate_index_bytes": candidate_count * 2 * torch.tensor([], dtype=torch.long).element_size(),
                    "profiled_positive_cpu_allocation_bytes": profiled_positive_allocations,
                    "device_memory_after_candidate": _device_memory(device),
                }
            )
    return results


def _model_latency(config: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = config["model_latency"]
    device = torch.device(benchmark["device"])
    _validate_device(device)
    rows = []
    for length in benchmark["sequence_lengths"]:
        values = _data(900 + length, count=benchmark["batch_size"], length=length, vocab_size=32).to(device)
        for condition, graph_enabled in (("transformer_control", False), ("graph", True)):
            model = CompactNLPDecoder(
                _config(
                    graph=graph_enabled,
                    sequence_length=length,
                    candidate_budget=benchmark["candidate_budget"] if graph_enabled else None,
                    batch_size=benchmark["batch_size"],
                )
            ).to(device).eval()
            _synchronize(device)
            memory_before = _device_memory(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(benchmark["warmup_calls"]):
                    model(values)
                _synchronize(device)
                samples = []
                for _ in range(benchmark["measured_calls"]):
                    _synchronize(device)
                    start = time.perf_counter()
                    output = model(values)
                    _synchronize(device)
                    samples.append((time.perf_counter() - start) * 1000.0)
            del output
            with _RSSSampler(
                config["memory_sampling_interval_microseconds"] / 1_000_000
            ) as rss_sampler:
                if device.type == "mps":
                    with _MPSMemorySampler(
                        config["memory_sampling_interval_microseconds"] / 1_000_000
                    ) as device_sampler:
                        with torch.no_grad():
                            live_output = model(values)
                            _synchronize(device)
                        memory_after = _device_memory(device)
                    sampled_device_memory = device_sampler.as_dict()
                else:
                    with torch.no_grad():
                        live_output = model(values)
                        _synchronize(device)
                    memory_after = _device_memory(device)
                    sampled_device_memory = {}
            host_memory = rss_sampler.as_dict()
            rows.append(
                {
                    "sequence_length": length,
                    "batch_size": benchmark["batch_size"],
                    "device": str(device),
                    "condition": condition,
                    "parameter_bytes": sum(
                        parameter.numel() * parameter.element_size() for parameter in model.parameters()
                    ),
                    "median_forward_ms": statistics.median(samples),
                    "p95_forward_ms": float(torch.tensor(samples).quantile(0.95).item()),
                    "tokens_per_second": length * benchmark["batch_size"] / (statistics.median(samples) / 1000.0),
                    "device_memory_before_forward": memory_before,
                    "device_memory_after_forward": memory_after,
                    "device_allocated_live_delta_bytes": memory_after.get(
                        "allocated_bytes", 0
                    ) - memory_before.get("allocated_bytes", 0),
                    "sampled_device_memory": sampled_device_memory,
                    "sampled_host_rss": host_memory,
                }
            )
            del live_output, model
            _synchronize(device)
        del values
    return rows


def run(*, steps: int | None = None) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    torch.set_num_threads(config["training"]["threads"])
    step_count = config["training"]["steps"] if steps is None else steps
    if step_count <= 0:
        raise ValueError("training steps must be positive")
    cases = []
    for seed in config["training"]["seeds"]:
        for budget in config["training"]["candidate_budgets"]:
            cases.append(
                _train_case(
                    seed=seed,
                    steps=step_count,
                    candidate_budget=budget,
                    segment_count=config["training"]["segment_count"],
                    parameter_matched_intermediate_size=config["training"][
                        "parameter_matched_intermediate_size"
                    ],
                    topology_perturbations=config["topology_perturbations"],
                    memory_sampling_interval_microseconds=config[
                        "memory_sampling_interval_microseconds"
                    ],
                )
            )
    code_files = (
        Path(__file__),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "src/newera/neuron/graph/topology.py",
        ROOT / "src/newera/neuron/graph/layer.py",
        ROOT / "src/newera/neuron/graph/diagnostics/metrics.py",
        ROOT / "src/newera/nlp_flow/model/components/architecture/foundation/model.py",
        ROOT / "src/newera/nlp_flow/components/research/training/objectives/training_losses.py",
    )
    code_digests = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in code_files
    }
    config_digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    paired_quality_deltas = []
    for case in cases:
        for key, row in case.items():
            if key.startswith("graph_budget_"):
                paired_quality_deltas.append(
                    {
                        "seed": row["seed"],
                        "condition": key,
                        "validation_cross_entropy_delta_vs_no_graph": row["validation_cross_entropy"]
                        - case["transformer_control"]["validation_cross_entropy"],
                        "validation_cross_entropy_delta_vs_parameter_matched": row[
                            "validation_cross_entropy"
                        ]
                        - case["parameter_matched_control"]["validation_cross_entropy"],
                    }
                )
    paired_quality_summary = []
    for condition in (f"graph_budget_{budget}" for budget in config["training"]["candidate_budgets"]):
        for baseline_name, delta_key in (
            ("transformer_control", "validation_cross_entropy_delta_vs_no_graph"),
            ("parameter_matched_control", "validation_cross_entropy_delta_vs_parameter_matched"),
        ):
            deltas = [
                row[delta_key]
                for row in paired_quality_deltas
                if row["condition"] == condition
            ]
            paired_quality_summary.append(
                {
                    "condition": condition,
                    "baseline": baseline_name,
                    "mean_validation_cross_entropy_delta": statistics.mean(deltas),
                    "sample_std_validation_cross_entropy_delta": statistics.stdev(deltas),
                    "seed_count": len(deltas),
                }
            )
    return {
        "schema_version": "newera-neuron-graph-validation-result-v3",
        "run_id": config["run_id"],
        "training_device": config["training"]["device"],
        "process_peak_rss_bytes": peak_rss,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "psutil": psutil.__version__,
        "platform": platform.platform(),
        "config_sha256": config_digest,
        "code_sha256": code_digests,
        "training_steps": step_count,
        "training_cases": cases,
        "paired_quality_deltas": paired_quality_deltas,
        "paired_quality_summary": paired_quality_summary,
        "candidate_search": _candidate_benchmark(config),
        "model_latency": _model_latency(config),
        "limitations": config["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="override training steps for a smoke run")
    parser.add_argument("--output", type=Path, help="write JSON results to this path")
    args = parser.parse_args()
    result = run(steps=args.steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
