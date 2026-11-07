"""Offline diagnostics for sparse, causal neuron graphs.

These functions intentionally operate on PyTorch tensors and the project's
edge-list topology. They do not assume a dense, symmetric adjacency matrix.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from newera.neuron.graph.topology import NeuronGraphTopology


def topology_metrics(
    topology: NeuronGraphTopology,
    *,
    edge_weights: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    max_edges: int | None = None,
) -> dict[str, float | int]:
    """Summarize edge scale and weighted in-degree for valid causal pairs."""
    device = topology.edge_index.device
    valid = torch.ones(
        (topology.batch_size, topology.nodes_per_graph), dtype=torch.bool, device=device
    )
    if attention_mask is not None:
        valid &= attention_mask.to(device=device, dtype=torch.bool)
    if segment_ids is not None:
        valid &= segment_ids.to(device=device).ge(0)
    edge_index = topology.edge_index
    source_batch = edge_index[0] // topology.nodes_per_graph
    source_local = edge_index[0] % topology.nodes_per_graph
    target_local = edge_index[1] % topology.nodes_per_graph
    keep = valid[source_batch, source_local] & valid[source_batch, target_local]
    if segment_ids is not None and edge_index.shape[1]:
        segments = segment_ids.to(device=device)
        keep &= segments[source_batch, source_local] == segments[source_batch, target_local]
    edge_index = edge_index[:, keep]
    strength = topology.strength if edge_weights is None else edge_weights
    weights = strength.to(device=device, dtype=torch.float32)[keep]

    valid_per_graph = valid.sum(dim=1).to(torch.float64)
    if segment_ids is None:
        possible_pairs = (valid_per_graph * (valid_per_graph - 1) / 2).sum()
    else:
        segments = segment_ids.to(device=device)
        possible_pairs = torch.zeros((), dtype=torch.float64, device=device)
        for batch_index in range(topology.batch_size):
            ids = segments[batch_index][valid[batch_index]]
            if ids.numel():
                _, counts = torch.unique(ids, return_counts=True)
                possible_pairs += (counts.to(torch.float64) * (counts - 1) / 2).sum()

    weighted_indegree = torch.zeros(
        topology.batch_size * topology.nodes_per_graph, dtype=torch.float32, device=device
    )
    if edge_index.shape[1]:
        weighted_indegree.index_add_(0, edge_index[1], weights)
    valid_degrees = weighted_indegree.view(topology.batch_size, topology.nodes_per_graph)[valid]
    if valid_degrees.numel():
        ordered_degrees = torch.sort(valid_degrees).values
        quantile_indices = torch.tensor(
            [
                int((ordered_degrees.numel() - 1) * 0.5),
                int((ordered_degrees.numel() - 1) * 0.95),
                int((ordered_degrees.numel() - 1) * 0.99),
            ],
            dtype=torch.long,
            device=device,
        )
        degree_quantiles = ordered_degrees.index_select(0, quantile_indices)
    else:
        degree_quantiles = torch.zeros(3, device=device)
    edge_count = int(edge_index.shape[1])
    denominator = max(float(possible_pairs.item()), 1.0)
    metrics: dict[str, float | int] = {
        "valid_node_count": int(valid.sum().item()),
        "edge_count": edge_count,
        "active_edge_count": int((weights > 0).sum().item()),
        "edge_density": edge_count / denominator,
        "mean_weighted_indegree": float(valid_degrees.mean().item()) if valid_degrees.numel() else 0.0,
        "weighted_indegree_p50": float(degree_quantiles[0].item()),
        "weighted_indegree_p95": float(degree_quantiles[1].item()),
        "weighted_indegree_p99": float(degree_quantiles[2].item()),
        "max_weighted_indegree": float(valid_degrees.max().item()) if valid_degrees.numel() else 0.0,
        "weighted_edge_mass": float(weights.sum().item()),
    }
    if max_edges is not None:
        if max_edges <= 0:
            raise ValueError("max_edges must be positive when supplied")
        metrics["edge_budget_utilization"] = edge_count / max_edges
    return metrics


def normalized_effective_rank(
    hidden_states: torch.Tensor, *, valid_mask: torch.Tensor | None = None, eps: float = 1e-12
) -> float:
    """Entropy effective rank, normalized by the maximum possible matrix rank."""
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have a node axis and a feature axis")
    matrix = hidden_states.reshape(-1, hidden_states.shape[-1]).detach().to(
        device="cpu", dtype=torch.float64
    )
    if valid_mask is not None:
        if valid_mask.numel() != matrix.shape[0]:
            raise ValueError("valid_mask must contain one entry per node")
        matrix = matrix[valid_mask.reshape(-1).to(device="cpu", dtype=torch.bool)]
    if matrix.numel() == 0:
        return 0.0
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if not torch.isfinite(total) or total <= eps:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float((entropy.exp() / min(matrix.shape)).clamp(0, 1).item())


def weighted_dirichlet_energy(
    hidden_states: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean weighted squared feature distance over directed edges."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if edge_index.shape[0] != 2 or edge_weights.shape != (edge_index.shape[1],):
        raise ValueError("edge_index and edge_weights have incompatible shapes")
    if edge_index.shape[1] == 0:
        return 0.0
    source = flat.index_select(0, edge_index[0])
    target = flat.index_select(0, edge_index[1])
    weights = edge_weights.to(dtype=source.dtype)
    numerator = (weights * (source - target).square().sum(dim=-1)).sum()
    return float((numerator / weights.sum().clamp_min(eps)).item())


def relative_dirichlet_energy(
    hidden_states: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Scale-normalized weighted edge variation in [0, 4] for nonnegative weights."""
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if edge_index.shape[0] != 2 or edge_weights.shape != (edge_index.shape[1],):
        raise ValueError("edge_index and edge_weights have incompatible shapes")
    if edge_index.shape[1] == 0:
        return 0.0
    source = flat.index_select(0, edge_index[0])
    target = flat.index_select(0, edge_index[1])
    weights = edge_weights.to(dtype=source.dtype)
    difference = (source - target).square().sum(dim=-1)
    node_energy = 0.5 * (source.square().sum(dim=-1) + target.square().sum(dim=-1))
    numerator = (weights * difference).sum()
    denominator = (weights * node_energy).sum()
    return float((numerator / denominator.clamp_min(eps)).item())


def action_distribution(logits: torch.Tensor) -> dict[str, float]:
    """Summarize proposal-policy uncertainty and predicted action frequencies."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [candidate_edges, actions]")
    if logits.shape[0] == 0:
        return {"action_entropy": 0.0, "action_confidence": 0.0, "action_count": 0.0}
    probabilities = logits.softmax(dim=-1)
    mean_probability = probabilities.mean(dim=0)
    entropy = -(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
    return {
        "action_entropy": float(entropy.item()),
        "action_confidence": float(probabilities.max(dim=-1).values.mean().item()),
        "action_count": float(logits.shape[0]),
        **{
            f"action_{index}_fraction": float(value.item())
            for index, value in enumerate(probabilities.argmax(dim=-1).bincount(minlength=logits.shape[-1]) / logits.shape[0])
        },
    }


def edge_set_jaccard(left: NeuronGraphTopology, right: NeuronGraphTopology) -> float:
    """Compare directed edge identity sets; reject ambiguous node remappings."""
    if (left.batch_size, left.nodes_per_graph) != (right.batch_size, right.nodes_per_graph):
        raise ValueError("topologies must have matching graph shapes")
    if left.node_ids is None or right.node_ids is None:
        raise ValueError("edge-set comparison requires stable node identities")
    if not torch.equal(left.node_ids.detach().cpu(), right.node_ids.detach().cpu()):
        raise ValueError("topologies must have identical node identities")
    node_count = left.batch_size * left.nodes_per_graph
    left_keys = set((left.edge_index[0] * node_count + left.edge_index[1]).detach().cpu().tolist())
    right_keys = set((right.edge_index[0] * node_count + right.edge_index[1]).detach().cpu().tolist())
    union = left_keys | right_keys
    return 1.0 if not union else len(left_keys & right_keys) / len(union)


def sampled_causal_gradient_transport(
    hidden_states: torch.Tensor,
    target_loss: torch.Tensor,
    *,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    nodes_per_graph: int,
    distance_bin_edges: tuple[int, ...] = (1, 4, 8, 16, 32, 64),
) -> dict[str, float | int]:
    """Measure target-loss gradient reaching sampled causal source positions."""
    if source_indices.shape != target_indices.shape or source_indices.ndim != 1:
        raise ValueError("source_indices and target_indices must be aligned vectors")
    if source_indices.numel() == 0:
        return {"sample_count": 0, "mean_gradient_norm": 0.0, "median_gradient_norm": 0.0}
    gradient = torch.autograd.grad(target_loss, hidden_states, retain_graph=True, allow_unused=True)[0]
    if gradient is None:
        norms = torch.zeros(source_indices.numel(), device=hidden_states.device)
    else:
        flat = gradient.reshape(-1, gradient.shape[-1])
        norms = flat.index_select(0, source_indices).norm(dim=-1)
    if nodes_per_graph <= 0:
        raise ValueError("samples must be causal and nodes_per_graph must be positive")
    source_batch, target_batch = source_indices // nodes_per_graph, target_indices // nodes_per_graph
    source_position, target_position = source_indices % nodes_per_graph, target_indices % nodes_per_graph
    distances = target_position - source_position
    if torch.any(source_batch != target_batch) or torch.any(distances < 0):
        raise ValueError("samples must be causal pairs within the same graph")
    if len(distance_bin_edges) < 2 or sorted(set(distance_bin_edges)) != list(distance_bin_edges):
        raise ValueError("distance_bin_edges must be strictly increasing integers")
    distance_bins: dict[str, dict[str, float | int]] = {}
    for lower, upper in pairwise(distance_bin_edges):
        in_bin = (distances >= lower) & (distances < upper)
        bin_norms = norms[in_bin]
        distance_bins[f"{lower}-{upper - 1}"] = {
            "sample_count": int(bin_norms.numel()),
            "mean_gradient_norm": float(bin_norms.mean().item()) if bin_norms.numel() else 0.0,
            "median_gradient_norm": float(bin_norms.median().item()) if bin_norms.numel() else 0.0,
        }
    if distance_bin_edges[-1] <= int(distances.max().item()):
        in_bin = distances >= distance_bin_edges[-1]
        bin_norms = norms[in_bin]
        distance_bins[f"{distance_bin_edges[-1]}+"] = {
            "sample_count": int(bin_norms.numel()),
            "mean_gradient_norm": float(bin_norms.mean().item()) if bin_norms.numel() else 0.0,
            "median_gradient_norm": float(bin_norms.median().item()) if bin_norms.numel() else 0.0,
        }
    return {
        "sample_count": int(norms.numel()),
        "mean_gradient_norm": float(norms.mean().item()),
        "median_gradient_norm": float(norms.median().item()),
        "mean_causal_distance": float(distances.float().mean().item()),
        "gradient_norm_by_distance": distance_bins,
    }


def rewire_topology_matched(
    topology: NeuronGraphTopology,
    *,
    segment_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    seed: int,
) -> NeuronGraphTopology:
    """Rewire sources while preserving each target's degree, edge types/weights,
    valid segment, and strict causal direction.
    """
    expected_shape = (topology.batch_size, topology.nodes_per_graph)
    if segment_ids.shape != expected_shape:
        raise ValueError("segment_ids must match topology graph shape")
    if attention_mask is not None and attention_mask.shape != expected_shape:
        raise ValueError("attention_mask must match topology graph shape")
    segments = segment_ids.detach().to(device="cpu", dtype=torch.long)
    valid = segments.ge(0)
    if attention_mask is not None:
        valid &= attention_mask.detach().to(device="cpu", dtype=torch.bool)
    edges = topology.edge_index.detach().cpu()
    types = topology.edge_type.detach().cpu()
    strengths = topology.strength.detach().cpu()
    nodes_per_graph = topology.nodes_per_graph
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grouped: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for index, (source, target) in enumerate(edges.t().tolist()):
        batch_index, source_local = divmod(source, nodes_per_graph)
        target_batch, target_local = divmod(target, nodes_per_graph)
        if batch_index != target_batch or source_local >= target_local:
            raise ValueError("rewiring requires a within-graph causal topology")
        if not valid[batch_index, source_local] or not valid[batch_index, target_local]:
            raise ValueError("topology contains an invalid source or target node")
        if segments[batch_index, source_local] != segments[batch_index, target_local]:
            raise ValueError("topology contains a cross-segment edge")
        grouped.setdefault((batch_index, target_local), []).append(
            (int(types[index]), float(strengths[index]), source_local)
        )

    rewired: list[tuple[int, int, int, float]] = []
    for (batch_index, target_local), edge_attributes in grouped.items():
        eligible = [
            source_local
            for source_local in range(target_local)
            if valid[batch_index, source_local]
            and segments[batch_index, source_local] == segments[batch_index, target_local]
        ]
        if len(eligible) < len(edge_attributes):
            raise ValueError("cannot preserve target degree with available causal in-segment sources")
        order = torch.randperm(len(eligible), generator=generator)[: len(edge_attributes)].tolist()
        for (edge_type, strength, _), selected_index in zip(edge_attributes, order, strict=True):
            source = batch_index * nodes_per_graph + eligible[selected_index]
            target = batch_index * nodes_per_graph + target_local
            rewired.append((target, source, edge_type, strength))
    rewired.sort()
    device = topology.edge_index.device
    edge_index = torch.tensor(
        [[source for _, source, _, _ in rewired], [target for target, _, _, _ in rewired]],
        dtype=torch.long,
        device=device,
    ).reshape(2, -1)
    edge_type = torch.tensor([edge_type for _, _, edge_type, _ in rewired], dtype=torch.long, device=device)
    strength = torch.tensor([weight for _, _, _, weight in rewired], dtype=topology.strength.dtype, device=device)
    return NeuronGraphTopology(
        topology.batch_size,
        topology.nodes_per_graph,
        edge_index,
        edge_type,
        strength,
        topology.node_ids,
    )


def drop_topology_edges(
    topology: NeuronGraphTopology, *, fraction: float, seed: int
) -> NeuronGraphTopology:
    """Deterministically remove a fixed fraction of edges without changing survivors."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    drop_count = round(topology.edge_count * fraction)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keep = torch.ones(topology.edge_count, dtype=torch.bool)
    if drop_count:
        keep[torch.randperm(topology.edge_count, generator=generator)[:drop_count]] = False
    keep = keep.to(device=topology.edge_index.device)
    return NeuronGraphTopology(
        topology.batch_size,
        topology.nodes_per_graph,
        topology.edge_index[:, keep],
        topology.edge_type[keep],
        topology.strength[keep],
        topology.node_ids,
    )


__all__ = [
    "action_distribution",
    "drop_topology_edges",
    "edge_set_jaccard",
    "normalized_effective_rank",
    "relative_dirichlet_energy",
    "rewire_topology_matched",
    "sampled_causal_gradient_transport",
    "topology_metrics",
    "weighted_dirichlet_energy",
]
