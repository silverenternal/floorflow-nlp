"""Sparse message passing across variable-degree neuron connections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from newera.neuron.graph.topology import NeuronGraphTopology


class _DeterministicGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.row_count = values.shape[0]
        return values.index_select(0, indices)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        if indices.numel() == 0:
            return torch.zeros(
                (ctx.row_count, *gradient.shape[1:]),
                dtype=gradient.dtype,
                device=gradient.device,
            ), None
        order = torch.argsort(indices, stable=True)
        sorted_indices = indices.index_select(0, order)
        sorted_gradient = gradient.index_select(0, order)
        unique_indices, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
        groups = torch.split(sorted_gradient, counts.detach().cpu().tolist(), dim=0)
        group_sums = [group.sum(dim=0) for group in groups]
        parts = []
        cursor = 0
        trailing_shape = gradient.shape[1:]
        zero_row = torch.zeros(trailing_shape, dtype=gradient.dtype, device=gradient.device)
        for index, group_sum in zip(unique_indices.detach().cpu().tolist(), group_sums, strict=True):
            if index > cursor:
                parts.append(zero_row.expand(index - cursor, *trailing_shape))
            parts.append(group_sum.unsqueeze(0))
            cursor = index + 1
        if cursor < ctx.row_count:
            parts.append(zero_row.expand(ctx.row_count - cursor, *trailing_shape))
        return torch.cat(parts, dim=0), None


def gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if torch.are_deterministic_algorithms_enabled():
        return _DeterministicGather.apply(values, indices)
    return values.index_select(0, indices)


@dataclass(frozen=True, slots=True)
class NeuronGraphOutput:
    hidden_states: torch.Tensor
    diagnostics: dict[str, torch.Tensor | int]


class DynamicNeuronGraphLayer(nn.Module):
    """Shared message-passing operator over a caller-owned sparse directed graph.

    Connections are edge-list entries, not fixed input slots. Messages are
    reduced by destination with ``index_add`` and a learned neuron update.
    Each graph node is a feature vector; this layer does not instantiate one
    independently parameterized model per neuron. It is a candidate connection
    mechanism for such models, not the modelized-neuron implementation itself.
    """

    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int = 4,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_edge_types <= 0:
            raise ValueError("hidden_size and num_edge_types must be positive")
        self.hidden_size = hidden_size
        self.num_edge_types = num_edge_types
        self.causal = causal
        rank = min(8, hidden_size)
        self.message_projection = nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False),
            nn.Linear(rank, hidden_size, bias=False),
        )
        self.edge_type_embedding = nn.Embedding(num_edge_types, rank)
        self.edge_type_gain = nn.Parameter(torch.ones(num_edge_types, rank))
        self.relation_projection = nn.Linear(rank, hidden_size, bias=False)
        self.edge_strength_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, 1),
        )
        nn.init.normal_(self.edge_strength_gate[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.edge_strength_gate[-1].bias)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False),
            nn.Linear(rank, hidden_size, bias=False),
        )
        self.degree_projection = nn.Linear(1, hidden_size, bias=False)
        self.update_candidate = nn.Sequential(
            nn.Linear(hidden_size * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, hidden_size),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, rank),
            nn.Tanh(),
            nn.Linear(rank, hidden_size),
        )
        self.residual_gate = nn.Parameter(torch.tensor(1.0e-2))

    def forward(
        self,
        nodes: torch.Tensor,
        topology: NeuronGraphTopology,
        *,
        candidate_edges: torch.Tensor | None = None,
        candidate_type_weights: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> NeuronGraphOutput:
        if nodes.ndim != 3 or nodes.shape[-1] != self.hidden_size:
            raise ValueError("nodes must have shape [batch, neurons, hidden_size]")
        batch_size, nodes_per_graph, _ = nodes.shape
        topology.validate(
            batch_size=batch_size,
            nodes_per_graph=nodes_per_graph,
            num_edge_types=self.num_edge_types,
            device=nodes.device,
            causal=self.causal,
            check_values=False,
        )
        topology = topology.filter_context(
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            num_edge_types=self.num_edge_types,
            causal=self.causal,
        )
        valid_topology_edges = (
            (topology.edge_type >= 0)
            & (topology.edge_type < self.num_edge_types)
            & torch.isfinite(topology.strength)
            & (topology.strength >= 0)
        )
        topology = NeuronGraphTopology(
            topology.batch_size,
            topology.nodes_per_graph,
            topology.edge_index[:, valid_topology_edges],
            topology.edge_type[valid_topology_edges],
            topology.strength[valid_topology_edges],
            topology.node_ids,
        )
        if (candidate_edges is None) != (candidate_type_weights is None):
            raise ValueError("candidate_edges and candidate_type_weights must be supplied together")
        if candidate_edges is not None and candidate_type_weights is not None:
            if candidate_edges.dtype != torch.long or candidate_edges.ndim != 2 or candidate_edges.shape[0] != 2:
                raise ValueError("candidate_edges must be a long tensor with shape [2, candidates]")
            if candidate_edges.device != nodes.device:
                raise ValueError("candidate_edges must be on the node tensor's device")
            if candidate_type_weights.shape != (candidate_edges.shape[1], self.num_edge_types):
                raise ValueError("candidate_type_weights must have shape [candidates, edge_types]")
            keep_candidates = torch.ones(candidate_edges.shape[1], dtype=torch.bool, device=nodes.device)
            if attention_mask is not None:
                if attention_mask.shape != (batch_size, nodes_per_graph):
                    raise ValueError("attention_mask must match the node batch/sequence")
                valid = attention_mask.to(torch.bool)
                source_batch = candidate_edges[0] // nodes_per_graph
                keep_candidates &= valid[source_batch, candidate_edges[0] % nodes_per_graph]
                keep_candidates &= valid[source_batch, candidate_edges[1] % nodes_per_graph]
            if segment_ids is not None:
                if segment_ids.shape != (batch_size, nodes_per_graph):
                    raise ValueError("segment_ids must match the node batch/sequence")
                source_batch = candidate_edges[0] // nodes_per_graph
                keep_candidates &= segment_ids[source_batch, candidate_edges[0] % nodes_per_graph] == segment_ids[
                    source_batch, candidate_edges[1] % nodes_per_graph
                ]
            candidate_indices = keep_candidates.nonzero(as_tuple=True)[0]
            candidate_edges = candidate_edges.index_select(1, candidate_indices)
            candidate_type_weights = gather_rows(candidate_type_weights, candidate_indices)
        flat = nodes.reshape(-1, self.hidden_size)
        messages = torch.zeros_like(flat)
        degree = flat.new_zeros((flat.shape[0], 1))
        source_ids, target_ids = topology.edge_index
        edge_types = topology.edge_type
        edge_weights = topology.strength.to(dtype=nodes.dtype)
        if candidate_edges is not None and candidate_type_weights is not None:
            node_count = batch_size * nodes_per_graph
            candidate_keys = candidate_edges[0] * node_count + candidate_edges[1]
            topology_keys = source_ids * node_count + target_ids
            keep_static = ~torch.isin(topology_keys, candidate_keys)
            source_ids = torch.cat((source_ids[keep_static], candidate_edges[0].repeat_interleave(self.num_edge_types)))
            target_ids = torch.cat((target_ids[keep_static], candidate_edges[1].repeat_interleave(self.num_edge_types)))
            edge_types = torch.cat(
                (
                    edge_types[keep_static],
                    torch.arange(self.num_edge_types, device=nodes.device).repeat(candidate_edges.shape[1]),
                )
            )
            edge_weights = torch.cat((
                edge_weights[keep_static],
                candidate_type_weights.reshape(-1).to(dtype=nodes.dtype),
            ))
        if source_ids.numel():
            source = gather_rows(flat, source_ids)
            target = gather_rows(flat, target_ids)
            relation = gather_rows(self.edge_type_embedding.weight, edge_types)
            strength_gate = torch.sigmoid(self.edge_strength_gate(torch.cat((source, target), dim=-1)))
            edge_strength = edge_weights.unsqueeze(-1)
            edge_type_gain = gather_rows(self.edge_type_gain, edge_types)
            edge_messages = self.message_projection(source)
            edge_messages = edge_messages + self.relation_projection(relation * edge_type_gain)
            edge_messages = edge_messages * edge_strength * strength_gate
            if torch.are_deterministic_algorithms_enabled():
                message_rows = []
                degree_rows = []
                for node_index in range(flat.shape[0]):
                    incoming = target_ids == node_index
                    edge_ids = incoming.nonzero(as_tuple=True)[0]
                    message_rows.append(gather_rows(edge_messages, edge_ids).sum(dim=0))
                    degree_rows.append(gather_rows(edge_strength, edge_ids).sum(dim=0))
                messages = torch.stack(message_rows)
                degree = torch.stack(degree_rows)
            else:
                messages.index_add_(0, target_ids, edge_messages)
                degree.index_add_(0, target_ids, edge_strength)
        normalized_messages = messages / degree.clamp_min(1.0).sqrt()
        degree_signal = self.degree_projection(torch.log1p(degree))
        drive = self.input_projection(flat) + normalized_messages + degree_signal
        update_features = torch.cat((flat, drive), dim=-1)
        candidate = self.update_candidate(update_features)
        update_gate = torch.sigmoid(self.update_gate(update_features))
        updated = flat + update_gate * (candidate - flat)
        output = flat + self.residual_gate * (updated - flat)
        output = output.view(batch_size, nodes_per_graph, self.hidden_size)
        degree_values = degree.squeeze(-1)
        valid_nodes = torch.ones(
            (batch_size, nodes_per_graph), dtype=torch.bool, device=nodes.device
        )
        if attention_mask is not None:
            valid_nodes &= attention_mask.to(device=nodes.device, dtype=torch.bool)
        if segment_ids is not None:
            valid_nodes &= segment_ids.to(device=nodes.device).ge(0)
        valid_degree_values = degree_values[valid_nodes.reshape(-1)]
        degree_quantiles = (
            torch.sort(valid_degree_values).values.index_select(
                0,
                torch.tensor(
                    [
                        int((valid_degree_values.numel() - 1) * 0.5),
                        int((valid_degree_values.numel() - 1) * 0.95),
                        int((valid_degree_values.numel() - 1) * 0.99),
                    ],
                    dtype=torch.long,
                    device=degree_values.device,
                ),
            )
            if valid_degree_values.numel()
            else degree_values.new_zeros(3)
        )
        diagnostics: dict[str, torch.Tensor | int] = {
            "edge_count": topology.edge_count,
            "valid_node_count": int(valid_degree_values.numel()),
            "mean_weighted_indegree": valid_degree_values.mean().detach() if valid_degree_values.numel() else 0.0,
            "max_weighted_indegree": valid_degree_values.max().detach() if valid_degree_values.numel() else 0.0,
            "weighted_indegree_p50": degree_quantiles[0].detach(),
            "weighted_indegree_p95": degree_quantiles[1].detach(),
            "weighted_indegree_p99": degree_quantiles[2].detach(),
            "soft_candidate_count": 0 if candidate_edges is None else candidate_edges.shape[1],
            "scored_edge_count": int(source_ids.numel()),
            "active_scored_edge_count": int((edge_weights > 0).sum().item()),
            "weighted_edge_mass": degree.sum().detach(),
            "residual_gate": self.residual_gate.detach(),
        }
        return NeuronGraphOutput(output, diagnostics)


__all__ = ["DynamicNeuronGraphLayer", "NeuronGraphOutput", "gather_rows"]
