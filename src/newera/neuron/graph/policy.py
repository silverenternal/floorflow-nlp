"""Trainable proposals for variable-degree neuron connectivity."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from newera.neuron.graph.layer import gather_rows
from newera.neuron.graph.topology import LinkMutation, NeuronGraphTopology


class LinkProposalPolicy(nn.Module):
    """Score any supplied candidate edge set without a top-k degree limit.

    Action classes are keep, connect/retype to each edge type, and remove. The
    score network is trainable with ``supervised_loss``; thresholding into
    discrete topology mutations is an explicit outer-loop operation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_edge_types: int,
        *,
        initial_keep_bias: float = 2.0,
        predict_strengths: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_edge_types <= 0:
            raise ValueError("hidden_size and num_edge_types must be positive")
        if isinstance(initial_keep_bias, bool) or not isinstance(initial_keep_bias, (int, float)) or not math.isfinite(initial_keep_bias):
            raise ValueError("initial_keep_bias must be finite")
        if not isinstance(predict_strengths, bool):
            raise TypeError("predict_strengths must be bool")
        self.num_edge_types = num_edge_types
        self.predict_strengths_enabled = predict_strengths
        rank = min(8, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, rank),
            nn.Tanh(),
            nn.Linear(rank, num_edge_types + 2),
        )
        final = self.scorer[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[0] = initial_keep_bias  # callers can tune prior strength for exploratory learning
        self.strength_scorer = (
            nn.Sequential(
                nn.Linear(hidden_size * 4, rank),
                nn.Tanh(),
                nn.Linear(rank, 1),
            )
            if predict_strengths
            else None
        )

    def _candidate_features(
        self, nodes: torch.Tensor, candidate_edges: torch.Tensor
    ) -> torch.Tensor:
        flattened = nodes.reshape(-1, nodes.shape[-1])
        source = gather_rows(flattened, candidate_edges[0])
        target = gather_rows(flattened, candidate_edges[1])
        return torch.cat((source, target, source * target, (source - target).abs()), dim=-1)

    def predict_strengths(
        self,
        nodes: torch.Tensor,
        candidate_edges: torch.Tensor,
        topology: NeuronGraphTopology,
        *,
        causal: bool = True,
        check_values: bool = True,
    ) -> torch.Tensor:
        if self.strength_scorer is None:
            raise RuntimeError("strength prediction is disabled for this policy")
        _validate_candidate_edges(nodes, candidate_edges, topology, self.num_edge_types, causal, check_values)
        return self.strength_scorer(self._candidate_features(nodes, candidate_edges)).sigmoid().squeeze(-1)

    def forward(
        self,
        nodes: torch.Tensor,
        candidate_edges: torch.Tensor,
        topology: NeuronGraphTopology,
        *,
        causal: bool = True,
        check_values: bool = True,
    ) -> torch.Tensor:
        _validate_candidate_edges(nodes, candidate_edges, topology, self.num_edge_types, causal, check_values)
        return self.scorer(self._candidate_features(nodes, candidate_edges))

    def expected_type_weights(
        self,
        logits: torch.Tensor,
        candidate_edges: torch.Tensor,
        topology: NeuronGraphTopology,
        *,
        candidate_strengths: torch.Tensor | None = None,
        straight_through: bool = False,
    ) -> torch.Tensor:
        """Return per-relation edge weights, optionally with hard forward actions.

        Straight-through mode uses the argmax action in the forward pass while
        retaining the softmax Jacobian for task-loss gradients. It lets training
        observe add/remove/retype decisions without action labels.
        """

        if logits.shape != (candidate_edges.shape[1], self.num_edge_types + 2):
            raise ValueError("logits must align with candidate_edges")
        probabilities = logits.softmax(dim=-1)
        if straight_through:
            hard_actions = torch.nn.functional.one_hot(
                probabilities.argmax(dim=-1), num_classes=probabilities.shape[-1]
            ).to(dtype=probabilities.dtype)
            probabilities = hard_actions - probabilities.detach() + probabilities
        if candidate_strengths is not None and (
            candidate_strengths.shape != (candidate_edges.shape[1],)
            or candidate_strengths.device != logits.device
            or not candidate_strengths.is_floating_point()
        ):
            raise ValueError("candidate_strengths must be a floating vector aligned with candidates")
        scale = candidate_strengths if candidate_strengths is not None else torch.ones_like(probabilities[:, 0])
        type_weights = probabilities[:, 1 : self.num_edge_types + 1] * scale.unsqueeze(-1)
        if topology.edge_count == 0 or candidate_edges.shape[1] == 0:
            return type_weights
        node_count = topology.batch_size * topology.nodes_per_graph
        candidate_keys = candidate_edges[0] * node_count + candidate_edges[1]
        topology_keys = topology.edge_index[0] * node_count + topology.edge_index[1]
        sorted_keys, order = torch.sort(topology_keys)
        slots = torch.searchsorted(sorted_keys, candidate_keys)
        safe_slots = slots.clamp(max=sorted_keys.numel() - 1)
        exists = (slots < sorted_keys.numel()) & (sorted_keys[safe_slots] == candidate_keys)
        existing_types = topology.edge_type[order[safe_slots]]
        existing_strengths = topology.strength[order[safe_slots]].to(dtype=logits.dtype)
        keep_scale = torch.where(exists, scale, existing_strengths)
        keep_weights = torch.nn.functional.one_hot(
            existing_types, num_classes=self.num_edge_types
        ).to(dtype=logits.dtype) * probabilities[:, :1] * keep_scale.unsqueeze(-1)
        return type_weights + keep_weights * exists.unsqueeze(-1)

    def supervised_loss(self, logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy objective for observed keep/add/retype/remove actions."""

        if logits.ndim != 2 or logits.shape[-1] != self.num_edge_types + 2:
            raise ValueError("logits have the wrong action dimension")
        if action_targets.shape != (logits.shape[0],) or action_targets.dtype != torch.long:
            raise ValueError("action_targets must be a long vector with one label per candidate")
        if action_targets.numel() == 0:
            return logits.sum() * 0.0
        if not action_targets.ne(-100).any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits, action_targets, ignore_index=-100)

    def propose(
        self,
        logits: torch.Tensor,
        candidate_edges: torch.Tensor,
        topology: NeuronGraphTopology,
        *,
        threshold: float = 0.8,
        candidate_strengths: torch.Tensor | None = None,
        strength_update_threshold: float = 0.05,
    ) -> list[LinkMutation]:
        """Turn confident candidate decisions into sparse graph mutations.

        There is no per-node top-k. Every candidate above ``threshold`` may
        produce a mutation; an optional global edge budget belongs to the
        topology controller.
        """

        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if logits.shape != (candidate_edges.shape[1], self.num_edge_types + 2):
            raise ValueError("logits must align with candidate_edges")
        if not 0.0 <= strength_update_threshold:
            raise ValueError("strength_update_threshold must be non-negative")
        if candidate_strengths is not None and (
            candidate_strengths.shape != (candidate_edges.shape[1],)
            or candidate_strengths.device != logits.device
            or not candidate_strengths.is_floating_point()
            or not torch.isfinite(candidate_strengths).all()
            or torch.any(candidate_strengths < 0)
        ):
            raise ValueError("candidate_strengths must be finite non-negative values aligned with candidates")
        probabilities = logits.softmax(dim=-1).detach().cpu()
        strengths_cpu = candidate_strengths.detach().cpu() if candidate_strengths is not None else None
        candidates = candidate_edges.detach().cpu()
        existing = {
            (int(source), int(target)): (int(edge_type), float(strength))
            for source, target, edge_type, strength in zip(
                topology.edge_index[0].detach().cpu().tolist(),
                topology.edge_index[1].detach().cpu().tolist(),
                topology.edge_type.detach().cpu().tolist(),
                topology.strength.detach().cpu().tolist(),
                strict=True,
            )
        }
        mutations: list[LinkMutation] = []
        keep_action = 0
        remove_action = self.num_edge_types + 1
        for index, (source, target) in enumerate(candidates.t().tolist()):
            probability, action = probabilities[index].max(dim=0)
            confidence = float(probability)
            action_id = int(action)
            if confidence < threshold:
                continue
            source, target = int(source), int(target)
            batch = source // topology.nodes_per_graph
            source_local = source % topology.nodes_per_graph
            target_local = target % topology.nodes_per_graph
            existing_value = existing.get((source, target))
            existing_type = existing_value[0] if existing_value is not None else None
            desired_strength = (
                float(strengths_cpu[index]) if strengths_cpu is not None else confidence
            )
            if action_id == remove_action:
                if existing_type is not None:
                    mutations.append(LinkMutation("remove", batch, source_local, target_local))
            elif action_id == keep_action:
                if (
                    existing_value is not None
                    and strengths_cpu is not None
                    and abs(existing_value[1] - desired_strength) >= strength_update_threshold
                ):
                    mutations.append(
                        LinkMutation("set_strength", batch, source_local, target_local, strength=desired_strength)
                    )
            else:
                edge_type = action_id - 1
                if existing_type is None:
                    mutations.append(LinkMutation("add", batch, source_local, target_local, edge_type, desired_strength))
                elif existing_type != edge_type:
                    mutations.append(LinkMutation("retype", batch, source_local, target_local, edge_type, desired_strength))
                elif (
                    strengths_cpu is not None
                    and existing_value is not None
                    and abs(existing_value[1] - desired_strength) >= strength_update_threshold
                ):
                    mutations.append(
                        LinkMutation("set_strength", batch, source_local, target_local, strength=desired_strength)
                    )
        return mutations


def _validate_candidate_edges(
    nodes: torch.Tensor,
    candidate_edges: torch.Tensor,
    topology: NeuronGraphTopology,
    num_edge_types: int,
    causal: bool,
    check_values: bool,
) -> None:
    if nodes.ndim != 3:
        raise ValueError("nodes must have shape [batch, neurons, hidden]")
    topology.validate(
        batch_size=nodes.shape[0],
        nodes_per_graph=nodes.shape[1],
        num_edge_types=num_edge_types,
        device=nodes.device,
        causal=causal,
        check_values=check_values,
    )
    if candidate_edges.dtype != torch.long or candidate_edges.ndim != 2 or candidate_edges.shape[0] != 2:
        raise ValueError("candidate_edges must be a long tensor with shape [2, candidates]")
    if candidate_edges.device != nodes.device:
        raise ValueError("candidate_edges must be on the node tensor's device")
    if candidate_edges.numel() == 0 or not check_values:
        return
    node_count = nodes.shape[0] * nodes.shape[1]
    if int(candidate_edges.min()) < 0 or int(candidate_edges.max()) >= node_count:
        raise ValueError("candidate edge references an unknown neuron")
    source, target = candidate_edges
    if torch.any(source == target):
        raise ValueError("candidate links cannot be self links")
    if torch.any(source // nodes.shape[1] != target // nodes.shape[1]):
        raise ValueError("candidate links cannot cross batch graph boundaries")
    if causal and torch.any(source % nodes.shape[1] > target % nodes.shape[1]):
        raise ValueError("causal candidate links cannot target future information")
    ordered = source * node_count + target
    if torch.unique(ordered).numel() != candidate_edges.shape[1]:
        raise ValueError("candidate_edges cannot contain duplicate directed edges")


__all__ = ["LinkProposalPolicy"]
