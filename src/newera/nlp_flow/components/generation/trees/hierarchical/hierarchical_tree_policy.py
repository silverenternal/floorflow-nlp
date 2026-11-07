"""Blueprint supervision and structure losses for the hierarchical tree policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def encode_blueprint_targets(
    rows: Sequence[Mapping[str, object]],
    vocabulary: Mapping[str, object],
    *,
    max_nodes: int,
) -> dict[str, torch.Tensor]:
    """Encode preorder node, parent and edge targets from teacher action records."""
    node_ids = vocabulary["node_ids"]
    edge_ids = vocabulary["edge_ids"]
    nodes = torch.zeros((len(rows), max_nodes), dtype=torch.long)
    edges = torch.zeros_like(nodes)
    parents = torch.zeros_like(nodes)
    presence = torch.zeros((len(rows), max_nodes), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        pending_edge: tuple[int, str] | None = None
        node_count = 0
        for action in row["actions"]:
            if action["type"] == "attach_edge":
                if pending_edge is not None:
                    raise ValueError("blueprint input has consecutive unattached edge actions")
                pending_edge = (int(action["parent_index"]), str(action["edge_field"]))
                continue
            if action["type"] != "open_node":
                continue
            if node_count >= max_nodes:
                raise ValueError("teacher blueprint exceeds the node budget")
            kind = str(action["node_kind"])
            kind_id = node_ids.get(kind)
            if kind_id is None:
                raise ValueError(f"blueprint node kind is outside the vocabulary: {kind}")
            nodes[row_index, node_count] = int(kind_id)
            if node_count == 0:
                if pending_edge is not None:
                    raise ValueError("blueprint root cannot have a parent edge")
            else:
                if pending_edge is None:
                    raise ValueError("non-root blueprint node is missing its teacher edge")
                parent_index, field = pending_edge
                if parent_index < 0 or parent_index >= node_count:
                    raise ValueError("blueprint parent must precede its child in preorder")
                field_id = edge_ids.get(field)
                if field_id is None:
                    raise ValueError(f"blueprint edge field is outside the vocabulary: {field}")
                parents[row_index, node_count] = parent_index + 1
                edges[row_index, node_count] = int(field_id)
                pending_edge = None
            presence[row_index, node_count] = True
            node_count += 1
        if pending_edge is not None or node_count == 0:
            raise ValueError("blueprint action sequence is incomplete")
    return {
        "blueprint_node": nodes,
        "blueprint_edge": edges,
        "blueprint_parent": parents,
        "blueprint_presence": presence,
    }


def blueprint_structure_loss(
    output: Mapping[str, torch.Tensor],
    *,
    node: torch.Tensor | None,
    edge: torch.Tensor | None,
    parent: torch.Tensor | None,
    presence: torch.Tensor | None,
) -> torch.Tensor:
    """Train predicted blueprint content without feeding targets to the planner."""
    if node is None or edge is None or parent is None or presence is None:
        return output["action"].new_zeros(())
    valid = presence.bool()
    if not valid.any():
        raise ValueError("blueprint node supervision cannot be empty")
    node_loss = F.cross_entropy(output["blueprint_node_logits"][valid], node[valid])
    parent_loss = F.cross_entropy(output["blueprint_parent_logits"][valid], parent[valid])
    edge_mask = valid & parent.ne(0)
    edge_loss = (
        F.cross_entropy(output["blueprint_edge_logits"][edge_mask], edge[edge_mask])
        if edge_mask.any()
        else node_loss.new_zeros(())
    )
    presence_logits = output["blueprint_presence_logits"]
    positive = presence.to(dtype=presence_logits.dtype)
    positives = positive.sum().clamp_min(1.0)
    negatives = (1.0 - positive).sum().clamp_min(1.0)
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logits,
        positive,
        pos_weight=(negatives / positives).detach(),
    )
    return node_loss + edge_loss + parent_loss + presence_loss


__all__ = ["blueprint_structure_loss", "encode_blueprint_targets"]
