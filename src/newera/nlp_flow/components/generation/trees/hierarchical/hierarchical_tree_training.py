"""Masked joint objectives and train-only rollout scheduling for tree actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import PrefixPolicyBatch
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import REAL_ACTIONS

LOSS_NAMES = (
    "action", "node_kind", "edge_field", "parent", "leaf_node", "leaf_class",
    "copy_mode", "lexical", "gap", "copy_start", "copy_length", "node_budget",
    "blueprint_node", "blueprint_edge", "blueprint_parent", "blueprint_presence",
    "viability",
)


@dataclass(frozen=True)
class RolloutCurriculum:
    """Teacher-forcing schedule; sampled prefixes are always sourced from train rows."""

    warmup_steps: int = 100
    decay_steps: int = 1000
    initial_teacher_probability: float = 1.0
    final_teacher_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.decay_steps < 1:
            raise ValueError("curriculum step counts are invalid")
        if not 0.0 <= self.final_teacher_probability <= self.initial_teacher_probability <= 1.0:
            raise ValueError("teacher-forcing probabilities must be ordered within [0, 1]")

    def teacher_probability(self, global_step: int) -> float:
        if global_step < 0:
            raise ValueError("global step cannot be negative")
        if global_step < self.warmup_steps:
            return self.initial_teacher_probability
        fraction = min(1.0, (global_step - self.warmup_steps) / self.decay_steps)
        return self.initial_teacher_probability + fraction * (
            self.final_teacher_probability - self.initial_teacher_probability
        )

    @staticmethod
    def require_train_row(row: Mapping[str, object]) -> None:
        if row.get("split") != "train":
            raise ValueError("on-policy rollout inputs may only be built from train rows")


def _zero(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return output["action"].sum() * 0.0


def _cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(logits[mask], targets[mask]) if mask.any() else zero


def hierarchical_tree_losses(
    output: Mapping[str, torch.Tensor],
    batch: PrefixPolicyBatch,
    *,
    aligned_mask: torch.Tensor | None = None,
    viability_target: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return separately logged losses; optional rollout alignment masks action labels."""
    if set(output).isdisjoint({"action"}) or output["action"].shape[:2] != batch.action.shape:
        raise ValueError("model action logits and policy batch shapes must agree")
    active = batch.step_mask.bool()
    if aligned_mask is not None:
        if aligned_mask.shape != active.shape:
            raise ValueError("rollout alignment mask must match the action steps")
        active = active & aligned_mask.bool()
    zero = _zero(output)
    open_mask = active & batch.action.eq(REAL_ACTIONS.index("open_node"))
    attach_mask = active & batch.action.eq(REAL_ACTIONS.index("attach_edge"))
    leaf_mask = active & batch.action.eq(REAL_ACTIONS.index("declare_leaf"))
    copy_mask = leaf_mask & batch.copy_label.bool()
    terms = {
        "action": _cross_entropy(output["action"], batch.action, active, zero),
        "node_kind": _cross_entropy(output["node_kind"], batch.node_kind, open_mask, zero),
        "edge_field": _cross_entropy(output["edge_field"], batch.edge_field, attach_mask, zero),
        "parent": _cross_entropy(output["parent"], batch.candidate_target, attach_mask, zero),
        "leaf_node": _cross_entropy(output["leaf_node"], batch.leaf_node, leaf_mask, zero),
        "leaf_class": _cross_entropy(output["leaf_class"], batch.leaf_class, leaf_mask, zero),
        "copy_mode": (
            F.binary_cross_entropy_with_logits(
                output["copy"].squeeze(-1)[leaf_mask], batch.copy_label[leaf_mask].float()
            ) if leaf_mask.any() else zero
        ),
        "lexical": _cross_entropy(output["lexical"], batch.lexical, leaf_mask, zero),
        "gap": _cross_entropy(output["gap"], batch.gap, leaf_mask, zero),
        "copy_start": _cross_entropy(output["copy_start"], batch.copy_start, copy_mask, zero),
        "copy_length": _cross_entropy(output["copy_length"], batch.copy_length, copy_mask, zero),
        "node_budget": F.cross_entropy(output["node_budget"], batch.node_budget),
    }
    blueprint_keys = {
        "blueprint_node_logits",
        "blueprint_edge_logits",
        "blueprint_parent_logits",
        "blueprint_presence_logits",
    }
    if blueprint_keys.issubset(output):
        blueprint_mask = batch.blueprint_presence.bool()
        if not blueprint_mask.any():
            raise ValueError("blueprint supervision cannot be empty")
        terms["blueprint_node"] = _cross_entropy(
            output["blueprint_node_logits"], batch.blueprint_node, blueprint_mask, zero
        )
        terms["blueprint_parent"] = _cross_entropy(
            output["blueprint_parent_logits"], batch.blueprint_parent, blueprint_mask, zero
        )
        edge_mask = blueprint_mask & batch.blueprint_parent.ne(0)
        terms["blueprint_edge"] = _cross_entropy(
            output["blueprint_edge_logits"], batch.blueprint_edge, edge_mask, zero
        )
        presence_logits = output["blueprint_presence_logits"]
        positive = batch.blueprint_presence.to(presence_logits.dtype)
        positive_count = positive.sum().clamp_min(1.0)
        negative_count = (1.0 - positive).sum().clamp_min(1.0)
        terms["blueprint_presence"] = F.binary_cross_entropy_with_logits(
            presence_logits, positive, pos_weight=(negative_count / positive_count).detach()
        )
    else:
        for name in ("blueprint_node", "blueprint_edge", "blueprint_parent", "blueprint_presence"):
            terms[name] = zero
    if viability_target is None:
        terms["viability"] = zero
    else:
        if viability_target.shape != active.shape:
            raise ValueError("viability targets must match rollout action steps")
        terms["viability"] = F.binary_cross_entropy_with_logits(
            output["viability"][active],
            viability_target[active].to(output["viability"].dtype),
        ) if active.any() else zero
    coefficients = dict(weights or {})
    unknown = set(coefficients) - set(LOSS_NAMES)
    if unknown or any(value < 0.0 for value in coefficients.values()):
        raise ValueError(f"invalid objective weights: unknown={sorted(unknown)}")
    total = sum((terms[name] * coefficients.get(name, 1.0) for name in LOSS_NAMES), zero)
    if not torch.isfinite(total):
        raise FloatingPointError("hierarchical tree objective is not finite")
    return total, terms


__all__ = ["LOSS_NAMES", "RolloutCurriculum", "hierarchical_tree_losses"]
