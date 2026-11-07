"""Parameter-isolated staged training for nlp_flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import (
    TRAINING_SCOPE_GROUP_NAMES,
)

SCOPE_PREFIXES: dict[str, tuple[str, ...]] = {
    "base": ("embedding.", "blocks.", "final_norm.", "output."),
    "turn": ("turn_embedding.",),
    "task": ("task_family_embedding.", "task_language_embedding."),
    "dynamic": ("dynamic_stack.",),
    "channels": ("typed_memory.",),
    "cross_attention": ("typed_fusion.",),
    "cfm": ("cfm.",),
    "json": (
        "json_head.",
        "json_typed_heads.",
        "json_schema_key_head.",
    ),
    "gap_adapters": (
        "gap_ensemble.adapters.",
        "gap_ensemble.loss_heads.",
    ),
    "gap_router": ("gap_ensemble.aggregator.",),
    "proxy": ("benchmark_proxy.",),
    "conversation": ("conversation_head.",),
    "factory": ("expert_factory.",),
    "growth": ("expert_bank.", "neuron."),
    "unified": ("unified_competence.",),
    "runtime_competence": (
        "runtime_competence.detector.",
    ),
    "unified_planner_shared": (
        "unified_competence.planner.norm.",
        "unified_competence.planner.latent.",
    ),
    "unified_depth_route": (
        "unified_competence.planner.head_adapters.depth.",
        "unified_competence.planner.depth.",
    ),
    "unified_lane_route": (
        "unified_competence.planner.head_adapters.lane.",
        "unified_competence.planner.lane.",
    ),
    "unified_competence_route": (
        "unified_competence.planner.head_adapters.competence.",
        "unified_competence.planner.competence.",
    ),
    "unified_birth_route": (
        "unified_competence.planner.head_adapters.birth.",
        "unified_competence.planner.birth.",
        "unified_competence.generated_expert.",
    ),
    "unified_budget_route": (
        "unified_competence.planner.head_adapters.budget.",
        "unified_competence.planner.budget.",
    ),
}

if set(SCOPE_PREFIXES) != set(TRAINING_SCOPE_GROUP_NAMES):
    raise RuntimeError(
        "training scope prefixes are out of sync with Rust declarations: "
        f"missing={sorted(set(TRAINING_SCOPE_GROUP_NAMES) - set(SCOPE_PREFIXES))}, "
        f"extra={sorted(set(SCOPE_PREFIXES) - set(TRAINING_SCOPE_GROUP_NAMES))}"
    )


@dataclass(frozen=True)
class TrainingScope:
    groups: tuple[str, ...]

    @property
    def name(self) -> str:
        return "+".join(self.groups)


ALIASES: dict[str, TrainingScope] = {
    "S0-base": TrainingScope(("base",)),
    "S1-channels": TrainingScope(("channels",)),
    "S2-cross-attention": TrainingScope(("cross_attention",)),
    "S3-cfm": TrainingScope(("cfm",)),
    "S4-joint": TrainingScope(("base", "channels", "cross_attention", "cfm", "json")),
    "S5-factory": TrainingScope(
        ("base", "channels", "cross_attention", "cfm", "json", "factory")
    ),
    "S1-gaps": TrainingScope(("gap_adapters",)),
    "S2-gap-router": TrainingScope(("gap_router",)),
    "S3-dynamic": TrainingScope(("dynamic",)),
    "S4-proxy": TrainingScope(("proxy",)),
    "S5-conversation": TrainingScope(("conversation", "turn")),
    "S5-task-metadata": TrainingScope(("task",)),
    "S6-growth": TrainingScope(("growth", "factory")),
    "S6-unified-competence": TrainingScope(("unified",)),
    "S8-planner-depth": TrainingScope(("unified_depth_route",)),
    "S8-planner-lane": TrainingScope(("unified_lane_route",)),
    "S8-planner-competence": TrainingScope(("unified_competence_route",)),
    "S8-planner-birth": TrainingScope(("unified_birth_route",)),
    "S8-planner-budget": TrainingScope(("unified_budget_route",)),
    "S9-runtime-competence": TrainingScope(("runtime_competence",)),
    "S7-joint-expanded": TrainingScope(
        (
            "base",
            "turn",
            "task",
            "dynamic",
            "channels",
            "cross_attention",
            "cfm",
            "json",
            "gap_adapters",
            "gap_router",
            "proxy",
            "conversation",
            "growth",
            "unified",
            "runtime_competence",
        )
    ),
}


def apply_training_scope(model: nn.Module, scope: TrainingScope) -> dict[str, object]:
    selected = []
    for name, parameter in model.named_parameters():
        enabled = any(name.startswith(SCOPE_PREFIXES[group]) for group in scope.groups)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise ValueError(f"training scope selects no parameters: {scope.name}")
    return {
        "scope": scope.name,
        "trainable_tensors": len(selected),
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
        "prefixes": tuple(sorted({name.split(".", 1)[0] for name, _ in selected})),
    }


class ScopeController:
    def __init__(self, scope: TrainingScope) -> None:
        self.scope = scope
        self.history: list[dict[str, object]] = []

    @classmethod
    def from_name(cls, name: str) -> ScopeController:
        if name not in ALIASES:
            raise ValueError(f"unknown training scope: {name}")
        return cls(ALIASES[name])

    def apply(self, model: nn.Module) -> dict[str, object]:
        report = apply_training_scope(model, self.scope)
        self.history.append(report)
        return report


@torch.no_grad()
def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


@torch.no_grad()
def frozen_parameter_delta(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {
        name: float((parameter - snapshot[name]).abs().max())
        for name, parameter in model.named_parameters()
        if name in snapshot
    }


__all__ = [
    "ALIASES",
    "SCOPE_PREFIXES",
    "ScopeController",
    "TrainingScope",
    "apply_training_scope",
    "frozen_parameter_delta",
    "parameter_snapshot",
]
