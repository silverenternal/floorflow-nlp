"""Independent typed channel objective heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import TYPED_CHANNEL_NAMES


@dataclass(frozen=True)
class ChannelObjectiveSpec:
    kind: str
    output_dim: int


class ChannelObjectiveHead(nn.Module):
    def __init__(self, input_size: int, spec: ChannelObjectiveSpec) -> None:
        super().__init__()
        self.spec = spec
        self.net = nn.Sequential(
            nn.RMSNorm(input_size),
            nn.Linear(input_size, input_size * 2),
            nn.SiLU(),
            nn.Linear(input_size * 2, spec.output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TypedChannelObjectiveBank(nn.Module):
    def __init__(self, input_size: int, specs: Mapping[str, ChannelObjectiveSpec]) -> None:
        super().__init__()
        if tuple(specs) != TYPED_CHANNEL_NAMES:
            raise ValueError("channel objective specs must follow the five typed channels")
        self.heads = nn.ModuleDict(
            {
                name: ChannelObjectiveHead(input_size, specs[name])
                for name in TYPED_CHANNEL_NAMES
            }
        )

    def forward(self, channel_states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(channel_states) != set(TYPED_CHANNEL_NAMES):
            raise ValueError("all typed channel states are required")
        return {
            name: self.heads[name](channel_states[name])
            for name in TYPED_CHANNEL_NAMES
        }


def channel_objective_loss(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor] | None,
    specs: Mapping[str, ChannelObjectiveSpec],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = None
    components = {}
    for name in TYPED_CHANNEL_NAMES:
        if name not in logits or name not in targets or name not in specs:
            raise ValueError(f"missing objective data for channel {name}")
        mask = (
            torch.ones(targets[name].shape, device=logits[name].device, dtype=torch.bool)
            if masks is None or name not in masks
            else masks[name].bool()
        )
        if mask.shape != targets[name].shape:
            raise ValueError(f"mask shape mismatch for {name}")
        if not mask.any():
            continue
        spec = specs[name]
        if spec.kind == "binary":
            loss = F.binary_cross_entropy_with_logits(
                logits[name][..., 0][mask],
                targets[name][mask].float(),
            )
        elif spec.kind == "multiclass":
            loss = F.cross_entropy(logits[name][mask], targets[name][mask].long())
        else:
            raise ValueError(f"unsupported channel objective: {spec.kind}")
        weighted = float(weights.get(name, 1.0)) * loss
        total = weighted if total is None else total + weighted
        components[f"{name}_loss"] = loss.detach()
    if total is None:
        raise ValueError("no channel positions are active")
    components["loss"] = total.detach()
    return total, components


__all__ = [
    "ChannelObjectiveSpec",
    "TypedChannelObjectiveBank",
    "channel_objective_loss",
]
