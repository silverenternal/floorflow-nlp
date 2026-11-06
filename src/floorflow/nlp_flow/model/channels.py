"""Five decoupled typed channel memories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from floorflow.nlp_flow._generated_protocol import TYPED_CHANNEL_NAMES


def channel_relation_prior() -> torch.Tensor:
    prior = torch.full((len(TYPED_CHANNEL_NAMES), len(TYPED_CHANNEL_NAMES)), -1.0)
    prior.diagonal().fill_(1.0)
    ids = {name: index for index, name in enumerate(TYPED_CHANNEL_NAMES)}
    links = {
        ("syntax", "symbol"): 1.2,
        ("syntax", "execution"): 0.9,
        ("symbol", "discourse"): 0.8,
        ("symbol", "execution"): 0.8,
        ("discourse", "engineering"): 0.6,
        ("execution", "engineering"): 1.4,
    }
    for (left, right), value in links.items():
        prior[ids[left], ids[right]] = value
        prior[ids[right], ids[left]] = value
    return prior


@dataclass(frozen=True)
class TypedChannelMemory:
    global_memory: torch.Tensor
    token_states: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    relation_prior: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class TypedChannelMemoryEncoder(nn.Module):
    """Independent low-rank encoders followed by masked channel pooling."""

    def __init__(self, hidden_size: int, channel_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.channel_size = int(channel_size)
        self.down = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, channel_size, bias=False)
                for name in TYPED_CHANNEL_NAMES
            }
        )
        self.norm = nn.ModuleDict(
            {name: nn.RMSNorm(channel_size) for name in TYPED_CHANNEL_NAMES}
        )
        self.out = nn.ModuleDict(
            {
                name: nn.Linear(channel_size, hidden_size, bias=False)
                for name in TYPED_CHANNEL_NAMES
            }
        )
        self.register_buffer("relation_prior", channel_relation_prior())

    def forward(
        self,
        hidden: torch.Tensor,
        channel_masks: dict[str, torch.Tensor] | None = None,
    ) -> TypedChannelMemory:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, channels]")
        batch, sequence, _ = hidden.shape
        token_states: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        tokens = []
        diagnostics: dict[str, torch.Tensor] = {}
        for name in TYPED_CHANNEL_NAMES:
            mask = (
                torch.ones(batch, sequence, device=hidden.device, dtype=torch.bool)
                if channel_masks is None or name not in channel_masks
                else channel_masks[name].bool()
            )
            if mask.shape != (batch, sequence):
                raise ValueError(f"channel mask {name} must be [batch, sequence]")
            effective = mask.clone()
            effective[~effective.any(dim=1), 0] = True
            state = torch.tanh(self.norm[name](self.down[name](hidden)))
            weights = effective.to(hidden.dtype)
            pooled = (state * weights[..., None]).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            channel_token = self.out[name](pooled)[:, None]
            token_states[name] = state
            masks[name] = effective
            tokens.append(channel_token)
            diagnostics[f"{name}_coverage"] = weights.mean()
            diagnostics[f"{name}_state_norm"] = state.square().mean()
        memory = torch.cat(tokens, dim=1)
        return TypedChannelMemory(
            global_memory=memory,
            token_states=token_states,
            masks=masks,
            relation_prior=self.relation_prior.to(memory.dtype),
            diagnostics=diagnostics,
        )


__all__ = [
    "TYPED_CHANNEL_NAMES",
    "TypedChannelMemory",
    "TypedChannelMemoryEncoder",
    "channel_relation_prior",
]
