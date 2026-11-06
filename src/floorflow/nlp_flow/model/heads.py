"""Zero-gated typed learning heads for the compact decoder."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.nlp_flow._generated_protocol import TYPED_CHANNEL_NAMES


class TypedChannelHeads(nn.Module):
    """Independent bottleneck projections for the five typed channels."""

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
        self.up = nn.ModuleDict(
            {
                name: nn.Linear(channel_size, hidden_size, bias=False)
                for name in TYPED_CHANNEL_NAMES
            }
        )
        self.norm = nn.ModuleDict(
            {
                name: nn.RMSNorm(channel_size)
                for name in TYPED_CHANNEL_NAMES
            }
        )
        for module in self.up.values():
            nn.init.zeros_(module.weight)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        for name in TYPED_CHANNEL_NAMES:
            state = torch.nn.functional.silu(self.norm[name](self.down[name](hidden)))
            outputs[name] = hidden + self.up[name](state)
        return outputs


__all__ = ["TYPED_CHANNEL_NAMES", "TypedChannelHeads"]
