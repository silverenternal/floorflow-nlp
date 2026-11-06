"""Matched fixed-expert MoE control for generated-expert disentanglement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FixedMoEOutput:
    hidden_states: torch.Tensor
    expert_weights: torch.Tensor
    effective_left: torch.Tensor
    effective_right: torch.Tensor


class FixedExpertPoolMoE(nn.Module):
    """G5 control: a router over a fixed, pretrained expert basis."""

    def __init__(
        self,
        hidden_size: int,
        *,
        expert_count: int = 4,
        rank: int = 16,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_count = int(expert_count)
        self.rank = int(rank)
        self.context = nn.Sequential(
            nn.Linear(hidden_size, context_dim),
            nn.SiLU(),
            nn.RMSNorm(context_dim),
        )
        self.router = nn.Linear(context_dim, expert_count)
        self.left_basis = nn.Parameter(
            torch.randn(expert_count, rank, hidden_size) * 0.02
        )
        self.right_basis = nn.Parameter(
            torch.randn(expert_count, hidden_size, rank) * 0.02
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.birth_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        parent_hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> FixedMoEOutput:
        if parent_hidden.ndim != 3 or context.ndim != 2:
            raise ValueError("parent_hidden must be [B,N,D] and context [B,D]")
        if context.shape != (parent_hidden.shape[0], self.hidden_size):
            raise ValueError("context shape does not match parent hidden size")
        weights = torch.softmax(self.router(self.context(context)), dim=-1)
        left = torch.einsum("bk,krd->brd", weights, self.left_basis)
        right = torch.einsum("bk,kdr->bdr", weights, self.right_basis)
        expert_hidden = F.silu(torch.einsum("bnd,brd->bnr", parent_hidden, left))
        expert_output = torch.einsum("bnr,bdr->bnd", expert_hidden, right)
        output = self.birth_gate.to(parent_hidden.dtype) * self.norm(expert_output)
        return FixedMoEOutput(
            hidden_states=parent_hidden + output,
            expert_weights=weights,
            effective_left=left,
            effective_right=right,
        )


__all__ = ["FixedExpertPoolMoE", "FixedMoEOutput"]
