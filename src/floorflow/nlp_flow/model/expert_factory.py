"""Context-conditioned low-rank expert synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ExpertBlueprint:
    rank: int
    width: int
    input_scale: torch.Tensor
    output_scale: torch.Tensor
    birth_gate: torch.Tensor


@dataclass(frozen=True)
class GeneratedExpertState:
    blueprint: ExpertBlueprint
    left: torch.Tensor
    right: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class MaterializedExpert:
    left: torch.Tensor
    right: torch.Tensor
    rank_mask: torch.Tensor
    width_mask: torch.Tensor
    input_scale: torch.Tensor
    output_scale: torch.Tensor
    birth_gate: torch.Tensor
    rank: torch.Tensor
    width: torch.Tensor


class ContextConditionedExpertFactory(nn.Module):
    """Synthesize an expert from a context vector and a parameter basis."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_rank: int = 16,
        basis_count: int = 4,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_rank = int(max_rank)
        self.basis_count = int(basis_count)
        self.context_dim = int(context_dim)
        self.context = nn.Sequential(
            nn.Linear(hidden_size, context_dim),
            nn.SiLU(),
            nn.RMSNorm(context_dim),
        )
        self.coefficients = nn.Linear(context_dim, basis_count)
        self.blueprint = nn.Linear(context_dim, 8)
        self.left_basis = nn.Parameter(
            torch.randn(basis_count, max_rank, hidden_size) * 0.02
        )
        self.right_basis = nn.Parameter(
            torch.randn(basis_count, hidden_size, max_rank) * 0.02
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.birth_gate = nn.Parameter(torch.zeros(1))

    def materialize(self, context: torch.Tensor) -> MaterializedExpert:
        if context.ndim != 2 or context.shape[-1] != self.hidden_size:
            raise ValueError("context must be [batch, hidden_size]")
        encoded = self.context(context)
        coefficients = self.coefficients(encoded)
        left = torch.einsum("bk,krd->brd", coefficients, self.left_basis)
        right = torch.einsum("bk,kdr->bdr", coefficients, self.right_basis)
        blueprint_logits = self.blueprint(encoded)
        rank_choice = blueprint_logits[:, :4].argmax(dim=-1)
        width_choice = blueprint_logits[:, 4:8].argmax(dim=-1)
        ranks = torch.tensor(
            (4, 8, 12, 16),
            device=context.device,
        )[rank_choice]
        widths = torch.tensor(
            (
                self.hidden_size // 2,
                self.hidden_size * 3 // 4,
                self.hidden_size,
                self.hidden_size,
            ),
            device=context.device,
        )[width_choice]
        rank_mask = (
            torch.arange(self.max_rank, device=context.device)[None]
            < ranks[:, None]
        )
        width_mask = (
            torch.arange(self.hidden_size, device=context.device)[None]
            < widths[:, None]
        )
        return MaterializedExpert(
            left=left * rank_mask[:, :, None],
            right=right * width_mask[:, :, None],
            rank_mask=rank_mask,
            width_mask=width_mask,
            input_scale=blueprint_logits[:, 0].sigmoid(),
            output_scale=blueprint_logits[:, 1].sigmoid(),
            birth_gate=self.birth_gate,
            rank=ranks,
            width=widths,
        )

    def apply_materialized(
        self,
        parent_hidden: torch.Tensor,
        expert: MaterializedExpert,
    ) -> torch.Tensor:
        if parent_hidden.ndim != 3 or parent_hidden.shape[-1] != self.hidden_size:
            raise ValueError("parent_hidden must be [B,N,D]")
        if expert.left.shape[0] != parent_hidden.shape[0]:
            raise ValueError("expert batch does not match parent hidden")
        expert_hidden = F.silu(
            torch.einsum("bnd,brd->bnr", parent_hidden, expert.left)
        )
        expert_output = torch.einsum(
            "bnr,bdr->bnd",
            expert_hidden,
            expert.right,
        )
        expert_output = self.norm(expert_output) * expert.width_mask[:, None]
        return (
            expert.birth_gate.to(parent_hidden.dtype)
            * expert.input_scale[:, None, None]
            * expert.output_scale[:, None, None]
            * expert_output
        )

    def forward(
        self,
        parent_hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> GeneratedExpertState:
        if parent_hidden.ndim != 3 or context.ndim != 2:
            raise ValueError("parent_hidden must be [B,N,D] and context [B,D]")
        batch = parent_hidden.shape[0]
        if context.shape != (batch, self.hidden_size):
            raise ValueError("context shape does not match parent hidden size")
        materialized = self.materialize(context)
        output = self.apply_materialized(parent_hidden, materialized)
        blueprint = ExpertBlueprint(
            rank=int(materialized.rank[0]),
            width=int(materialized.width[0]),
            input_scale=materialized.input_scale.mean(),
            output_scale=materialized.output_scale.mean(),
            birth_gate=self.birth_gate,
        )
        return GeneratedExpertState(
            blueprint=blueprint,
            left=materialized.left,
            right=materialized.right,
            output=output,
        )


__all__ = [
    "ContextConditionedExpertFactory",
    "ExpertBlueprint",
    "GeneratedExpertState",
    "MaterializedExpert",
]
