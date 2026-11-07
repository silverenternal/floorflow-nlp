"""Low-rank output scorers for sparse, high-cardinality prediction heads."""

from __future__ import annotations

import torch
from torch import nn


class FactorizedClassHead(nn.Module):
    """Factorize hidden-to-class weights while preserving the class-logit interface."""

    def __init__(self, hidden_size: int, class_count: int, rank: int) -> None:
        super().__init__()
        if min(hidden_size, class_count, rank) < 1 or rank >= hidden_size:
            raise ValueError("rank must be positive and smaller than the hidden and class dimensions")
        self.hidden_size = hidden_size
        self.class_count = class_count
        self.rank = rank
        self.query_projection = nn.Linear(hidden_size, rank, bias=False)
        self.class_embeddings = nn.Parameter(torch.empty(class_count, rank))
        self.class_bias = nn.Parameter(torch.zeros(class_count))
        nn.init.normal_(self.class_embeddings, mean=0.0, std=rank**-0.5)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden state width differs from the factorized head contract")
        return torch.nn.functional.linear(
            self.query_projection(hidden),
            self.class_embeddings,
            self.class_bias,
        )


def apply_legal_candidate_mask(
    logits: torch.Tensor,
    legal_candidates: torch.Tensor,
    active_steps: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask the same grammar-certified candidates used by free-running selection."""
    if logits.shape != legal_candidates.shape or logits.shape[:2] != active_steps.shape:
        raise ValueError("candidate mask, logits, and active-step shapes must align")
    if torch.any(active_steps & ~legal_candidates.any(dim=-1)):
        raise ValueError("an active node-kind decision has no legal grammar candidate")
    if targets is not None:
        target_is_legal = legal_candidates.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if torch.any(active_steps & ~target_is_legal):
            raise ValueError("teacher node-kind target is outside the grammar candidate set")
    checked_mask = torch.where(active_steps[..., None], legal_candidates, True)
    return logits.masked_fill(~checked_mask, -torch.inf)


def rebase_residual_budget_logits(
    residual_logits: torch.Tensor,
    minimum_budget: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Place learned nonnegative residual scores above a deterministic feasible floor."""
    if residual_logits.ndim != 2 or minimum_budget.shape != residual_logits.shape[:1]:
        raise ValueError("budget logits and per-row certified minimums must align")
    if torch.any(minimum_budget < 1) or torch.any(minimum_budget >= residual_logits.shape[-1]):
        raise ValueError("certified minimum budget is outside the model support")
    if targets is not None and torch.any(targets < minimum_budget):
        raise ValueError("target budget is below its certified grammar minimum")
    absolute_logits = torch.full_like(residual_logits, -torch.inf)
    for row, minimum in enumerate(minimum_budget.tolist()):
        count = residual_logits.shape[-1] - minimum
        absolute_logits[row, minimum:] = residual_logits[row, :count]
    return absolute_logits


__all__ = [
    "FactorizedClassHead",
    "apply_legal_candidate_mask",
    "rebase_residual_budget_logits",
]
