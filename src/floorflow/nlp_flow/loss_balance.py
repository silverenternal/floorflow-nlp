"""Learned multi-objective balancing for the NLP joint training objective."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class NLPJointLossBalancer(nn.Module):
    """Kendall-style uncertainty weighting without static loss coefficients."""

    def __init__(self, objectives: tuple[str, ...]) -> None:
        super().__init__()
        if not objectives or len(set(objectives)) != len(objectives):
            raise ValueError("objectives must be unique and non-empty")
        self.objectives = objectives
        self.log_variance = nn.Parameter(torch.zeros(len(objectives)))

    def forward(
        self,
        losses: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = set(self.objectives) - set(losses)
        extra = set(losses) - set(self.objectives)
        if missing or extra:
            raise ValueError(
                f"loss keys disagree: missing={sorted(missing)} extra={sorted(extra)}"
            )
        values = torch.stack(
            [losses[name].reshape(()) for name in self.objectives]
        )
        log_variance = self.log_variance.to(values)
        precision = (-log_variance).exp()
        weighted = precision * values + log_variance
        total = weighted.sum()
        weights = {
            name: precision[index].detach()
            for index, name in enumerate(self.objectives)
        }
        return total, weights


NLP_OBJECTIVES = (
    "task",
    "ast",
    "json",
    "execution",
    "emergence",
    "proxy",
    "validity",
)


__all__ = ["NLP_OBJECTIVES", "NLPJointLossBalancer"]
