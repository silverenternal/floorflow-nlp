"""Target-free learned control of bounded neuron birth, routing, and retirement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

GROWTH_ACTIONS = ("birth", "route", "merge", "retire")
ACTION_TO_ID = {action: index for index, action in enumerate(GROWTH_ACTIONS)}


@dataclass(frozen=True)
class GrowthFeatures:
    error: torch.Tensor
    residual: torch.Tensor
    utilization: torch.Tensor
    routing_entropy: torch.Tensor
    redundancy: torch.Tensor
    parameter_cost: torch.Tensor
    module_count: torch.Tensor

    def stack(self) -> torch.Tensor:
        values = (
            self.error,
            self.residual,
            self.utilization,
            self.routing_entropy,
            self.redundancy,
            self.parameter_cost,
            self.module_count,
        )
        if any(value.ndim != 1 for value in values):
            raise ValueError("every growth feature must be one-dimensional")
        if len({value.shape for value in values}) != 1:
            raise ValueError("growth feature lengths must match")
        return torch.stack(values, dim=-1).float()


@dataclass(frozen=True)
class GrowthDecision:
    action_logits: torch.Tensor
    utility: torch.Tensor

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(GROWTH_ACTIONS[int(index)] for index in self.action_logits.argmax(dim=-1))


@dataclass(frozen=True)
class GrowthPolicyLoss:
    total: torch.Tensor
    action: torch.Tensor
    utility: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "action": self.action,
            "utility": self.utility,
        }


def growth_action_utility(features: torch.Tensor) -> torch.Tensor:
    """Return four training-only action utilities from runtime features."""

    if features.ndim != 2 or features.shape[-1] != 7:
        raise ValueError("features must be [batch, 7]")
    error, residual, utilization, entropy, redundancy, cost, count = features.unbind(dim=-1)
    birth = error * residual * (1.0 - utilization) * (count > 0).float() - cost
    route = error * utilization * entropy - 0.1 * cost
    merge = redundancy * (1.0 - error) * utilization - 0.1 * cost
    retire = 0.5 * (1.0 - utilization) + 0.2 * (1.0 - error) - cost
    return torch.stack((birth, route, merge, retire), dim=-1)


class LearnedGrowthController(nn.Module):
    """Predict action and birth utility from target-free runtime statistics."""

    def __init__(self, feature_size: int = 7, hidden_size: int = 64) -> None:
        super().__init__()
        if feature_size != 7:
            raise ValueError("growth controller requires exactly seven features")
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden_size, len(GROWTH_ACTIONS))
        self.utility_head = nn.Linear(hidden_size, len(GROWTH_ACTIONS))

    def forward(self, features: torch.Tensor) -> GrowthDecision:
        if features.ndim != 2 or features.shape[-1] != 7:
            raise ValueError("features must be [batch, 7]")
        hidden = self.encoder(features)
        return GrowthDecision(
            action_logits=self.action_head(hidden),
            utility=self.utility_head(hidden),
        )

    def policy_loss(
        self,
        features: torch.Tensor,
        *,
        action_targets: torch.Tensor,
        utility_targets: torch.Tensor,
    ) -> GrowthPolicyLoss:
        if features.ndim != 2 or features.shape[-1] != 7:
            raise ValueError("features must be [batch, 7]")
        if action_targets.shape != (features.shape[0],):
            raise ValueError("action_targets must be [batch]")
        if utility_targets.shape != (features.shape[0], len(GROWTH_ACTIONS)):
            raise ValueError("utility_targets must be [batch, actions]")
        decision = self(features)
        action = F.cross_entropy(decision.action_logits, action_targets.long())
        utility = F.smooth_l1_loss(decision.utility, utility_targets.float())
        return GrowthPolicyLoss(
            total=action + 0.5 * utility,
            action=action,
            utility=utility,
        )


class BoundedGrowthManager:
    """Apply learned decisions under hard round, budget, and freeze limits."""

    def __init__(
        self,
        *,
        max_growth_rounds: int = 3,
        parameter_budget: int,
        initial_parameters: int = 0,
    ) -> None:
        if not 1 <= max_growth_rounds <= 3:
            raise ValueError("max_growth_rounds must be in [1, 3]")
        if parameter_budget <= 0 or initial_parameters < 0:
            raise ValueError("parameters must be non-negative")
        if initial_parameters > parameter_budget:
            raise ValueError("initial parameters exceed the budget")
        self.max_growth_rounds = int(max_growth_rounds)
        self.parameter_budget = int(parameter_budget)
        self.active_parameters = int(initial_parameters)
        self.growth_rounds = 0
        self.frozen = False
        self.events: list[dict[str, object]] = []

    def _validate_unfrozen(self) -> None:
        if self.frozen:
            raise RuntimeError("growth manager is frozen")

    def apply(
        self,
        action: str,
        *,
        birth_parameters: int = 0,
        retired_parameters: int = 0,
    ) -> dict[str, object]:
        self._validate_unfrozen()
        if action not in GROWTH_ACTIONS:
            raise ValueError(f"unknown growth action: {action}")
        if birth_parameters < 0 or retired_parameters < 0:
            raise ValueError("parameter deltas must be non-negative")
        if action == "birth" and birth_parameters <= 0:
            raise ValueError("birth requires positive parameters")
        if action != "birth" and birth_parameters:
            raise ValueError("only birth can allocate new parameters")
        net_delta = birth_parameters - retired_parameters
        next_total = self.active_parameters + net_delta
        if net_delta > 0:
            raise ValueError("net parameter growth is forbidden")
        if next_total > self.parameter_budget:
            raise ValueError("growth exceeds the parameter budget")
        if action == "birth":
            if self.growth_rounds >= self.max_growth_rounds:
                raise ValueError("maximum growth rounds reached")
            self.growth_rounds += 1
        self.active_parameters = next_total
        event = {
            "index": len(self.events),
            "action": action,
            "birth_parameters": birth_parameters,
            "retired_parameters": retired_parameters,
            "net_parameter_delta": net_delta,
            "active_parameters": self.active_parameters,
            "growth_round": self.growth_rounds,
        }
        self.events.append(event)
        return event

    def freeze(self) -> None:
        self.frozen = True
        self.events.append(
            {
                "index": len(self.events),
                "action": "freeze",
                "active_parameters": self.active_parameters,
                "growth_round": self.growth_rounds,
            }
        )


def random_growth_action(seed: int, count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, len(GROWTH_ACTIONS), (count,), generator=generator)


__all__ = [
    "ACTION_TO_ID",
    "GROWTH_ACTIONS",
    "BoundedGrowthManager",
    "GrowthDecision",
    "GrowthFeatures",
    "GrowthPolicyLoss",
    "LearnedGrowthController",
    "growth_action_utility",
    "random_growth_action",
]
