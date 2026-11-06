"""Bounded meta-learning loop for deciding when to synthesize a neuron expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

EPISODE_STEPS = (
    "attempt",
    "verify",
    "diagnose",
    "synthesize",
    "adapt",
    "commit_or_retire",
)


@dataclass(frozen=True)
class MetaGrowthConfig:
    max_adaptation_steps: int = 8
    parameter_cost_weight: float = 0.2
    compute_cost_weight: float = 0.1
    dead_module_penalty: float = 0.5

    def __post_init__(self) -> None:
        if not 1 <= self.max_adaptation_steps <= 32:
            raise ValueError("max_adaptation_steps must be in [1, 32]")
        for name in (
            "parameter_cost_weight",
            "compute_cost_weight",
            "dead_module_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BirthUtilityLoss:
    total: torch.Tensor
    birth: torch.Tensor
    utility: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "birth": self.birth,
            "utility": self.utility,
        }


@dataclass(frozen=True)
class EpisodeResult:
    phase: str
    committed: bool
    predicted_utility: float
    adaptation_steps: int
    events: tuple[str, ...]


class BirthUtilityNet(nn.Module):
    """Predict post-adaptation birth utility from diagnosis-time features only."""

    def __init__(self, feature_size: int = 7, hidden_size: int = 48) -> None:
        super().__init__()
        if feature_size != 7:
            raise ValueError("birth utility requires exactly seven features")
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.utility_head = nn.Linear(hidden_size, 1)
        self.birth_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != 7:
            raise ValueError("features must be [batch, 7]")
        hidden = self.encoder(features)
        return self.utility_head(hidden).squeeze(-1), self.birth_head(hidden).squeeze(-1)

    def utility_loss(
        self,
        features: torch.Tensor,
        *,
        target_utility: torch.Tensor,
    ) -> BirthUtilityLoss:
        if target_utility.shape != (features.shape[0],):
            raise ValueError("target_utility must be [batch]")
        predicted_utility, birth_logits = self(features)
        utility = F.smooth_l1_loss(predicted_utility, target_utility)
        birth = F.binary_cross_entropy_with_logits(
            birth_logits,
            target_utility.gt(0.0).float(),
        )
        return BirthUtilityLoss(
            total=utility + 0.5 * birth,
            birth=birth,
            utility=utility,
        )


def birth_utility_target(
    *,
    post_adaptation_gain: torch.Tensor,
    random_expert_gain: torch.Tensor,
    fixed_moe_gain: torch.Tensor,
    parameter_cost: torch.Tensor,
    compute_cost: torch.Tensor,
    dead_module: torch.Tensor,
    config: MetaGrowthConfig,
) -> torch.Tensor:
    """Compute training-only utility net of parameter, compute, and dead costs."""

    values = (
        post_adaptation_gain,
        random_expert_gain,
        fixed_moe_gain,
        parameter_cost,
        compute_cost,
        dead_module,
    )
    if len({value.shape for value in values}) != 1:
        raise ValueError("all utility terms must share a shape")
    control_hurdle = torch.maximum(random_expert_gain, fixed_moe_gain)
    return (
        post_adaptation_gain
        - control_hurdle
        - config.parameter_cost_weight * parameter_cost
        - config.compute_cost_weight * compute_cost
        - config.dead_module_penalty * dead_module
    )


class MetaGrowthLoop:
    """Run bounded attempt-verify-diagnose-synthesize-adapt episodes."""

    def __init__(
        self,
        utility_model: BirthUtilityNet,
        *,
        config: MetaGrowthConfig | None = None,
    ) -> None:
        self.utility_model = utility_model
        self.config = config or MetaGrowthConfig()
        self.factory_frozen = False
        self.events: list[dict[str, object]] = []

    def freeze_factory(self) -> None:
        self.factory_frozen = True
        self.events.append({"event": "factory_frozen"})

    def run_episode(
        self,
        features: torch.Tensor,
        *,
        phase: str,
        observed_gain: float,
        requested_adaptation_steps: int,
    ) -> EpisodeResult:
        if phase not in {"meta_train", "meta_validation"}:
            raise ValueError("phase must be meta_train or meta_validation")
        if self.factory_frozen:
            raise RuntimeError("expert factory is frozen")
        if features.ndim != 2 or features.shape[0] != 1 or features.shape[-1] != 7:
            raise ValueError("episode features must be [1, 7]")
        if requested_adaptation_steps < 0:
            raise ValueError("adaptation steps must be non-negative")
        if phase == "meta_validation":
            self.events.append({"event": "evaluate_frozen_policy", "phase": phase})
            return EpisodeResult(
                phase=phase,
                committed=False,
                predicted_utility=0.0,
                adaptation_steps=0,
                events=("evaluate_frozen_policy",),
            )

        with torch.no_grad():
            utility, birth_logit = self.utility_model(features)
        predicted_utility = float(utility[0])
        should_birth = bool(torch.sigmoid(birth_logit)[0] >= 0.5) and predicted_utility > 0.0
        local_events = ["attempt", "verify", "diagnose"]
        adaptation_steps = 0
        committed = False
        if should_birth:
            local_events.append("synthesize")
            adaptation_steps = min(
                int(requested_adaptation_steps),
                self.config.max_adaptation_steps,
            )
            local_events.extend(["adapt"] * adaptation_steps)
            committed = observed_gain > 0.0
        local_events.append("commit" if committed else "retire")
        self.events.append(
            {
                "phase": phase,
                "events": tuple(local_events),
                "predicted_utility": predicted_utility,
                "observed_gain": observed_gain,
                "committed": committed,
            }
        )
        return EpisodeResult(
            phase=phase,
            committed=committed,
            predicted_utility=predicted_utility,
            adaptation_steps=adaptation_steps,
            events=tuple(local_events),
        )


__all__ = [
    "EPISODE_STEPS",
    "BirthUtilityLoss",
    "BirthUtilityNet",
    "EpisodeResult",
    "MetaGrowthConfig",
    "MetaGrowthLoop",
    "birth_utility_target",
]
