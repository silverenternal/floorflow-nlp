"""Independently trainable capability-gap adapters and cross-attention fusion."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

GAP_NAMES = (
    "json_validity",
    "ast_structure",
    "python_execution",
    "rust_execution",
    "cpp_execution",
    "engineering_acceleration",
    "model_architecture",
    "language_retention",
    "compositional_emergence",
)


@dataclass(frozen=True)
class GapEnsembleConfig:
    hidden_size: int
    enabled_gaps: tuple[str, ...] = GAP_NAMES
    adapter_rank: int = 8
    fusion_dim: int = 64
    heads: int = 4

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "adapter_rank",
            "fusion_dim",
            "heads",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fusion_dim % self.heads:
            raise ValueError("fusion_dim must be divisible by heads")
        if not self.enabled_gaps:
            raise ValueError("at least one gap must be enabled")
        if len(set(self.enabled_gaps)) != len(self.enabled_gaps):
            raise ValueError("enabled gaps must be unique")
        unknown = set(self.enabled_gaps) - set(GAP_NAMES)
        if unknown:
            raise ValueError(f"unknown gaps: {sorted(unknown)}")


@dataclass(frozen=True)
class GapEnsembleOutput:
    hidden_states: torch.Tensor
    gap_states: dict[str, torch.Tensor]
    gap_logits: dict[str, torch.Tensor]
    routing_weights: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class _GapAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(hidden)))


class GapCrossAttentionAggregator(nn.Module):
    """Token-conditioned sparse soft attention over independently produced gaps."""

    def __init__(
        self,
        hidden_size: int,
        fusion_dim: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.fusion_dim = int(fusion_dim)
        self.heads = int(heads)
        self.head_dim = self.fusion_dim // self.heads
        self.query = nn.Linear(hidden_size, fusion_dim, bias=False)
        self.key = nn.Linear(hidden_size, fusion_dim, bias=False)
        self.value = nn.Linear(hidden_size, fusion_dim, bias=False)
        self.query_norm = nn.RMSNorm(self.head_dim)
        self.key_norm = nn.RMSNorm(self.head_dim)
        self.out = nn.Linear(fusion_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: torch.Tensor,
        gap_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        gap_count = len(gap_states)
        query = self.query(hidden).reshape(
            batch,
            sequence,
            self.heads,
            self.head_dim,
        )
        memory = torch.stack(gap_states, dim=1).reshape(
            batch,
            gap_count * sequence,
            self.hidden_size,
        )
        key = self.key(memory).reshape(
            batch,
            gap_count * sequence,
            self.heads,
            self.head_dim,
        )
        value = self.value(memory).reshape(
            batch,
            gap_count * sequence,
            self.heads,
            self.head_dim,
        )
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        weights = torch.softmax(
            torch.matmul(query, key) / math.sqrt(self.head_dim),
            dim=-1,
        )
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch,
            sequence,
            self.fusion_dim,
        )
        residual = self.gate.to(hidden.dtype) * self.out(attended)
        gap_weights = weights.reshape(
            batch,
            self.heads,
            sequence,
            gap_count,
            sequence,
        ).sum(dim=-1).mean(dim=1)
        return residual, gap_weights


class GapEnsemble(nn.Module):
    """Keep gap optimization isolated until a zero-gated attention aggregation."""

    def __init__(self, config: GapEnsembleConfig) -> None:
        super().__init__()
        self.config = config
        names = tuple(config.enabled_gaps)
        self.enabled_gaps = names
        self.adapters = nn.ModuleDict(
            {
                name: _GapAdapter(config.hidden_size, config.adapter_rank)
                for name in names
            }
        )
        self.loss_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_size, 1)
                for name in names
            }
        )
        self.aggregator = GapCrossAttentionAggregator(
            config.hidden_size,
            config.fusion_dim,
            config.heads,
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _active_gaps(self, active_gaps: Iterable[str] | None) -> tuple[str, ...]:
        if active_gaps is None:
            return self.enabled_gaps
        requested = tuple(active_gaps)
        if not requested:
            raise ValueError("active_gaps cannot be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("active_gaps must be unique")
        unavailable = set(requested) - set(self.enabled_gaps)
        if unavailable:
            raise ValueError(
                f"gaps are unavailable in this module: {sorted(unavailable)}"
            )
        return requested

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        active_gaps: Iterable[str] | None = None,
        router_enabled: bool = True,
    ) -> GapEnsembleOutput:
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden must be [batch, sequence, hidden_size]")
        names = self._active_gaps(active_gaps)
        gap_states = {
            name: hidden + self.adapters[name](hidden)
            for name in names
        }
        gap_logits = {
            name: self.loss_heads[name](gap_states[name]).squeeze(-1)
            for name in names
        }
        if router_enabled:
            residual, routing_weights = self.aggregator(
                hidden,
                [gap_states[name] for name in names],
            )
        else:
            residual = torch.zeros_like(hidden)
            routing_weights = hidden.new_full(
                (hidden.shape[0], hidden.shape[1], len(names)),
                1.0 / len(names),
            )
        updated = hidden + residual
        utilization = routing_weights.mean(dim=(0, 1))
        balance_loss = (
            utilization.square().sum() * len(names) - 1.0
        )
        self.last_diagnostics = {
            "aggregator_gate": self.aggregator.gate,
            "gap_balance_loss": balance_loss,
            "active_gap_count": hidden.new_tensor(float(len(names))),
            "routing_entropy": -(
                utilization.clamp_min(1.0e-8).log() * utilization
            ).sum(),
            "residual_norm": residual.square().mean(),
            "router_enabled": hidden.new_tensor(float(router_enabled)),
        }
        diagnostics = {
            **self.last_diagnostics,
            **{
                f"{name}_utilization": utilization[index]
                for index, name in enumerate(names)
            },
        }
        return GapEnsembleOutput(
            hidden_states=updated,
            gap_states=gap_states,
            gap_logits=gap_logits,
            routing_weights=routing_weights,
            diagnostics=diagnostics,
        )

    def supervised_loss(
        self,
        output: GapEnsembleOutput,
        gap_name: str,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gap_name not in output.gap_logits:
            raise ValueError(f"gap {gap_name} was not active in this forward")
        logits = output.gap_logits[gap_name]
        target = targets.to(device=logits.device, dtype=logits.dtype)
        if target.shape != logits.shape:
            raise ValueError("targets must match gap logits")
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        if mask is None:
            return loss.mean()
        selected = mask.to(device=loss.device, dtype=torch.bool)
        if selected.shape != loss.shape:
            raise ValueError("mask must match gap logits")
        if not selected.any():
            raise ValueError("gap supervision mask has no active targets")
        return loss[selected].mean()

    def parameter_breakdown(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            prefix = name.split(".", 1)[0]
            groups[prefix] = groups.get(prefix, 0) + parameter.numel()
        return dict(sorted(groups.items()))


__all__ = [
    "GAP_NAMES",
    "GapCrossAttentionAggregator",
    "GapEnsemble",
    "GapEnsembleConfig",
    "GapEnsembleOutput",
]
