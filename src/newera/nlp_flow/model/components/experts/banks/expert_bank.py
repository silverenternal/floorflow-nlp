"""Generated, fixed-pool, and disabled expert residual modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from newera.nlp_flow.model.components.experts.banks.expert_factory import (
    ContextConditionedExpertFactory,
)


@dataclass(frozen=True)
class ExpertBankConfig:
    hidden_size: int
    mode: Literal["generated", "fixed_pool", "random_frozen", "disabled"] = "generated"
    expert_count: int = 4
    top_k: int = 2
    max_rank: int = 16
    basis_count: int = 4
    context_dim: int = 64

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "expert_count",
            "top_k",
            "max_rank",
            "basis_count",
            "context_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mode not in {"generated", "fixed_pool", "random_frozen", "disabled"}:
            raise ValueError("unsupported expert-bank mode")
        if self.top_k > self.expert_count:
            raise ValueError("top_k cannot exceed expert_count")


@dataclass(frozen=True)
class ExpertBankOutput:
    residual: torch.Tensor
    router_weights: torch.Tensor | None
    diagnostics: dict[str, torch.Tensor | str]


class FixedExpertPool(nn.Module):
    """Parameter-matched fixed expert bank used as the generated-expert control."""

    def __init__(
        self,
        hidden_size: int,
        *,
        expert_count: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.expert_count = int(expert_count)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.left = nn.Parameter(
            torch.randn(expert_count, rank, hidden_size)
            / math.sqrt(hidden_size)
        )
        self.right = nn.Parameter(
            torch.randn(expert_count, hidden_size, rank)
            / math.sqrt(rank)
        )
        self.router = nn.Linear(hidden_size, expert_count)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor) -> ExpertBankOutput:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, hidden]")
        router = torch.softmax(self.router(hidden), dim=-1)
        top_values, top_indices = torch.topk(
            router,
            self.top_k,
            dim=-1,
        )
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)
        sparse_weights = torch.zeros_like(router).scatter(
            -1,
            top_indices,
            top_values,
        )
        local = F.silu(
            torch.einsum("bnd,krd->bnkr", hidden, self.left)
        )
        expert_outputs = torch.einsum(
            "bnkr,kdr->bnkd",
            local,
            self.right,
        )
        residual = torch.einsum(
            "bnk,bnkd->bnd",
            sparse_weights,
            expert_outputs,
        )
        residual = self.gate.to(hidden.dtype) * residual
        usage = sparse_weights.mean(dim=(0, 1))
        balance = usage.square().sum() * self.expert_count - 1.0
        return ExpertBankOutput(
            residual=residual,
            router_weights=router,
            diagnostics={
                "expert_balance_loss": balance,
                "expert_usage_entropy": -(
                    usage.clamp_min(1.0e-8).log() * usage
                ).sum(),
                "expert_gate": self.gate,
            },
        )


class ExpertResidualBank(nn.Module):
    """Switch generated expert synthesis and fixed-pool MoE without changing the trunk."""

    def __init__(self, config: ExpertBankConfig) -> None:
        super().__init__()
        self.config = config
        self.fixed_pool = (
            FixedExpertPool(
                config.hidden_size,
                expert_count=config.expert_count,
                rank=config.max_rank,
                top_k=config.top_k,
            )
            if config.mode == "fixed_pool"
            else None
        )
        self.factory = (
            ContextConditionedExpertFactory(
                config.hidden_size,
                max_rank=config.max_rank,
                basis_count=config.basis_count,
                context_dim=config.context_dim,
            )
            if config.mode in {"generated", "random_frozen"}
            else None
        )
        if self.factory is not None and config.mode == "random_frozen":
            for parameter in self.factory.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                self.factory.birth_gate.fill_(1.0)
        self.disabled_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ExpertBankOutput:
        if self.config.mode == "disabled":
            zero = hidden * self.disabled_scale.to(hidden.dtype)
            return ExpertBankOutput(
                residual=zero,
                router_weights=None,
                diagnostics={"expert_mode": "disabled"},
            )
        if self.config.mode == "fixed_pool":
            assert self.fixed_pool is not None
            output = self.fixed_pool(hidden)
            return ExpertBankOutput(
                residual=output.residual,
                router_weights=output.router_weights,
                diagnostics={**output.diagnostics, "expert_mode": "fixed_pool"},
            )
        assert self.factory is not None
        mask = (
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        weights = mask.to(hidden.dtype)[..., None]
        context = (hidden * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        generated = self.factory(hidden, context)
        return ExpertBankOutput(
            residual=generated.output,
            router_weights=None,
            diagnostics={
                "expert_mode": self.config.mode,
                "birth_gate": generated.blueprint.birth_gate,
                "inference_generated_weights": hidden.new_tensor(1.0),
                "generated_rank": hidden.new_tensor(float(generated.blueprint.rank)),
                "generated_width": hidden.new_tensor(float(generated.blueprint.width)),
                "generated_weight_elements": hidden.new_tensor(
                    float(
                        2
                        * hidden.shape[0]
                        * generated.blueprint.rank
                        * self.config.hidden_size
                    )
                ),
                "expert_residual_norm": generated.output.square().mean(),
            },
        )

    def freeze_growth(self) -> None:
        """Freeze generated/fixed expert parameters for primary inference."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()


__all__ = [
    "ExpertBankConfig",
    "ExpertBankOutput",
    "ExpertResidualBank",
    "FixedExpertPool",
]
