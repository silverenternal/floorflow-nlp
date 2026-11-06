"""Differentiable heads aligned with externally reported LLM benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.nlp_flow._generated_protocol import (
    BENCHMARK_REGRESSION_TARGETS,
    BENCHMARK_SCALAR_TARGETS,
    BENCHMARK_TOKEN_TARGETS,
)

SCALAR_PROXY_TARGETS = BENCHMARK_SCALAR_TARGETS
TOKEN_PROXY_TARGETS = BENCHMARK_TOKEN_TARGETS
REGRESSION_PROXY_TARGETS = BENCHMARK_REGRESSION_TARGETS


@dataclass(frozen=True)
class BenchmarkProxyConfig:
    hidden_size: int
    test_output_classes: int = 64
    max_constraints: int = 16
    retrieval_dim: int = 32

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "test_output_classes",
            "max_constraints",
            "retrieval_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class BenchmarkProxyPrediction:
    scalar_logits: dict[str, torch.Tensor]
    token_logits: dict[str, torch.Tensor]
    test_output_logits: torch.Tensor
    test_output_token_logits: torch.Tensor
    regression_logits: dict[str, torch.Tensor]
    constraint_logits: torch.Tensor
    retrieval_embeddings: torch.Tensor


class BenchmarkProxyObjective(nn.Module):
    """Outcome, trace, retrieval, patch, instruction, and calibration proxies."""

    def __init__(self, config: BenchmarkProxyConfig) -> None:
        super().__init__()
        self.config = config
        self.scalar_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.RMSNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, 1),
                )
                for name in SCALAR_PROXY_TARGETS
            }
        )
        self.token_heads = nn.ModuleDict(
            {
                name: nn.Linear(config.hidden_size, 1)
                for name in TOKEN_PROXY_TARGETS
            }
        )
        self.regression_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.RMSNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, 1),
                )
                for name in REGRESSION_PROXY_TARGETS
            }
        )
        self.test_output_head = nn.Linear(
            config.hidden_size,
            config.test_output_classes,
        )
        self.test_output_token_head = nn.Linear(
            config.hidden_size,
            config.test_output_classes,
        )
        self.constraint_head = nn.Linear(
            config.hidden_size,
            config.max_constraints,
        )
        self.retrieval_project = nn.Linear(
            config.hidden_size,
            config.retrieval_dim,
        )
        self.answer_project = nn.Linear(
            config.hidden_size,
            config.retrieval_dim,
        )

    def predict(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BenchmarkProxyPrediction:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, hidden]")
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
            if mask.shape != hidden.shape[:2]:
                raise ValueError("attention_mask must match hidden states")
            pooled = (hidden * mask[..., None]).sum(dim=1) / mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
        return BenchmarkProxyPrediction(
            scalar_logits={
                name: self.scalar_heads[name](pooled).squeeze(-1)
                for name in SCALAR_PROXY_TARGETS
            },
            token_logits={
                name: self.token_heads[name](hidden).squeeze(-1)
                for name in TOKEN_PROXY_TARGETS
            },
            test_output_logits=self.test_output_head(pooled),
            test_output_token_logits=self.test_output_token_head(hidden),
            regression_logits={
                name: self.regression_heads[name](pooled).squeeze(-1)
                for name in REGRESSION_PROXY_TARGETS
            },
            constraint_logits=self.constraint_head(pooled),
            retrieval_embeddings=F.normalize(
                self.retrieval_project(pooled),
                dim=-1,
            ),
        )

    def regression_losses(
        self,
        pooled: torch.Tensor,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        for name, target in targets.items():
            if name not in self.regression_heads:
                raise ValueError(f"unknown regression proxy target: {name}")
            logits = self.regression_heads[name](pooled).squeeze(-1)
            labels = target.to(device=logits.device, dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise ValueError(f"{name} target must match [batch]")
            active = labels.ge(0.0)
            if masks is not None and name in masks:
                active = active & masks[name].to(
                    device=logits.device,
                    dtype=torch.bool,
                )
            if not active.any():
                continue
            losses[f"L_{name}"] = F.smooth_l1_loss(
                logits[active],
                labels[active],
            )
        return losses

    def scalar_losses(
        self,
        pooled: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = {}
        for name, target in targets.items():
            if name not in self.scalar_heads:
                raise ValueError(f"unknown scalar proxy target: {name}")
            logits = self.scalar_heads[name](pooled).squeeze(-1)
            labels = target.to(device=logits.device, dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise ValueError(f"{name} target must match [batch]")
            losses[f"L_{name}"] = F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )
        return losses

    def token_losses(
        self,
        hidden: torch.Tensor,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        for name, target in targets.items():
            if name not in self.token_heads:
                raise ValueError(f"unknown token proxy target: {name}")
            logits = self.token_heads[name](hidden).squeeze(-1)
            labels = target.to(device=logits.device, dtype=logits.dtype)
            if labels.shape != logits.shape:
                raise ValueError(f"{name} target must match [batch, sequence]")
            active = labels.ge(0.0)
            if masks is not None and name in masks:
                active = active & masks[name].to(
                    device=logits.device,
                    dtype=torch.bool,
                )
            if not active.any():
                continue
            losses[f"L_{name}"] = F.binary_cross_entropy_with_logits(
                logits[active],
                labels[active],
            )
        return losses

    def test_output_loss(
        self,
        pooled: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.test_output_head(pooled)
        labels = targets.to(device=logits.device, dtype=torch.long)
        if labels.shape != (logits.shape[0],):
            raise ValueError("test output targets must match [batch]")
        return F.cross_entropy(logits, labels)

    def test_output_token_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        logits = self.test_output_token_head(hidden)
        labels = targets.to(device=logits.device, dtype=torch.long)
        if labels.shape != hidden.shape[:2]:
            raise ValueError("token test-output targets must match hidden tokens")
        active = (
            labels.ge(0)
            if mask is None
            else (
                labels.ge(0)
                & mask.to(device=logits.device, dtype=torch.bool)
            )
        )
        if active.shape != labels.shape:
            raise ValueError("token test-output mask must match targets")
        if not active.any():
            return None
        return F.cross_entropy(logits[active], labels[active])

    def constraint_loss(
        self,
        pooled: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.constraint_head(pooled)
        labels = targets.to(device=logits.device, dtype=logits.dtype)
        if labels.shape != logits.shape:
            raise ValueError("constraint targets must match [batch, constraints]")
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        if mask is None:
            return loss.mean()
        selected = mask.to(device=loss.device, dtype=torch.bool)
        if selected.shape != loss.shape or not selected.any():
            raise ValueError("constraint mask has no active entries")
        return loss[selected].mean()

    def retrieval_loss(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        *,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        if query.shape[:2] != (candidates.shape[0], self.config.hidden_size):
            raise ValueError("query must be [batch, hidden_size]")
        if (
            candidates.ndim != 3
            or candidates.shape[0] != query.shape[0]
            or candidates.shape[-1] != self.config.hidden_size
        ):
            raise ValueError("candidates must be [batch, candidates, hidden_size]")
        labels = targets.to(device=query.device, dtype=torch.long)
        if labels.shape != (query.shape[0],):
            raise ValueError("retrieval targets must match [batch]")
        projected_query = F.normalize(self.retrieval_project(query), dim=-1)
        projected_candidates = F.normalize(
            self.retrieval_project(candidates),
            dim=-1,
        )
        logits = torch.einsum(
            "bd,bkd->bk",
            projected_query,
            projected_candidates,
        )
        return F.cross_entropy(logits / temperature, labels)

    def multiple_choice_loss(
        self,
        query: torch.Tensor,
        choices: torch.Tensor,
        targets: torch.Tensor,
        *,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        if query.ndim != 2 or choices.ndim != 3:
            raise ValueError("query must be [batch, hidden] and choices [batch, choices, hidden]")
        if choices.shape[0] != query.shape[0] or choices.shape[-1] != query.shape[-1]:
            raise ValueError("choice and query shapes do not align")
        labels = targets.to(device=query.device, dtype=torch.long)
        if labels.shape != (query.shape[0],):
            raise ValueError("choice targets must match [batch]")
        projected_query = F.normalize(self.answer_project(query), dim=-1)
        projected_choices = F.normalize(
            self.answer_project(choices),
            dim=-1,
        )
        scores = torch.einsum("bd,bkd->bk", projected_query, projected_choices)
        return F.cross_entropy(scores / temperature, labels)

    @staticmethod
    def metamorphic_consistency_loss(
        original_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if original_logits.shape != perturbed_logits.shape:
            raise ValueError("original and perturbed logits must match")
        log_probability = F.log_softmax(
            perturbed_logits / temperature,
            dim=-1,
        )
        probability = F.softmax(original_logits / temperature, dim=-1)
        return F.kl_div(log_probability, probability, reduction="batchmean")

    @staticmethod
    def total_loss(
        losses: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        if not losses:
            raise ValueError("at least one proxy loss is required")
        if weights is not None:
            unknown = set(weights) - set(losses)
            if unknown:
                raise ValueError(f"weights reference absent losses: {sorted(unknown)}")
        total = None
        for name, loss in losses.items():
            weighted = loss * (1.0 if weights is None else weights.get(name, 1.0))
            total = weighted if total is None else total + weighted
        assert total is not None
        return total

    def parameter_breakdown(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            prefix = name.split(".", 1)[0]
            groups[prefix] = groups.get(prefix, 0) + parameter.numel()
        return dict(sorted(groups.items()))


__all__ = [
    "REGRESSION_PROXY_TARGETS",
    "SCALAR_PROXY_TARGETS",
    "TOKEN_PROXY_TARGETS",
    "BenchmarkProxyConfig",
    "BenchmarkProxyObjective",
    "BenchmarkProxyPrediction",
]
