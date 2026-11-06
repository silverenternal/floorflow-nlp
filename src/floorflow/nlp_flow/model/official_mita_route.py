"""Naive route-only MiTA attention adapted from the official implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.nlp_flow.model.tensor_checks import assert_true_async


class OfficialMiTARouteAttention(nn.Module):
    """Cross-attend queries to per-landmark top-k keys without agent-key mixing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        router_count: int = 25,
        top_k: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.router_count = int(router_count)
        self.top_k = int(top_k)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _router(self, query: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = query.shape
        router_count = min(self.router_count, tokens)
        router = F.adaptive_avg_pool1d(
            query.reshape(batch * heads, head_dim, tokens),
            router_count,
        )
        return router.reshape(batch, heads, head_dim, router_count).transpose(2, 3)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        *,
        router_prior: torch.Tensor | None = None,
        router_budgets: torch.Tensor | None = None,
        router_soft_budgets: torch.Tensor | None = None,
        key_value_mask: torch.Tensor | None = None,
        router_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query_tokens.ndim != 3 or key_value_tokens.ndim != 3:
            raise ValueError("tokens must be [B,N,D]")
        batch, query_count, channels = query_tokens.shape
        key_count = key_value_tokens.shape[1]
        if channels != self.dim:
            raise ValueError("query width does not match attention dimension")
        if key_value_tokens.shape[-1] != self.dim:
            raise ValueError("key/value width does not match attention dimension")
        if key_value_mask is not None:
            if key_value_mask.shape != (batch, key_count):
                raise ValueError("key_value_mask must be [B,K]")
            assert_true_async(
                key_value_mask.any(dim=1).all(),
                "each sample needs at least one valid key",
            )
        query = self._heads(self.q_proj(query_tokens))
        key = self._heads(self.k_proj(key_value_tokens))
        value = self._heads(self.v_proj(key_value_tokens))
        router = self._router(query)
        router_count = router.shape[2]
        if (
            router_mask is None
            and key_value_mask is not None
            and key_value_mask.shape[1] == router_count
        ):
            router_mask = key_value_mask
        if router_mask is not None:
            if router_mask.shape != (batch, router_count):
                raise ValueError("router_mask must be [B,R]")
            assert_true_async(
                router_mask.any(dim=1).all(),
                "each sample needs at least one valid router",
            )
        base_top_k = min(self.top_k, key_count)
        router_key_scores = torch.einsum("bhmd,bhnd->bhmn", router, key)
        if key_value_mask is not None:
            router_key_scores = router_key_scores.masked_fill(
                ~key_value_mask[:, None, None, :],
                -1.0e9,
            )
        gate = torch.einsum("bhmd,bhnd->bhmn", router, query)
        if router_prior is not None:
            if router_prior.shape != (batch, query_count, router_count):
                raise ValueError("router_prior must be [B,N,R]")
            gate = gate + router_prior.transpose(1, 2)[:, None]
        if router_mask is not None:
            gate = gate.masked_fill(
                ~router_mask[:, None, :, None],
                -1.0e9,
            )
        if router_budgets is None:
            budgets = torch.full(
                (batch, router_count),
                base_top_k,
                device=query_tokens.device,
                dtype=torch.long,
            )
        else:
            budgets = router_budgets.to(
                device=query_tokens.device,
                dtype=torch.long,
            ).clamp(1, base_top_k)
            if budgets.shape != (batch, router_count):
                raise ValueError("router_budgets must be [B,R]")
        top_indices = router_key_scores.topk(base_top_k, dim=-1).indices
        rank = torch.arange(base_top_k, device=query_tokens.device)
        valid = rank[None, None, None, :] < budgets[:, None, :, None]
        soft_valid = None
        if router_soft_budgets is not None:
            if router_soft_budgets.shape != (batch, router_count):
                raise ValueError("router_soft_budgets must be [B,R]")
            soft_valid = torch.sigmoid(
                (
                    router_soft_budgets[:, :, None]
                    - rank[None, None, :]
                    - 0.5
                )
                / 0.1
            )
        router_index = gate.argmax(dim=2)
        gather_index = (
            router_index.unsqueeze(-1) * base_top_k
            + rank[None, None, None, :]
        )
        selected_indices = torch.gather(
            top_indices.reshape(batch, self.num_heads, router_count * base_top_k),
            2,
            gather_index.reshape(batch, self.num_heads, -1),
        ).reshape(batch, self.num_heads, query_count, base_top_k)
        selected_soft = None
        if soft_valid is not None:
            selected_soft = torch.gather(
                soft_valid[:, None].expand(
                    batch,
                    self.num_heads,
                    router_count,
                    base_top_k,
                ).reshape(
                    batch,
                    self.num_heads,
                    router_count * base_top_k,
                ),
                2,
                gather_index.reshape(batch, self.num_heads, -1),
            ).reshape(batch, self.num_heads, query_count, base_top_k)
        selected_valid = torch.gather(
            valid.expand(
                batch,
                self.num_heads,
                router_count,
                base_top_k,
            ).reshape(batch, self.num_heads, router_count * base_top_k),
            2,
            gather_index.reshape(batch, self.num_heads, -1),
        ).reshape(batch, self.num_heads, query_count, base_top_k)
        if key_value_mask is not None:
            selected_valid = selected_valid & torch.gather(
                key_value_mask[:, None, None, :].expand(
                    batch,
                    self.num_heads,
                    query_count,
                    key_count,
                ),
                -1,
                selected_indices,
            )
        safe_indices = selected_indices.clamp(0, key_count - 1)
        selected_key = torch.gather(
            key,
            2,
            safe_indices.reshape(
                batch,
                self.num_heads,
                query_count * base_top_k,
            ).unsqueeze(-1).expand(
                batch,
                self.num_heads,
                query_count * base_top_k,
                self.head_dim,
            ),
        ).reshape(batch, self.num_heads, query_count, base_top_k, self.head_dim)
        selected_value = torch.gather(
            value,
            2,
            safe_indices.reshape(
                batch,
                self.num_heads,
                query_count * base_top_k,
            ).unsqueeze(-1).expand(
                batch,
                self.num_heads,
                query_count * base_top_k,
                self.head_dim,
            ),
        ).reshape(batch, self.num_heads, query_count, base_top_k, self.head_dim)
        logits = torch.einsum("bhnd,bhnkd->bhnk", query, selected_key)
        logits = logits / math.sqrt(self.head_dim)
        if selected_soft is not None:
            logits = logits + selected_soft.clamp_min(1.0e-6).log()
        logits = logits.masked_fill(~selected_valid, float("-inf"))
        weights = self.dropout(torch.softmax(logits, dim=-1))
        output = torch.einsum("bhnk,bhnkd->bhnd", weights, selected_value)
        if router_soft_budgets is not None:
            normalized = (
                router_soft_budgets - 1.0
            ) / max(1.0, float(base_top_k - 1))
            router_scale = 0.5 + 0.5 * normalized
            selected_scale = torch.gather(
                router_scale[:, None].expand(
                    batch,
                    self.num_heads,
                    router_count,
                ),
                2,
                router_index,
            )
            output = output * selected_scale[..., None]
        output = output.transpose(1, 2).reshape(batch, query_count, self.dim)
        output = self.out_proj(output)
        routing = torch.softmax(gate, dim=2).transpose(-1, -2)
        router_assignments = F.one_hot(router_index, router_count).float()
        usage = router_assignments.mean(dim=(0, 2))
        router_load = router_assignments.mean(dim=(1, 2))
        usage = usage.mean(dim=0)
        expert_count = torch.exp(
            -(
                usage.clamp_min(1.0e-8)
                * usage.clamp_min(1.0e-8).log()
            ).sum()
        )
        valid_router = (
            router_mask
            if router_mask is not None
            else torch.ones(
                batch,
                router_count,
                device=query_tokens.device,
                dtype=torch.bool,
            )
        )
        active_budget = budgets.float() * valid_router.float()
        active_budget_mean = active_budget.sum(dim=1) / valid_router.sum(
            dim=1
        ).clamp_min(1)
        active_budget_max = active_budget.masked_fill(
            ~valid_router,
            -1.0,
        ).amax(dim=1)
        active_budget_min = active_budget.masked_fill(
            ~valid_router,
            float("inf"),
        ).amin(dim=1)
        return output, {
            "routing": routing,
            "router_index": router_index,
            "router_usage": usage,
            "router_load": router_load,
            "routing_entropy": _entropy(routing, dim=-1).mean(),
            "attention_entropy": _entropy(weights, dim=-1).mean(),
            "effective_expert_fraction": expert_count / router_count,
            "top_k": budgets.float().mean(),
            "router_budget_min": budgets.float().amin(),
            "router_budget_max": budgets.float().amax(),
            "router_budget_active_mean": active_budget_mean,
            "router_budget_active_min": active_budget_min,
            "router_budget_active_max": active_budget_max,
            "active_key_count": (
                key_value_mask.sum(dim=1).float().mean()
                if key_value_mask is not None
                else query_tokens.new_tensor(float(key_count))
            ),
        }


def _entropy(probability: torch.Tensor, dim: int) -> torch.Tensor:
    cardinality = probability.shape[dim]
    if cardinality <= 1:
        return torch.ones((), device=probability.device, dtype=probability.dtype)
    entropy = -(
        probability.clamp_min(1.0e-8) * probability.clamp_min(1.0e-8).log()
    ).sum(dim=dim)
    return entropy / math.log(cardinality)


__all__ = ["OfficialMiTARouteAttention"]
