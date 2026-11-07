"""Thin adapter around PyG DeepSetsAggregation with explicit empty-set semantics."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import DeepSetsAggregation


class DeepSetsMessageAggregator(nn.Module):
    """Permutation-invariant nonlinear set aggregation for one model neuron."""

    def __init__(self, channels: int, *, expansion: int = 2) -> None:
        super().__init__()
        if channels <= 0 or expansion <= 0:
            raise ValueError("channels and expansion must be positive")
        self.channels = channels
        width = channels * expansion
        self.upstream = DeepSetsAggregation(
            local_nn=nn.Sequential(
                nn.Linear(channels, width),
                nn.GELU(),
                nn.Linear(width, channels),
                nn.GELU(),
            ),
            global_nn=nn.Sequential(
                nn.Linear(channels, width),
                nn.GELU(),
                nn.Linear(width, channels),
            ),
        )

    def forward(self, messages: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if messages.ndim == 3:
            return self.forward_batched(messages, mask=mask)
        if messages.ndim != 2 or messages.shape[-1] != self.channels:
            raise ValueError(f"messages must have shape [items, {self.channels}]")
        if not messages.is_floating_point():
            raise ValueError("messages must be floating-point")
        if mask is None:
            mask = torch.ones(messages.shape[0], dtype=torch.bool, device=messages.device)
        elif mask.shape != (messages.shape[0],) or mask.dtype != torch.bool or mask.device != messages.device:
            raise ValueError("mask must be a boolean vector aligned with messages on the same device")
        selected = messages[mask]
        if selected.shape[0] == 0:
            return messages.new_zeros((1, self.channels))
        return self.forward_batched(
            selected.unsqueeze(0),
            mask=torch.ones((1, selected.shape[0]), dtype=torch.bool, device=messages.device),
        )

    def forward_batched(self, messages: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if messages.ndim != 3 or messages.shape[-1] != self.channels:
            raise ValueError(f"messages must have shape [batch, items, {self.channels}]")
        if not messages.is_floating_point():
            raise ValueError("messages must be floating-point")
        batch_size, item_count, _ = messages.shape
        if mask is None:
            mask = torch.ones((batch_size, item_count), dtype=torch.bool, device=messages.device)
        elif mask.shape != (batch_size, item_count) or mask.dtype != torch.bool or mask.device != messages.device:
            raise ValueError("mask must be a boolean matrix aligned with messages on the same device")
        active_groups = mask.any(dim=1).nonzero(as_tuple=False).flatten()
        if not active_groups.numel():
            return messages.new_zeros((batch_size, self.channels))
        selected = messages.index_select(0, active_groups)
        local = self.upstream.local_nn(selected)
        local = local * mask.index_select(0, active_groups).unsqueeze(-1)
        pooled = local.sum(dim=1)
        aggregated = self.upstream.global_nn(pooled)
        return messages.new_zeros((batch_size, self.channels)).index_copy(0, active_groups, aggregated)


class ContextualSumResidualDeepSetsAggregator(nn.Module):
    """Preserve sum aggregation and gate nonlinear interaction by neuron context."""

    def __init__(self, channels: int, context_size: int, *, expansion: int = 2) -> None:
        super().__init__()
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        self.deep_sets = DeepSetsMessageAggregator(channels, expansion=expansion)
        self.context_gate = nn.Linear(context_size, channels)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

    def forward_batched_with_context(self, messages: Tensor, *, mask: Tensor, context: Tensor) -> Tensor:
        if messages.ndim != 3 or messages.shape[-1] != self.deep_sets.channels:
            raise ValueError("messages must have shape [batch, items, channels]")
        if mask.shape != messages.shape[:2] or mask.dtype != torch.bool or mask.device != messages.device:
            raise ValueError("mask must be a boolean matrix aligned with messages")
        if context.shape != (messages.shape[0], self.context_gate.in_features):
            raise ValueError("context must have shape [batch, context_size]")
        sum_path = (messages * mask.unsqueeze(-1)).sum(dim=1)
        interaction = self.deep_sets.forward_batched(messages, mask=mask)
        gate = torch.tanh(self.context_gate(context))
        return sum_path + gate * interaction

    def forward(self, messages: Tensor, *, mask: Tensor, context: Tensor) -> Tensor:
        if messages.ndim == 2:
            if context.shape != (1, self.context_gate.in_features):
                raise ValueError("single-set context must have shape [1, context_size]")
            return self.forward_batched_with_context(
                messages.unsqueeze(0), mask=mask.unsqueeze(0), context=context
            )
        return self.forward_batched_with_context(messages, mask=mask, context=context)


__all__ = ["ContextualSumResidualDeepSetsAggregator", "DeepSetsMessageAggregator"]
