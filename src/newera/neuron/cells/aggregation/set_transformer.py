"""Thin NewEra boundary around PyG's Set Transformer aggregation."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import SetTransformerAggregation


class SetTransformerSetAggregator(nn.Module):
    """Aggregate one variable-size, permutation-invariant message set.

    PyG owns the attention implementation and all learned parameters. NewEra
    defines the single-set shape and the empty-set identity value at this edge.
    Repeated rows are preserved as repeated contacts, not deduplicated.
    """

    def __init__(self, channels: int, *, heads: int = 2) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        self.channels = channels
        self.upstream = SetTransformerAggregation(
            channels=channels,
            num_seed_points=1,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
            heads=heads,
            concat=False,
            layer_norm=True,
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

        return self.forward_batched(selected.unsqueeze(0), mask=torch.ones(
            (1, selected.shape[0]), dtype=torch.bool, device=messages.device
        ))

    def forward_batched(self, messages: Tensor, *, mask: Tensor | None = None) -> Tensor:
        """Aggregate padded ``[batch, items, channels]`` sets together."""
        if messages.ndim != 3 or messages.shape[-1] != self.channels:
            raise ValueError(f"messages must have shape [batch, items, {self.channels}]")
        if not messages.is_floating_point():
            raise ValueError("messages must be floating-point")
        batch_size, item_count, _ = messages.shape
        if mask is None:
            mask = torch.ones((batch_size, item_count), dtype=torch.bool, device=messages.device)
        elif (
            mask.shape != (batch_size, item_count)
            or mask.dtype != torch.bool
            or mask.device != messages.device
        ):
            raise ValueError("mask must be a boolean matrix aligned with messages on the same device")

        active_groups = mask.any(dim=1).nonzero(as_tuple=False).flatten()
        if not active_groups.numel():
            return messages.new_zeros((batch_size, self.channels))
        selected = messages.index_select(0, active_groups)
        selected_mask = mask.index_select(0, active_groups)
        # Feed padded batches directly through PyG's public submodules. Calling
        # upstream.forward would use to_dense_batch/scatter_reduce, which has
        # no deterministic MPS implementation.
        for encoder in self.upstream.encoders:
            selected = encoder(selected, selected_mask)
        selected = self.upstream.pma(selected, selected_mask)
        for decoder in self.upstream.decoders:
            selected = decoder(selected)
        aggregated = selected.nan_to_num().mean(dim=1)
        return messages.new_zeros((batch_size, self.channels)).index_copy(
            0, active_groups, aggregated
        )


class SumResidualSetTransformerAggregator(nn.Module):
    """Keep the sum path and learn a gated set-interaction residual."""

    def __init__(self, channels: int, *, heads: int = 2) -> None:
        super().__init__()
        self.set_transformer = SetTransformerSetAggregator(channels, heads=heads)
        self.residual_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, messages: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if messages.ndim == 3:
            return self.forward_batched(messages, mask=mask)
        if messages.ndim != 2 or messages.shape[-1] != self.residual_scale.numel():
            raise ValueError("messages have an incompatible shape")
        if mask is None:
            mask = torch.ones(messages.shape[0], dtype=torch.bool, device=messages.device)
        elif mask.shape != (messages.shape[0],) or mask.dtype != torch.bool or mask.device != messages.device:
            raise ValueError("mask must be a boolean vector aligned with messages on the same device")
        selected = messages[mask]
        sum_path = selected.sum(dim=0, keepdim=True)
        context = self.set_transformer(selected)
        return sum_path + self.residual_scale * context

    def forward_batched(self, messages: Tensor, *, mask: Tensor | None = None) -> Tensor:
        if messages.ndim != 3 or messages.shape[-1] != self.residual_scale.numel():
            raise ValueError("messages have an incompatible shape")
        if mask is None:
            mask = torch.ones(messages.shape[:2], dtype=torch.bool, device=messages.device)
        elif mask.shape != messages.shape[:2] or mask.dtype != torch.bool or mask.device != messages.device:
            raise ValueError("mask must be a boolean matrix aligned with messages on the same device")
        sum_path = (messages * mask.unsqueeze(-1)).sum(dim=1)
        context = self.set_transformer.forward_batched(messages, mask=mask)
        return sum_path + self.residual_scale * context


class ContextualSumResidualSetTransformerAggregator(nn.Module):
    """Selectively add set interaction from the receiving neuron's context."""

    def __init__(self, channels: int, context_size: int, *, heads: int = 2) -> None:
        super().__init__()
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        self.set_transformer = SetTransformerSetAggregator(channels, heads=heads)
        self.context_gate = nn.Linear(context_size, channels)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

    def forward_batched_with_context(
        self, messages: Tensor, *, mask: Tensor, context: Tensor
    ) -> Tensor:
        if messages.ndim != 3 or mask.shape != messages.shape[:2]:
            raise ValueError("messages and mask must have shapes [batch, items, channels] and [batch, items]")
        if context.ndim != 2 or context.shape[0] != messages.shape[0]:
            raise ValueError("context must have shape [batch, context_size]")
        sum_path = (messages * mask.unsqueeze(-1)).sum(dim=1)
        interaction = self.set_transformer.forward_batched(messages, mask=mask)
        gate = torch.tanh(self.context_gate(context))
        return sum_path + gate * interaction

    def forward(self, messages: Tensor, *, mask: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        if context is None:
            raise ValueError("context is required")
        if messages.ndim == 3:
            if mask is None:
                mask = torch.ones(messages.shape[:2], dtype=torch.bool, device=messages.device)
            return self.forward_batched_with_context(messages, mask=mask, context=context)
        if messages.ndim != 2 or context.shape != (1, self.context_gate.in_features):
            raise ValueError("single-set messages/context have incompatible shapes")
        if mask is None:
            mask = torch.ones(messages.shape[0], dtype=torch.bool, device=messages.device)
        selected = messages[mask]
        sum_path = selected.sum(dim=0, keepdim=True)
        interaction = self.set_transformer(selected)
        return sum_path + torch.tanh(self.context_gate(context)) * interaction

__all__ = [
    "ContextualSumResidualSetTransformerAggregator",
    "SetTransformerSetAggregator",
    "SumResidualSetTransformerAggregator",
]
