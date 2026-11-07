"""Relation-aware set aggregation keyed by exact integer group identities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _PermuteRows(torch.autograd.Function):
    """Reorder rows with a deterministic inverse permutation on backward."""

    @staticmethod
    def forward(ctx, values: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        return values.index_select(0, order)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        (order,) = ctx.saved_tensors
        restored = torch.empty_like(gradient)
        restored.index_copy_(0, order, gradient)
        return restored, None


class _SegmentSum(torch.autograd.Function):
    """Sum contiguous segments with linear memory and deterministic backward."""

    @staticmethod
    def forward(
        ctx, values: Tensor, starts: Tensor, ends: Tensor, lengths: Tensor
    ) -> Tensor:
        ctx.save_for_backward(lengths)
        prefix = values.cumsum(dim=0)
        previous = prefix.index_select(0, (starts - 1).clamp_min(0))
        previous = torch.where(starts.unsqueeze(-1) > 0, previous, torch.zeros_like(previous))
        return prefix.index_select(0, ends) - previous

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None, None, None]:
        (lengths,) = ctx.saved_tensors
        return torch.repeat_interleave(gradient, lengths, dim=0), None, None, None


class GroupRoleDeepSetsAggregator(nn.Module):
    """Encode role summaries within each group, then sum group embeddings.

    Group labels are opaque integers: they are never embedded, normalized, or
    used as allocation indices. Sorting only forms contiguous equality groups.
    A bounded dense kernel serves common degrees; larger inputs use stable
    sorting and segmented sums with linear workspace, without a label-size cap.
    """

    def __init__(
        self,
        channels: int,
        num_edge_types: int,
        *,
        expansion: int = 2,
        context_size: int = 0,
        pairwise_interactions: bool = False,
        bilinear_interactions: bool = False,
        exact_bilinear: bool = False,
        full_bilinear: bool = False,
        sum_residual: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_edge_types <= 0 or expansion <= 0 or context_size < 0:
            raise ValueError("channels, num_edge_types, and expansion must be positive; context_size cannot be negative")
        self.channels = channels
        self.num_edge_types = num_edge_types
        self.context_size = context_size
        self.pairwise_interactions = pairwise_interactions or bilinear_interactions
        self.bilinear_interactions = bilinear_interactions
        self.exact_bilinear = exact_bilinear
        self.full_bilinear = full_bilinear
        self.sum_residual = sum_residual
        if sum_residual and context_size <= 0:
            raise ValueError("sum_residual requires a positive context_size")
        pair_count = num_edge_types * (num_edge_types - 1) // 2 if self.pairwise_interactions else 0
        if (exact_bilinear or full_bilinear) and (not bilinear_interactions or pair_count != 1):
            raise ValueError("bilinear readouts require exactly two role types and bilinear interactions")
        if exact_bilinear and full_bilinear:
            raise ValueError("exact_bilinear and full_bilinear are mutually exclusive")
        self.pair_projections = nn.ModuleList()
        if bilinear_interactions and not full_bilinear:
            self.pair_projections.extend(
                nn.ModuleList(
                    (
                        nn.Linear(channels, channels, bias=False),
                        nn.Linear(channels, channels, bias=False),
                    )
                )
                for _ in range(pair_count)
            )
        self.full_bilinear_layer = (
            nn.Bilinear(channels, channels, channels) if full_bilinear else None
        )
        width = channels * expansion
        if full_bilinear:
            self.group_network = nn.Identity()
        elif exact_bilinear:
            self.group_network = nn.Linear(channels * pair_count, channels)
        else:
            self.group_network = nn.Sequential(
                nn.Linear(channels * (num_edge_types + pair_count) + context_size, width),
                nn.GELU(),
                nn.Linear(width, channels),
            )
        if sum_residual:
            self.residual_gate = nn.Linear(context_size, channels)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.zeros_(self.residual_gate.bias)

    def forward_batched_with_relations(
        self,
        messages: Tensor,
        *,
        edge_types: Tensor,
        group_ids: Tensor,
        mask: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        if messages.ndim != 3 or messages.shape[-1] != self.channels:
            raise ValueError(f"messages must have shape [batch, items, {self.channels}]")
        shape = messages.shape[:2]
        if group_ids.shape != shape or group_ids.dtype != torch.long:
            raise ValueError("group_ids must be long and align with messages")
        if edge_types.shape != shape or edge_types.dtype != torch.long:
            raise ValueError("edge_types must be long and align with messages")
        if mask.shape != shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and align with messages")
        if any(x.device != messages.device for x in (group_ids, edge_types, mask)):
            raise ValueError("messages, group_ids, edge_types, and mask must share a device")
        if self.context_size:
            if context is None or context.shape != (messages.shape[0], self.context_size):
                raise ValueError("context must align with batch and configured context_size")
            if context.device != messages.device or context.dtype != messages.dtype:
                raise ValueError("context must share message device and dtype")
        valid_types = edge_types[mask]
        if valid_types.numel() and (valid_types.min() < 0 or valid_types.max() >= self.num_edge_types):
            raise ValueError("edge type is outside the configured range")

        batch, items, _ = messages.shape
        if items == 0 or not mask.any():
            return messages.new_zeros((batch, self.channels))

        # Keep the vectorized kernel for common degrees. Its workspace is
        # explicitly bounded; larger workloads use sorted O(N log N) grouping.
        pair_budget = 256**2 if messages.device.type == "cpu" else 3000**2
        if batch * items * items <= pair_budget:
            positions = torch.arange(items, device=messages.device)
            same_group = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
            earlier_valid = mask.unsqueeze(1) & (
                positions.view(1, 1, items) < positions.view(1, items, 1)
            )
            first = mask & ~(same_group & earlier_valid).any(dim=2)
            group_members = same_group & first.unsqueeze(2) & mask.unsqueeze(1)
            role_summaries = []
            for role in range(self.num_edge_types):
                role_messages = messages * ((edge_types == role) & mask).unsqueeze(-1)
                role_summaries.append(torch.bmm(group_members.to(messages.dtype), role_messages))
            if self.full_bilinear:
                group_features = self.full_bilinear_layer(role_summaries[0], role_summaries[1])
            else:
                group_features = torch.cat(role_summaries, dim=-1)
            if self.pairwise_interactions and not self.full_bilinear:
                products = []
                projection_index = 0
                for left in range(self.num_edge_types):
                    for right in range(left + 1, self.num_edge_types):
                        if self.bilinear_interactions:
                            left_projection, right_projection = self.pair_projections[projection_index]
                            products.append(
                                left_projection(role_summaries[left])
                                * right_projection(role_summaries[right])
                            )
                            projection_index += 1
                        else:
                            products.append(role_summaries[left] * role_summaries[right])
                group_features = (
                    torch.cat(products, dim=-1)
                    if self.exact_bilinear
                    else torch.cat((group_features, *products), dim=-1)
                )
            if self.context_size and not (self.exact_bilinear or self.full_bilinear):
                group_features = torch.cat(
                    (group_features, context.unsqueeze(1).expand(-1, items, -1)), dim=-1
                )
            group_values = self.group_network(group_features)
            interaction = (group_values * first.unsqueeze(-1)).sum(dim=1)
        else:
            row_interactions = []
            for row in range(batch):
                sorted_groups, order = torch.sort(group_ids[row], stable=True)
                sorted_messages = _PermuteRows.apply(messages[row], order)
                sorted_types = edge_types[row].index_select(0, order)
                sorted_mask = mask[row].index_select(0, order)
                starts = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long, device=messages.device),
                        torch.nonzero(sorted_groups[1:] != sorted_groups[:-1]).flatten() + 1,
                    )
                )
                ends = torch.cat(
                    (
                        starts[1:] - 1,
                        torch.tensor([items - 1], dtype=torch.long, device=messages.device),
                    )
                )
                lengths = ends - starts + 1

                role_summaries = [
                    _SegmentSum.apply(
                        sorted_messages
                        * ((sorted_types == role) & sorted_mask).unsqueeze(-1),
                        starts,
                        ends,
                        lengths,
                    )
                    for role in range(self.num_edge_types)
                ]
                if self.full_bilinear:
                    group_features = self.full_bilinear_layer(
                        role_summaries[0], role_summaries[1]
                    )
                else:
                    group_features = torch.cat(role_summaries, dim=-1)
                if self.pairwise_interactions and not self.full_bilinear:
                    products = []
                    projection_index = 0
                    for left in range(self.num_edge_types):
                        for right in range(left + 1, self.num_edge_types):
                            if self.bilinear_interactions:
                                left_projection, right_projection = self.pair_projections[projection_index]
                                products.append(
                                    left_projection(role_summaries[left])
                                    * right_projection(role_summaries[right])
                                )
                                projection_index += 1
                            else:
                                products.append(role_summaries[left] * role_summaries[right])
                    group_features = (
                        torch.cat(products, dim=-1)
                        if self.exact_bilinear
                        else torch.cat((group_features, *products), dim=-1)
                    )
                if self.context_size and not (self.exact_bilinear or self.full_bilinear):
                    group_features = torch.cat(
                        (
                            group_features,
                            context[row].unsqueeze(0).expand(group_features.shape[0], -1),
                        ),
                        dim=-1,
                    )
                group_values = self.group_network(group_features)
                group_valid = _SegmentSum.apply(
                    sorted_mask.to(messages.dtype).unsqueeze(-1), starts, ends, lengths
                ) > 0
                group_values = group_values * group_valid.to(messages.dtype)
                row_interactions.append(group_values.sum(dim=0))
            interaction = torch.stack(row_interactions)
        if self.sum_residual:
            if context is None or context.shape != (batch, self.context_size):
                raise ValueError("sum-residual aggregation requires aligned neuron context")
            sum_path = (messages * mask.unsqueeze(-1)).sum(dim=1)
            interaction = sum_path + torch.tanh(self.residual_gate(context)) * interaction
        return interaction


__all__ = ["GroupRoleDeepSetsAggregator"]
