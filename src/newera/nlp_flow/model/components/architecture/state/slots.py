"""Semantic-slot primitives used by typed NLP channel fusion.

The decomposition and gated cross-attention are self-contained PyTorch utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SlotEncoding:
    slots: torch.Tensor
    slot_weights: torch.Tensor
    token_assignments: torch.Tensor
    slot_activations: torch.Tensor
    active_slot_mask: torch.Tensor


@dataclass(frozen=True)
class CrossAttentionPath:
    attended: torch.Tensor
    gate: torch.Tensor
    attention: torch.Tensor


def deterministic_projection(in_dim: int, out_dim: int, device, dtype, salt: float = 0.0) -> torch.Tensor:
    idx = torch.arange(int(in_dim) * int(out_dim), device=device, dtype=dtype).reshape(int(in_dim), int(out_dim))
    return torch.sin(idx * 0.173 + float(salt)) / math.sqrt(max(1, int(in_dim)))


def soft_slot_decompose(
    tokens: torch.Tensor,
    *,
    num_slots: int = 16,
    activation_temperature: float = 1.0,
    activation_threshold: float = 0.10,
    minimum_visible_slots: int = 1,
) -> SlotEncoding:
    """Decompose ``[B,N,D]`` tokens into an adaptive set of semantic slots."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
    batch, token_count, hidden_dim = tokens.shape
    device = tokens.device
    dtype = tokens.dtype
    num_slots = int(num_slots)
    if token_count == 0 or hidden_dim == 0:
        zeros = torch.zeros(batch, num_slots, device=device, dtype=dtype)
        return SlotEncoding(
            torch.zeros(batch, num_slots, hidden_dim, device=device, dtype=dtype),
            zeros,
            torch.zeros(batch, token_count, num_slots, device=device, dtype=dtype),
            zeros,
            zeros,
        )

    slot_queries = deterministic_projection(hidden_dim, num_slots, device, dtype, salt=0.41)
    token_assignments = torch.softmax(tokens @ slot_queries, dim=-1)
    slot_mass = token_assignments.sum(dim=1).clamp_min(1.0e-6)
    normalized_assignments = token_assignments / slot_mass[:, None, :]
    projected = torch.tanh(tokens @ deterministic_projection(hidden_dim, hidden_dim, device, dtype, salt=1.13))
    slots = torch.bmm(normalized_assignments.transpose(1, 2), projected)
    slot_weights = slot_mass / slot_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    mass_prior = torch.log(slot_weights.clamp_min(1.0e-6) * num_slots).clamp(-3.0, 3.0)
    activation_query = deterministic_projection(hidden_dim, num_slots, device, dtype, salt=2.31)
    slot_activations = torch.sigmoid(
        (tokens.mean(dim=1) @ activation_query + mass_prior) / max(float(activation_temperature), 1.0e-6)
    )
    active_slot_mask = (slot_activations >= float(activation_threshold)).to(dtype)
    if minimum_visible_slots > 0:
        visible = active_slot_mask.sum(dim=1)
        top = torch.topk(slot_activations, k=min(int(minimum_visible_slots), num_slots), dim=1).indices
        forced = torch.zeros_like(active_slot_mask).scatter(1, top, 1.0)
        active_slot_mask = torch.where(visible[:, None] < minimum_visible_slots, forced, active_slot_mask)
    slots = slots * slot_activations.unsqueeze(-1) * active_slot_mask.unsqueeze(-1)
    return SlotEncoding(slots, slot_weights, token_assignments, slot_activations, active_slot_mask)


def gated_cross_attention(
    target: SlotEncoding,
    source: SlotEncoding,
    bias: torch.Tensor | None = None,
) -> CrossAttentionPath:
    """Gated slot-to-slot attention with active-source masking."""

    query = F.normalize(target.slots, dim=-1)
    key = F.normalize(source.slots, dim=-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(max(1, target.slots.shape[-1]))
    if bias is not None and bias.shape == scores.shape:
        scores = scores + bias.to(device=scores.device, dtype=scores.dtype)
    source_mask = source.active_slot_mask[:, None, :]
    scores = scores.masked_fill(source_mask <= 0.0, -1.0e4)
    attention = torch.softmax(scores, dim=-1) * source_mask
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    attended = torch.bmm(attention, source.slots)
    gate_logit = (query * F.normalize(attended + 1.0e-6, dim=-1)).sum(dim=-1).mean()
    gate = torch.sigmoid(gate_logit)
    return CrossAttentionPath(gate * attended * target.active_slot_mask.unsqueeze(-1), gate, attention)
