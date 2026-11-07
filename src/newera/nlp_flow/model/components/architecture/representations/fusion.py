"""Typed channel fusion with low-rank cross-attention and sparse MiTA routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from newera.nlp_flow.model.components.architecture.representations.channels import TypedChannelMemory
from newera.nlp_flow.model.components.architecture.routing.official_mita_route import OfficialMiTARouteAttention
from newera.nlp_flow.model.components.architecture.state.slots import gated_cross_attention, soft_slot_decompose


@dataclass(frozen=True)
class TypedFusionOutput:
    hidden_states: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class CompactTypedCrossAttention(nn.Module):
    """Low-rank relation-biased cross-attention over five channel tokens."""

    def __init__(self, hidden_size: int, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("fusion hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.q = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.type_bias = nn.Parameter(torch.zeros(5))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        channel_count = memory.shape[1]
        query = self.q(hidden).reshape(batch, sequence, self.heads, self.head_dim)
        key = self.k(memory).reshape(batch, channel_count, self.heads, self.head_dim)
        value = self.v(memory).reshape(batch, channel_count, self.heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.einsum("bhqd,bhkd->bhqk", query, key)
        scores = scores / self.head_dim**0.5
        scores = scores + self.type_bias[None, None, None, :channel_count]
        attended = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bhqk,bhkd->bhqd", attended, value)
        mixed = mixed.transpose(1, 2).reshape(batch, sequence, -1)
        return self.gate.to(hidden.dtype) * self.out(mixed)


class CompactMiTARouter(nn.Module):
    """Sparse top-k channel routing in a low-dimensional MiTA space."""

    def __init__(self, hidden_size: int, router_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.state = nn.Linear(hidden_size, router_dim, bias=False)
        self.memory = nn.Linear(hidden_size, router_dim, bias=False)
        self.out = nn.Linear(router_dim, hidden_size, bias=False)
        self.route = OfficialMiTARouteAttention(
            router_dim,
            num_heads=heads,
            router_count=5,
            top_k=3,
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, diagnostics = self.route(
            self.state(hidden),
            self.memory(memory),
        )
        self.last_diagnostics = diagnostics
        return self.gate.to(hidden.dtype) * self.out(attended)


class TypedSlotMixer(nn.Module):
    """Bidirectional slot exchange with zero-gated token feedback."""

    def __init__(self, hidden_size: int, num_slots: int = 8) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_slots = int(num_slots)
        self.gate = nn.Parameter(torch.zeros(1))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        hidden_slots = soft_slot_decompose(hidden, num_slots=self.num_slots)
        memory_slots = soft_slot_decompose(memory, num_slots=self.num_slots)
        hidden_from_memory = gated_cross_attention(hidden_slots, memory_slots)
        memory_from_hidden = gated_cross_attention(memory_slots, hidden_slots)
        updated_hidden = hidden_slots.slots + 0.5 * hidden_from_memory.attended
        updated_memory = memory_slots.slots + 0.5 * memory_from_hidden.attended
        hidden_feedback = hidden_slots.slots + 0.5 * updated_hidden
        delta = torch.einsum(
            "bnk,bkd->bnd",
            hidden_slots.token_assignments,
            hidden_feedback,
        ) - torch.einsum(
            "bnk,bkd->bnd",
            hidden_slots.token_assignments,
            hidden_slots.slots,
        )
        self.last_diagnostics = {
            "hidden_to_memory_gate": hidden_from_memory.gate,
            "memory_to_hidden_gate": memory_from_hidden.gate,
            "active_hidden_slots": hidden_slots.active_slot_mask.sum(dim=1).float().mean(),
            "active_memory_slots": memory_slots.active_slot_mask.sum(dim=1).float().mean(),
            "refined_memory_norm": updated_memory.square().mean(),
        }
        return self.gate.to(hidden.dtype) * delta


class TypedChannelFusion(nn.Module):
    """Compose typed attention, MiTA routing, and slot feedback."""

    def __init__(self, hidden_size: int, fusion_dim: int = 64, heads: int = 4, slots: int = 8) -> None:
        super().__init__()
        self.cross_attention = CompactTypedCrossAttention(hidden_size, fusion_dim, heads)
        self.router = CompactMiTARouter(hidden_size, fusion_dim, heads)
        self.slot_mixer = TypedSlotMixer(hidden_size, slots)

    def forward(self, hidden: torch.Tensor, memory: TypedChannelMemory) -> TypedFusionOutput:
        typed = self.cross_attention(hidden, memory.global_memory)
        sparse = self.router(hidden, memory.global_memory)
        slots = self.slot_mixer(hidden, memory.global_memory)
        updated = hidden + typed + sparse + slots
        diagnostics = dict(memory.diagnostics)
        diagnostics.update(
            {
                f"mita_{name}": value
                for name, value in self.router.last_diagnostics.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
        )
        diagnostics.update(
            {
                f"slots_{name}": value
                for name, value in self.slot_mixer.last_diagnostics.items()
            }
        )
        return TypedFusionOutput(updated, diagnostics)


__all__ = [
    "CompactMiTARouter",
    "CompactTypedCrossAttention",
    "TypedChannelFusion",
    "TypedFusionOutput",
    "TypedSlotMixer",
]
