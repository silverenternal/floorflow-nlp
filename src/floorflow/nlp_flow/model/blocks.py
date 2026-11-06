"""Modern pre-norm decoder blocks with GQA, RoPE, and SwiGLU."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.nlp_flow.model.config import NLPModelConfig

RMSNorm = nn.RMSNorm


class RotaryEmbedding(nn.Module):
    """Precomputed rotary positions with explicit dtype/device handling."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inverse = 1.0 / (
            float(base)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, value: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        length = value.shape[-2]
        if position_ids is None:
            cos = self.cos[:length].to(device=value.device, dtype=value.dtype)
            sin = self.sin[:length].to(device=value.device, dtype=value.dtype)
            cos = cos[None, None]
            sin = sin[None, None]
        else:
            if position_ids.shape != (value.shape[0], length):
                raise ValueError("position_ids must match [batch, sequence]")
            cos = self.cos.to(device=value.device, dtype=value.dtype)[position_ids][:, None]
            sin = self.sin.to(device=value.device, dtype=value.dtype)[position_ids][:, None]
        even = value[..., 0::2]
        odd = value[..., 1::2]
        rotated = torch.stack(
            [even * cos - odd * sin, even * sin + odd * cos],
            dim=-1,
        )
        return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Causal GQA with q/k RMS normalization and scaled dot-product attention."""

    def __init__(self, config: NLPModelConfig, *, causal: bool = True) -> None:
        super().__init__()
        self.causal = bool(causal)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)
        self.dropout = config.dropout

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.q_proj(hidden).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden).reshape(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden).reshape(
            batch, sequence, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        query = self.rope(self.q_norm(query), position_ids)
        key = self.rope(self.k_norm(key), position_ids)
        repeat = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        if self.causal:
            causal = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden.device,
            ).tril()
            mask = causal[None, None]
        else:
            mask = torch.ones(
                1,
                1,
                sequence,
                sequence,
                dtype=torch.bool,
                device=hidden.device,
            )
        if attention_mask is not None:
            if attention_mask.shape != (batch, sequence):
                raise ValueError("attention_mask must be [batch, sequence]")
            key_padding = attention_mask[:, None, None, :].bool()
            mask = mask & key_padding
            active = mask.any(dim=-1, keepdim=True)
            first_key = torch.arange(sequence, device=hidden.device).eq(0)
            mask = mask | (~active & first_key)
            if not mask.any(dim=-1).all():
                raise ValueError("causal attention has no valid key")
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.gate(hidden)) * self.up(hidden)
        return self.dropout(self.down(value))


class CompactDecoderBlock(nn.Module):
    def __init__(self, config: NLPModelConfig, *, causal: bool = True) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config, causal=causal)
        self.norm_mlp = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.dropout(
            self.attention(
                self.norm_attention(hidden),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        )
        return hidden + self.dropout(self.mlp(self.norm_mlp(hidden)))


__all__ = [
    "CompactDecoderBlock",
    "GroupedQueryAttention",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
]
