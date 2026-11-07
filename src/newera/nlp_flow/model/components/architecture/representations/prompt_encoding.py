"""Semantic prompt encoding for AST-first generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SemanticPromptState:
    context: torch.Tensor
    token_states: torch.Tensor
    token_mask: torch.Tensor


class SemanticPromptEncoder(nn.Module):
    """Encode actual prompt tokens instead of hashing the prompt string."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        max_tokens: int = 512,
        layers: int = 2,
        heads: int = 4,
        language_count: int = 4,
        token_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        for value, name in (
            (vocab_size, "vocab_size"),
            (hidden_size, "hidden_size"),
            (max_tokens, "max_tokens"),
            (layers, "layers"),
            (heads, "heads"),
            (language_count, "language_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = int(hidden_size)
        if token_embedding is not None:
            if (
                token_embedding.num_embeddings != vocab_size
                or token_embedding.embedding_dim != hidden_size
            ):
                raise ValueError("shared token embedding dimensions do not match")
            self.token_embedding = token_embedding
        else:
            self.token_embedding = nn.Embedding(vocab_size, hidden_size)
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        self.language_embedding = nn.Embedding(language_count, hidden_size)
        self.position_embedding = nn.Embedding(max_tokens, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.RMSNorm(hidden_size)
        self.pool_query = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> SemanticPromptState:
        if input_ids.ndim != 2:
            raise ValueError("prompt token ids must be [batch, sequence]")
        batch, sequence = input_ids.shape
        positions = torch.arange(sequence, device=input_ids.device)
        if sequence > self.position_embedding.num_embeddings:
            raise ValueError("prompt exceeds the semantic encoder length")
        mask = (
            torch.ones(batch, sequence, dtype=torch.bool, device=input_ids.device)
            if attention_mask is None
            else attention_mask.to(input_ids.device, torch.bool)
        )
        if mask.shape != input_ids.shape or not mask.any(dim=1).all():
            raise ValueError("attention mask must retain at least one prompt token")
        language = (
            torch.zeros(batch, dtype=torch.long, device=input_ids.device)
            if language_ids is None
            else language_ids.to(input_ids.device, torch.long)
        )
        if language.shape != (batch,):
            raise ValueError("language_ids must be [batch]")
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None]
            + self.language_embedding(language)[:, None]
        )
        hidden = self.encoder(
            hidden,
            src_key_padding_mask=~mask,
        )
        hidden = self.norm(hidden)
        scores = torch.einsum("bnd,d->bn", hidden, self.pool_query)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = (hidden * weights[..., None]).sum(dim=1)
        return SemanticPromptState(
            context=self.context_projection(pooled),
            token_states=hidden,
            token_mask=mask,
        )


__all__ = ["SemanticPromptEncoder", "SemanticPromptState"]
