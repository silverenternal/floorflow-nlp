"""Causal AST-conditioned copy/generate planner with native typed heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

VALUE_HEADS = ("identifier", "literal", "operator", "trivia")


def shift_lexical_history(target_ids: torch.Tensor, *, bos_id: int) -> torch.Tensor:
    """Build prefix-only inputs: decision t sees BOS or target tokens before t."""
    if target_ids.ndim != 2 or bos_id < 0:
        raise ValueError("lexical targets must be [batch, steps] and BOS must be nonnegative")
    history = torch.empty_like(target_ids)
    history[:, 0] = bos_id
    if target_ids.shape[1] > 1:
        history[:, 1:] = target_ids[:, :-1]
    return history


def valid_span_length_mask(
    start: torch.Tensor,
    source_length: torch.Tensor,
    *,
    max_length: int,
) -> torch.Tensor:
    """Return legal variable lengths for half-open Unicode codepoint spans."""
    if max_length < 1 or start.shape != source_length.shape:
        raise ValueError("span start/length shapes must match and max_length must be positive")
    lengths = torch.arange(1, max_length + 1, device=start.device)
    return (start.unsqueeze(-1) >= 0) & (start.unsqueeze(-1) + lengths <= source_length.unsqueeze(-1))


class TypedLexicalPlanner(nn.Module):
    """Predict typed lexical choices from structure, visible evidence and causal history."""

    def __init__(
        self,
        *,
        node_kinds: int,
        leaf_classes: int,
        lexical_tokens: int,
        value_vocab_sizes: Mapping[str, int],
        hidden_size: int = 96,
        heads: int = 4,
        max_steps: int = 256,
        max_span_length: int = 128,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if min(node_kinds, leaf_classes, lexical_tokens, hidden_size, max_steps, max_span_length, layers) < 1:
            raise ValueError("planner dimensions and layer count must be positive")
        if set(value_vocab_sizes) != set(VALUE_HEADS) or any(size < 1 for size in value_vocab_sizes.values()):
            raise ValueError(f"value vocabularies must define exactly {VALUE_HEADS}")
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.max_span_length = max_span_length
        self.node_embedding = nn.Embedding(node_kinds + 1, hidden_size)
        self.history_embedding = nn.Embedding(lexical_tokens + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_steps, hidden_size)
        self.input_projection = nn.Linear(hidden_size * 3, hidden_size)
        layer = nn.TransformerDecoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.RMSNorm(hidden_size)
        self.mode_head = nn.Linear(hidden_size, 2)
        self.class_head = nn.Linear(hidden_size, leaf_classes)
        self.value_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, size) for name, size in value_vocab_sizes.items()}
        )
        self.copy_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.evidence_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.copy_length_head = nn.Linear(hidden_size, max_span_length)

    def forward(
        self,
        structure_states: torch.Tensor,
        node_kind_ids: torch.Tensor,
        history_ids: torch.Tensor,
        prompt_states: torch.Tensor,
        prompt_mask: torch.Tensor,
        scope_evidence_mask: torch.Tensor,
        leaf_class_mask: torch.Tensor,
        value_masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, steps, hidden = structure_states.shape
        if hidden != self.hidden_size or not 1 <= steps <= self.max_steps:
            raise ValueError("structure state shape exceeds the planner contract")
        if node_kind_ids.shape != (batch, steps) or history_ids.shape != (batch, steps):
            raise ValueError("node and committed lexical history shapes must match structure states")
        if prompt_states.shape[0] != batch or prompt_states.shape[-1] != hidden:
            raise ValueError("visible evidence states do not match the structure state shape")
        evidence_count = prompt_states.shape[1]
        if prompt_mask.shape != (batch, evidence_count) or scope_evidence_mask.shape != (batch, steps, evidence_count):
            raise ValueError("evidence visibility and lexical scope masks have invalid shapes")
        if leaf_class_mask.ndim != 3 or leaf_class_mask.shape[:2] != (batch, steps):
            raise ValueError("native leaf-class mask must be [batch, steps, classes]")
        if not leaf_class_mask.any(dim=-1).all():
            raise ValueError("each lexical decision must allow at least one native leaf class")
        if set(value_masks) != set(VALUE_HEADS):
            raise ValueError(f"native value masks must define exactly {VALUE_HEADS}")
        for name, head in self.value_heads.items():
            if value_masks[name].shape != (batch, steps, head.out_features):
                raise ValueError(f"native value mask has an invalid shape for {name}")
        positions = torch.arange(steps, device=structure_states.device)
        query = self.input_projection(
            torch.cat(
                (
                    structure_states,
                    self.node_embedding(node_kind_ids),
                    self.history_embedding(history_ids) + self.position_embedding(positions)[None],
                ),
                dim=-1,
            )
        )
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=query.device), diagonal=1)
        hidden_states = self.norm(
            self.decoder(
                query,
                prompt_states,
                tgt_mask=causal,
                memory_key_padding_mask=~prompt_mask,
            )
        )
        visible_copy = scope_evidence_mask & prompt_mask[:, None, :]
        mode_logits = self.mode_head(hidden_states)
        copy_mode = mode_logits[..., 0].masked_fill(~visible_copy.any(dim=-1), -torch.inf)
        mode_logits = torch.stack((copy_mode, mode_logits[..., 1]), dim=-1)
        class_logits = self.class_head(hidden_states).masked_fill(~leaf_class_mask, -torch.inf)
        output = {
            "mode_logits": mode_logits,
            "leaf_class_logits": class_logits,
            "copy_start_logits": torch.einsum(
                "btd,bed->bte",
                self.copy_query(hidden_states),
                self.evidence_key(prompt_states),
            ).masked_fill(~visible_copy, -torch.inf),
            "copy_length_logits": self.copy_length_head(hidden_states),
        }
        output.update(
            {
                f"{name}_logits": self.value_heads[name](hidden_states).masked_fill(~value_masks[name], -torch.inf)
                for name in VALUE_HEADS
            }
        )
        return output


__all__ = ["VALUE_HEADS", "TypedLexicalPlanner", "shift_lexical_history", "valid_span_length_mask"]
