"""Prompt-predicted causal AST blueprints for subtree-conditioned decoding."""

from __future__ import annotations

import torch
from torch import nn


class ASTStructureBlueprint(nn.Module):
    """Predict ordered node, edge and parent slots using prompt context only."""

    def __init__(
        self,
        node_vocab_size: int,
        edge_vocab_size: int,
        *,
        hidden_size: int,
        heads: int,
        max_steps: int,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        if min(node_vocab_size, edge_vocab_size, hidden_size, max_steps, layers) < 1:
            raise ValueError("blueprint dimensions and layer count must be positive")
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.query_position = nn.Parameter(torch.zeros(1, max_steps, hidden_size))
        self.step_embedding = nn.Embedding(max_steps, hidden_size)
        layer = nn.TransformerDecoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.planner = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.RMSNorm(hidden_size)
        self.node_head = nn.Linear(hidden_size, node_vocab_size + 1)
        self.edge_head = nn.Linear(hidden_size, edge_vocab_size + 1)
        self.presence_head = nn.Linear(hidden_size, 1)
        self.parent_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.parent_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.root_parent = nn.Parameter(torch.zeros(hidden_size))
        self.node_embedding = nn.Embedding(node_vocab_size + 1, hidden_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_vocab_size + 1, hidden_size, padding_idx=0)
        self.memory_projection = nn.Linear(hidden_size * 3, hidden_size)
        self.memory_norm = nn.RMSNorm(hidden_size)

    def forward(self, prompt: torch.Tensor, prompt_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, hidden = prompt.shape
        if hidden != self.hidden_size or prompt_mask.shape != prompt.shape[:2]:
            raise ValueError("prompt memory and mask shapes do not match the blueprint")
        positions = torch.arange(self.max_steps, device=prompt.device)
        queries = self.query_position + self.step_embedding(positions)[None]
        queries = queries.expand(batch, -1, -1)
        causal = torch.triu(
            torch.ones(self.max_steps, self.max_steps, dtype=torch.bool, device=prompt.device),
            diagonal=1,
        )
        hidden_states = self.norm(
            self.planner(
                queries,
                prompt,
                tgt_mask=causal,
                memory_key_padding_mask=~prompt_mask,
            )
        )
        node_logits = self.node_head(hidden_states)
        edge_logits = self.edge_head(hidden_states)
        presence_logits = self.presence_head(hidden_states).squeeze(-1)
        root_key = self.root_parent.view(1, 1, -1).expand(batch, 1, -1)
        parent_keys = torch.cat((root_key, self.parent_key(hidden_states)), dim=1)
        parent_logits = torch.einsum("btd,bpd->btp", self.parent_query(hidden_states), parent_keys)
        parent_positions = torch.arange(self.max_steps, device=prompt.device)
        parent_indices = torch.arange(self.max_steps + 1, device=prompt.device)
        parent_mask = parent_indices[None, :] > parent_positions[:, None]
        parent_logits = parent_logits.masked_fill(parent_mask[None], -torch.inf)

        node_values = node_logits.softmax(dim=-1) @ self.node_embedding.weight
        edge_values = edge_logits.softmax(dim=-1) @ self.edge_embedding.weight
        parent_values = parent_logits.softmax(dim=-1) @ parent_keys
        confidence = presence_logits.sigmoid().unsqueeze(-1)
        blueprint_memory = self.memory_norm(
            hidden_states + self.memory_projection(torch.cat((node_values, edge_values, parent_values), dim=-1))
        ) * confidence
        return {
            "memory": blueprint_memory,
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "parent_logits": parent_logits,
            "presence_logits": presence_logits,
            "presence": confidence.squeeze(-1),
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = ["ASTStructureBlueprint", "parameter_count"]
