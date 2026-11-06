"""Prompt-conditioned causal-prefix transformer for typed AST actions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from floorflow.nlp_flow.ast_structure_blueprint import ASTStructureBlueprint
from floorflow.nlp_flow.efficient_output_heads import (
    FactorizedClassHead,
    apply_legal_candidate_mask,
    rebase_residual_budget_logits,
)
from floorflow.nlp_flow.prefix_policy_data import PrefixPolicyBatch
from floorflow.nlp_flow.real_corpus_policy_data import (
    MAX_LEXEME,
    MAX_NODES,
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    PROMPT_TOKEN_BUCKETS,
    REAL_ACTIONS,
    STATE_FEATURES,
)


class PrefixTreePolicy(nn.Module):
    def __init__(
        self,
        vocabulary: Mapping[str, object],
        *,
        hidden_size: int = 128,
        layers: int = 4,
        heads: int = 8,
        max_steps: int = 768,
        blueprint_conditioning: bool = True,
        node_kind_rank: int | None = None,
        budget_residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.vocabulary = dict(vocabulary)
        self.hidden_size = hidden_size
        self.blueprint_conditioning = blueprint_conditioning
        self.node_kind_rank = node_kind_rank
        self.budget_residual = budget_residual
        self.prompt_token_embedding = nn.Embedding(
            PROMPT_TOKEN_BUCKETS + 1,
            hidden_size,
            padding_idx=0,
        )
        self.prompt_position_embedding = nn.Embedding(
            MAX_PROMPT_TOKENS,
            hidden_size,
        )
        prompt_layer = nn.TransformerEncoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(
            prompt_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.prompt_norm = nn.RMSNorm(hidden_size)
        self.blueprint = (
            ASTStructureBlueprint(
                len(vocabulary["node_ids"]),
                len(vocabulary["edge_ids"]),
                hidden_size=hidden_size,
                heads=heads,
                max_steps=MAX_NODES,
            )
            if blueprint_conditioning
            else None
        )

        self.previous_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocabulary[key]) + 1, hidden_size)
                for key in (
                    "action_ids",
                    "node_ids",
                    "edge_ids",
                    "class_ids",
                    "lexeme_ids",
                    "gap_ids",
                )
            ]
        )
        self.history_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocabulary[key]) + 1, hidden_size)
                for key in (
                    "action_ids",
                    "node_ids",
                    "edge_ids",
                    "class_ids",
                    "lexeme_ids",
                    "gap_ids",
                )
            ]
        )
        self.state_projection = nn.Linear(STATE_FEATURES, hidden_size)
        self.context_kind_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.context_edge_embedding = nn.Embedding(
            len(vocabulary["edge_ids"]) + 1,
            hidden_size,
        )
        self.start_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.step_position_embedding = nn.Embedding(max_steps, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )
        self.decoder_norm = nn.RMSNorm(hidden_size)

        self.prompt_char_embedding = nn.Embedding(
            len(vocabulary["chars"]) + 1,
            24,
        )
        self.prompt_char_position = nn.Embedding(MAX_PROMPT, hidden_size)
        self.prompt_char_projection = nn.Linear(24, hidden_size)
        self.copy_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.parent_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_node_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.candidate_scope_embedding = nn.Embedding(
            len(vocabulary["scope_ids"]) + 1,
            hidden_size,
        )
        self.candidate_position_embedding = nn.Embedding(MAX_NODES, hidden_size)
        self.candidate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(hidden_size, len(REAL_ACTIONS)),
                "node_kind": (
                    nn.Linear(hidden_size, len(vocabulary["node_ids"]) + 1)
                    if node_kind_rank is None
                    else FactorizedClassHead(
                        hidden_size,
                        len(vocabulary["node_ids"]) + 1,
                        node_kind_rank,
                    )
                ),
                "edge_field": nn.Linear(hidden_size, len(vocabulary["edge_ids"]) + 1),
                "leaf_node": nn.Linear(hidden_size, MAX_NODES),
                "leaf_class": nn.Linear(hidden_size, len(vocabulary["class_ids"]) + 1),
                "copy": nn.Linear(hidden_size, 1),
                "copy_length": nn.Linear(hidden_size, MAX_LEXEME + 1),
                "lexical": nn.Linear(hidden_size, len(vocabulary["lexeme_ids"]) + 1),
                "gap": nn.Linear(hidden_size, len(vocabulary["gap_ids"]) + 1),
                "node_budget": nn.Linear(hidden_size, MAX_NODES + 1),
                "viability": nn.Linear(hidden_size, 1),
            }
        )

    def encode_prompt(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        chars: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        prompt = (
            self.prompt_token_embedding(token_ids)
            + self.prompt_position_embedding(positions)[None]
        )
        memory = self.prompt_norm(
            self.prompt_encoder(
                prompt,
                src_key_padding_mask=~token_mask,
            )
        )
        weights = token_mask.to(memory.dtype)
        pooled = (memory * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        char_positions = torch.arange(chars.shape[1], device=chars.device)
        char_states = (
            self.prompt_char_projection(self.prompt_char_embedding(chars))
            + self.prompt_char_position(char_positions)[None]
        )
        return memory, pooled, char_states

    def forward(
        self,
        batch: PrefixPolicyBatch,
        *,
        steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        sequence_steps = batch.action.shape[1] if steps is None else steps
        sequence_steps = min(sequence_steps, batch.action.shape[1])
        memory, pooled, char_states = self.encode_prompt(
            batch.prompt_token_ids,
            batch.prompt_token_mask,
            batch.prompts,
            batch.prompt_masks,
        )
        blueprint = (
            self.blueprint(memory, batch.prompt_token_mask)
            if self.blueprint is not None
            else None
        )
        decoder_memory = (
            torch.cat((memory, blueprint["memory"]), dim=1)
            if blueprint is not None
            else memory
        )
        decoder_memory_mask = (
            torch.cat(
                (
                    batch.prompt_token_mask,
                    torch.ones(
                        blueprint["memory"].shape[:2],
                        dtype=torch.bool,
                        device=memory.device,
                    ),
                ),
                dim=1,
            )
            if blueprint is not None
            else batch.prompt_token_mask
        )
        previous = sum(
            embedding(batch.previous[key][:, :sequence_steps])
            for embedding, key in zip(
                self.previous_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        action_history = sum(
            embedding(batch.history[key][:, :sequence_steps]).mean(dim=-2)
            for embedding, key in zip(
                self.history_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        context = (
            self.context_kind_embedding(batch.context_kind[:, :sequence_steps])
            + self.context_edge_embedding(batch.context_edge[:, :sequence_steps])
        )
        positions = torch.arange(sequence_steps, device=batch.action.device)
        target = (
            previous
            + action_history
            + context
            + self.state_projection(batch.state_features[:, :sequence_steps])
            + self.step_position_embedding(positions)[None]
        )
        target = target + self.start_embedding[None, None, :] * positions.eq(0)[
            None,
            :,
            None,
        ]
        causal = torch.triu(
            torch.ones(
                sequence_steps,
                sequence_steps,
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder_norm(
            self.decoder(
                target,
                decoder_memory,
                tgt_mask=causal,
                tgt_key_padding_mask=~batch.step_mask[:, :sequence_steps],
                memory_key_padding_mask=~decoder_memory_mask,
            )
        )
        node_context = (
            self.context_kind_embedding(batch.context_kind[:, :sequence_steps])
            + self.context_edge_embedding(batch.context_edge[:, :sequence_steps])
        )
        output = {
            name: head(hidden)
            for name, head in self.heads.items()
            if name != "node_kind"
        }
        node_logits = self.heads["node_kind"](hidden + node_context)
        if self.node_kind_rank is not None and batch.node_kind_legal_mask is not None:
            active = batch.step_mask[:, :sequence_steps] & batch.action[:, :sequence_steps].eq(
                REAL_ACTIONS.index("open_node")
            )
            node_logits = apply_legal_candidate_mask(
                node_logits,
                batch.node_kind_legal_mask[:, :sequence_steps],
                active,
                batch.node_kind[:, :sequence_steps],
            )
        output["node_kind"] = node_logits
        candidate_state = self.candidate_projection(
            self.candidate_node_embedding(
                batch.candidate_kind[:, :sequence_steps]
            )
            + self.candidate_scope_embedding(
                batch.candidate_scope[:, :sequence_steps]
            )
            + self.candidate_position_embedding(
                batch.candidate_position[:, :sequence_steps]
            )
        )
        output["parent"] = torch.einsum(
            "btd,btcd->btc",
            self.parent_query(hidden),
            candidate_state,
        ).masked_fill(
            ~batch.candidate_mask[:, :sequence_steps],
            -torch.inf,
        )
        copy_scores = torch.einsum(
            "btd,bld->btl",
            self.copy_query(hidden),
            char_states,
        )
        output["copy_start"] = copy_scores.masked_fill(
            ~batch.prompt_masks[:, None, :],
            -torch.inf,
        )
        budget_logits = output["node_budget"][:, 0]
        if self.budget_residual:
            if batch.node_budget_minimum is None:
                raise ValueError("residual budget head requires certified per-example minimums")
            budget_logits = rebase_residual_budget_logits(
                budget_logits,
                batch.node_budget_minimum,
                batch.node_budget,
            )
        output["node_budget"] = budget_logits
        output["viability"] = output["viability"].squeeze(-1)
        output["pooled_prompt"] = pooled
        if blueprint is not None:
            output.update(
                {
                    f"blueprint_{name}": value
                    for name, value in blueprint.items()
                    if name != "memory"
                }
            )
            output["blueprint_memory"] = blueprint["memory"]
        return output


def train_prefix_tree_policy(*args, **kwargs):
    """Lazy compatibility wrapper for the separate training module."""
    from floorflow.nlp_flow.prefix_tree_training import train_prefix_tree_policy as train

    return train(*args, **kwargs)


def evaluate_prefix_policy(*args, **kwargs):
    """Lazy compatibility wrapper for the separate evaluation module."""
    from floorflow.nlp_flow.prefix_tree_training import evaluate_prefix_policy as evaluate

    return evaluate(*args, **kwargs)


def _loss(*args, **kwargs):
    """Lazy compatibility wrapper for focused objective tests and callers."""
    from floorflow.nlp_flow.prefix_tree_training import _loss as training_loss

    return training_loss(*args, **kwargs)


__all__ = ["PrefixTreePolicy", "evaluate_prefix_policy", "train_prefix_tree_policy"]
