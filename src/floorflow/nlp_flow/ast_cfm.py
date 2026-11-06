"""Hierarchical Stage-A conditional flow matching for canonical AST graphs."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from floorflow.nlp_flow.ast_cfm_data import (
    AST_LEVELS,
    ASTCodecVocabulary,
    ASTFlowOutput,
    ASTObjectiveLosses,
    ASTTensorBatch,
    collate_ast_batches,
    encode_ast_graph,
    prompt_context,
)
from floorflow.nlp_flow.ast_cfm_decoding import (
    decode_generated_source,
    decode_generated_template,
    decode_generated_typed_tree,
    validate_generated_ast,
)
from floorflow.nlp_flow.ast_cfm_objectives import ast_cfm_objective_losses
from floorflow.nlp_flow.ast_cfm_runtime import (
    embed_ast_target,
    placeholder_ast_batch,
    sample_target_free,
)
from floorflow.nlp_flow.ast_generation import (
    predict_generated_topology_batch,
    source_token_plan,
)
from floorflow.nlp_flow.ast_token_decoder import (
    ASTTokenEndpointDecoder,
)


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must be [batch]")
        half = self.hidden_size // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(1, half - 1)
        )
        phases = time[:, None] * frequencies[None]
        values = torch.cat([phases.sin(), phases.cos()], dim=-1)
        if values.shape[-1] < self.hidden_size:
            values = torch.nn.functional.pad(values, (0, 1))
        return values


class ASTStageAFlow(nn.Module):
    """Conditional flow over typed AST slots with hierarchical objective heads."""

    def __init__(
        self,
        vocabulary: ASTCodecVocabulary,
        *,
        hidden_size: int = 64,
        num_heads: int = 4,
        max_nodes: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.vocabulary = vocabulary
        self.hidden_size = int(hidden_size)
        self.max_nodes = int(max_nodes)
        self.node_embedding = nn.Embedding(len(vocabulary.node_types), hidden_size)
        self.edge_embedding = nn.Embedding(len(vocabulary.edge_fields), hidden_size)
        self.lexical_embedding = nn.Embedding(vocabulary.num_lexemes, hidden_size)
        self.gap_embedding = nn.Embedding(vocabulary.num_gaps, hidden_size)
        self.level_embedding = nn.Embedding(4, hidden_size)
        self.slot_embedding = nn.Embedding(max_nodes, hidden_size)
        self.prompt_projection = nn.Linear(hidden_size, hidden_size)
        self.time_projection = nn.Sequential(
            TimeEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size) for _ in range(layers)]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size) for _ in range(layers)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.SiLU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(layers)
            ]
        )
        self.velocity_head = nn.Linear(hidden_size, hidden_size)
        self.categorical_conditioner = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.parent_context_conditioner = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.categorical_gate = nn.Parameter(torch.tensor(0.1))
        self.node_type_head = nn.Linear(hidden_size, len(vocabulary.node_types))
        self.edge_type_head = nn.Linear(hidden_size, len(vocabulary.edge_fields))
        self.lexical_head = nn.Linear(hidden_size, vocabulary.num_lexemes)
        self.gap_head = nn.Linear(hidden_size, vocabulary.num_gaps)
        self.level_head = nn.Linear(hidden_size, 3)
        self.parent_query = nn.Linear(hidden_size, hidden_size // num_heads, bias=False)
        self.parent_key = nn.Linear(hidden_size, hidden_size // num_heads, bias=False)
        self.root_head = nn.Linear(hidden_size, 1)
        self.node_count_head = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, max_nodes + 1),
        )
        self.source_length_head = nn.Linear(
            hidden_size,
            max_nodes * 2 + 2,
        )
        self.uncertainty_head = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.template_head = nn.Linear(hidden_size, len(vocabulary.templates))
        self.parseability_head = nn.Linear(hidden_size, 1)
        self.token_endpoint = ASTTokenEndpointDecoder(
            vocabulary,
            hidden_size,
            max_tokens=max_nodes * 2 + 1,
            layers=layers,
            heads=num_heads,
        )

    def embed_target(self, batch: ASTTensorBatch, device: torch.device | str) -> torch.Tensor:
        return embed_ast_target(self, batch, device)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        batch: ASTTensorBatch,
        *,
        parent_indices: torch.Tensor | None = None,
        source_token_inputs: torch.Tensor | None = None,
        source_token_node_ids: torch.Tensor | None = None,
        source_token_phase_ids: torch.Tensor | None = None,
        source_token_teacher_state: torch.Tensor | None = None,
    ) -> ASTFlowOutput:
        local = batch.to(state.device)
        if state.shape != (
            local.node_mask.shape[0],
            local.node_mask.shape[1],
            self.hidden_size,
        ):
            raise ValueError("state must be [batch, nodes, hidden]")
        if context.shape != (state.shape[0], self.hidden_size):
            raise ValueError("context must be [batch, hidden]")
        if time.shape != (state.shape[0],):
            raise ValueError("time must be [batch]")
        parent_indices = local.parent_indices if parent_indices is None else parent_indices
        if parent_indices.shape != local.parent_indices.shape:
            raise ValueError("parent_indices must be [batch, nodes]")
        if parent_indices[local.node_mask].lt(-1).any():
            raise ValueError("parent indices must be -1 for roots or non-negative")
        if parent_indices[local.node_mask].ge(local.node_mask.shape[1]).any():
            raise ValueError("parent index is outside the AST slot range")

        slots = torch.arange(state.shape[1], device=state.device)
        hidden = self.input_projection(
            state
            + self.slot_embedding(slots)[None]
            + self.prompt_projection(context)[:, None]
            + self.time_projection(time)[:, None]
        )
        for attention, attention_norm, ffn_norm, ffn in zip(
            self.attention_layers,
            self.attention_norms,
            self.ffn_norms,
            self.ffns,
            strict=True,
        ):
            normalized = attention_norm(hidden)
            attended, _weights = attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~local.node_mask,
                need_weights=False,
            )
            hidden = hidden + attended
            hidden = hidden + ffn(ffn_norm(hidden))
        pooled = (hidden * local.node_mask[..., None]).sum(dim=1) / local.node_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1)
        velocity = self.velocity_head(hidden)
        base_endpoint = state + (1.0 - time)[:, None, None] * velocity
        node_type_logits = self.node_type_head(base_endpoint)
        edge_type_logits = self.edge_type_head(base_endpoint)
        lexical_logits = self.lexical_head(base_endpoint)
        gap_logits = self.gap_head(base_endpoint)
        level_logits = self.level_head(base_endpoint)
        parent_scores = torch.einsum(
            "bnd,bmd->bnm",
            self.parent_query(base_endpoint),
            self.parent_key(base_endpoint),
        ) / math.sqrt(self.parent_query.out_features)
        parent_valid = (
            local.node_mask[:, None, :]
            & ~torch.eye(
                local.node_mask.shape[1],
                dtype=torch.bool,
                device=parent_scores.device,
            )[None]
        )
        parent_valid = parent_valid | ~local.node_mask[..., None]
        parent_probabilities = torch.softmax(
            parent_scores.masked_fill(~parent_valid, -torch.inf),
            dim=-1,
        )
        parent_context = torch.einsum(
            "bnm,bmd->bnd",
            parent_probabilities,
            base_endpoint,
        )
        categorical_state = (
            torch.matmul(
                torch.softmax(node_type_logits, dim=-1),
                self.node_embedding.weight,
            )
            + torch.matmul(
                torch.softmax(edge_type_logits, dim=-1),
                self.edge_embedding.weight,
            )
            + torch.matmul(
                torch.softmax(lexical_logits, dim=-1),
                self.lexical_embedding.weight,
            )
            + torch.matmul(
                torch.softmax(gap_logits, dim=-1),
                self.gap_embedding.weight,
            )
            + torch.matmul(
                torch.softmax(level_logits, dim=-1),
                self.level_embedding.weight[:3],
            )
            + self.parent_context_conditioner(parent_context)
        )
        endpoint = base_endpoint + (
            self.categorical_gate.to(base_endpoint.dtype)
            * self.categorical_conditioner(categorical_state)
        )
        parent_scores = torch.einsum(
            "bnd,bmd->bnm",
            self.parent_query(endpoint),
            self.parent_key(endpoint),
        )
        parent_scores = parent_scores / math.sqrt(self.parent_query.out_features)
        no_parent = torch.zeros(
            (*local.node_mask.shape, 1),
            dtype=parent_scores.dtype,
            device=parent_scores.device,
        )
        parent_logits = torch.cat((no_parent, parent_scores), dim=-1)
        valid_parent = torch.cat(
            (
                torch.ones_like(no_parent, dtype=torch.bool),
                local.node_mask[:, None, :].expand(
                    -1,
                    local.node_mask.shape[1],
                    -1,
                ),
            ),
            dim=-1,
        )
        valid_parent[..., 1:] = (
            valid_parent[..., 1:]
            & ~torch.eye(
                local.node_mask.shape[1],
                dtype=torch.bool,
                device=parent_scores.device,
            )[None]
        )
        parent_logits = parent_logits.masked_fill(~valid_parent, -torch.inf)
        root_logits = self.root_head(endpoint).squeeze(-1)
        source_token_logits = (
            None
            if source_token_inputs is None
            else self.token_endpoint(
                endpoint,
                local.node_mask,
                source_token_inputs.to(state.device),
                (
                    None
                    if source_token_node_ids is None
                    else source_token_node_ids.to(state.device)
                ),
                (
                    None
                    if source_token_phase_ids is None
                    else source_token_phase_ids.to(state.device)
                ),
                self.lexical_head(endpoint),
                self.gap_head(endpoint),
            )
        )
        source_token_free_logits = None
        source_token_teacher_logits = None
        if source_token_logits is not None:
            free_running_ids = source_token_logits.argmax(dim=-1)
            free_running_ids[:, 0] = source_token_inputs[:, 0].to(
                free_running_ids.device
            )
            source_token_free_logits = self.token_endpoint(
                endpoint,
                local.node_mask,
                free_running_ids,
                (
                    None
                    if source_token_node_ids is None
                    else source_token_node_ids.to(state.device)
                ),
                (
                    None
                    if source_token_phase_ids is None
                    else source_token_phase_ids.to(state.device)
                ),
                self.lexical_head(endpoint),
                self.gap_head(endpoint),
            )
            if source_token_teacher_state is not None:
                if source_token_teacher_state.shape != endpoint.shape:
                    raise ValueError(
                        "teacher source state must match AST endpoint shape"
                    )
                source_token_teacher_logits = self.token_endpoint(
                    source_token_teacher_state.to(state.device),
                    local.node_mask,
                    source_token_inputs.to(state.device),
                    (
                        None
                        if source_token_node_ids is None
                        else source_token_node_ids.to(state.device)
                    ),
                    (
                        None
                        if source_token_phase_ids is None
                        else source_token_phase_ids.to(state.device)
                    ),
                    self.lexical_head(source_token_teacher_state.to(state.device)),
                    self.gap_head(source_token_teacher_state.to(state.device)),
                )
        return ASTFlowOutput(
            velocity=velocity,
            flow_endpoint=base_endpoint,
            endpoint=endpoint,
            node_type_logits=self.node_type_head(endpoint),
            edge_type_logits=self.edge_type_head(endpoint),
            lexical_logits=self.lexical_head(endpoint),
            gap_logits=self.gap_head(endpoint),
            level_logits=self.level_head(endpoint),
            parent_logits=parent_logits,
            root_logits=root_logits,
            node_count_logits=self.node_count_head(context),
            source_length_logits=self.source_length_head(context),
            uncertainty_log_std=self.uncertainty_head(pooled)
            .squeeze(-1)
            .clamp(-10.0, 5.0),
            template_logits=self.template_head(pooled),
            parseability_logits=self.parseability_head(pooled).squeeze(-1),
            level_velocities={
                name: velocity * local.level_masks[name][..., None]
                for name in AST_LEVELS
            },
            source_token_logits=source_token_logits,
            source_token_free_logits=source_token_free_logits,
            source_token_teacher_logits=source_token_teacher_logits,
        )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        level_masks: dict[str, torch.Tensor] | None = None,
        parent_indices: torch.Tensor | None = None,
        steps: int | None = 4,
        max_steps: int | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ASTFlowOutput:
        """Generate AST slots from context and masks; no target AST is accepted."""

        batch_size, max_nodes = node_mask.shape
        if steps is not None and steps <= 0:
            raise ValueError("steps must be positive")
        loop_steps = (
            min(32, max(1, max_nodes * 2))
            if steps is None and max_steps is None
            else max_steps
            if steps is None
            else steps
        )
        if loop_steps is None or loop_steps <= 0:
            raise ValueError("adaptive step cap must be positive")
        if context.shape != (batch_size, self.hidden_size):
            raise ValueError("context must be [batch, hidden]")
        if level_masks is not None:
            expected_levels = {"root", "subtree", "leaf"}
            if set(level_masks) != expected_levels:
                raise ValueError("level_masks must define root, subtree, and leaf")
            for name, mask in level_masks.items():
                if mask.shape != node_mask.shape:
                    raise ValueError(f"{name} mask must match node_mask")
            union = (
                level_masks["root"]
                | level_masks["subtree"]
                | level_masks["leaf"]
            )
            if not torch.equal(union, node_mask):
                raise ValueError("root/subtree/leaf masks must partition node_mask")
            overlap = (
                level_masks["root"].to(torch.int8)
                + level_masks["subtree"].to(torch.int8)
                + level_masks["leaf"].to(torch.int8)
            ).gt(1)
            if overlap.any():
                raise ValueError("root/subtree/leaf masks must partition node_mask")
        state = (
            torch.randn(
                batch_size,
                max_nodes,
                self.hidden_size,
                device=context.device,
                dtype=context.dtype,
                generator=generator,
            )
            if noise is None
            else noise.to(context)
        )
        if state.shape != (batch_size, max_nodes, self.hidden_size):
            raise ValueError("noise must be [batch, nodes, hidden]")
        batch = placeholder_ast_batch(
            node_mask=node_mask,
            parent_indices=parent_indices,
            level_masks=level_masks,
        )
        executed_steps = loop_steps
        for step in range(loop_steps):
            time = torch.full(
                (batch_size,),
                step / loop_steps,
                device=context.device,
                dtype=context.dtype,
            )
            output = self.forward(state, time, context, batch)
            if level_masks is None:
                probabilities = torch.softmax(output.level_logits, dim=-1)
                velocity = (
                    output.level_velocities["root"] * probabilities[..., 0:1]
                    + output.level_velocities["subtree"] * probabilities[..., 1:2]
                    + output.level_velocities["leaf"] * probabilities[..., 2:3]
                )
            else:
                velocity = output.velocity
            delta = velocity * node_mask[..., None] / loop_steps
            euler = state + delta
            previous = state
            state = torch.where(node_mask[..., None], euler, state)
            if steps is None:
                delta_rms = (
                    (state - previous).square()
                    * node_mask[..., None]
                ).sum(dim=(1, 2)).div(
                    node_mask.sum(dim=1).clamp_min(1) * self.hidden_size
                ).sqrt()
                uncertainty = output.uncertainty_log_std.exp()
                if torch.all(delta_rms <= uncertainty) and step >= 0:
                    executed_steps = step + 1
                    break
        final = self.forward(
            state,
            torch.ones(batch_size, device=context.device, dtype=context.dtype),
            context,
            batch,
        )
        source_length = (
            torch.softmax(final.source_length_logits, dim=-1)
            * torch.arange(
                final.source_length_logits.shape[-1],
                device=final.source_length_logits.device,
            )
        ).sum(dim=-1).round().long()
        parent_indices, _node_levels = predict_generated_topology_batch(
            final,
            node_mask,
        )
        token_node_ids, token_phase_ids = source_token_plan(
            node_mask,
            parent_indices,
            max_tokens=final.source_token_ids.shape[-1]
            if final.source_token_ids is not None
            else max_nodes * 2 + 1,
        )
        return replace(
            final,
            source_token_ids=self.token_endpoint.sample(
                final.endpoint,
                node_mask,
                source_token_node_ids=token_node_ids,
                source_token_phase_ids=token_phase_ids,
                node_lexical_logits=self.lexical_head(final.endpoint),
                node_gap_logits=self.gap_head(final.endpoint),
                minimum_tokens=source_length,
            ),
            executed_nfe=executed_steps,
        )

    @torch.no_grad()
    def sample_target_free(
        self,
        context: torch.Tensor,
        *,
        steps: int | None = None,
        max_steps: int | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        max_nodes: int | None = None,
    ) -> ASTFlowOutput:
        """Generate an AST node budget and mask from context, never targets."""

        return sample_target_free(
            self,
            context,
            steps=steps,
            max_steps=max_steps,
            noise=noise,
            generator=generator,
            max_nodes=max_nodes,
        )


__all__ = [
    "AST_LEVELS",
    "ASTCodecVocabulary",
    "ASTFlowOutput",
    "ASTObjectiveLosses",
    "ASTStageAFlow",
    "ASTTensorBatch",
    "ast_cfm_objective_losses",
    "collate_ast_batches",
    "decode_generated_source",
    "decode_generated_template",
    "decode_generated_typed_tree",
    "encode_ast_graph",
    "prompt_context",
    "validate_generated_ast",
]
