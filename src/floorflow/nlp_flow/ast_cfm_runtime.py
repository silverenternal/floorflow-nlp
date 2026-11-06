"""Runtime batch assembly for target-free AST flow sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from floorflow.nlp_flow.ast_cfm_data import ASTTensorBatch

if TYPE_CHECKING:
    from floorflow.nlp_flow.ast_cfm import ASTFlowOutput, ASTStageAFlow


def placeholder_ast_batch(
    *,
    node_mask: torch.Tensor,
    parent_indices: torch.Tensor | None,
    level_masks: dict[str, torch.Tensor] | None,
) -> ASTTensorBatch:
    batch_size, max_nodes = node_mask.shape
    device = node_mask.device
    placeholder_levels = {
        "root": node_mask,
        "subtree": torch.zeros_like(node_mask),
        "leaf": torch.zeros_like(node_mask),
    }
    source_size = max_nodes * 2 + 1
    return ASTTensorBatch(
        node_mask=node_mask,
        node_types=torch.zeros_like(node_mask, dtype=torch.long),
        edge_fields=torch.zeros_like(node_mask, dtype=torch.long),
        lexemes=torch.zeros_like(node_mask, dtype=torch.long),
        gaps=torch.zeros_like(node_mask, dtype=torch.long),
        parent_indices=(
            torch.full_like(node_mask, -1, dtype=torch.long)
            if parent_indices is None
            else parent_indices
        ),
        levels=torch.full_like(node_mask, 3, dtype=torch.long),
        level_masks=level_masks or placeholder_levels,
        parseability_targets=torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        template_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        source_token_ids=torch.zeros(
            batch_size,
            source_size,
            dtype=torch.long,
            device=device,
        ),
        source_token_mask=torch.zeros(
            batch_size,
            source_size,
            dtype=torch.bool,
            device=device,
        ),
        source_token_node_ids=torch.zeros(
            batch_size,
            source_size,
            dtype=torch.long,
            device=device,
        ),
        source_token_phase_ids=torch.zeros(
            batch_size,
            source_size,
            dtype=torch.long,
            device=device,
        ),
    )


def embed_ast_target(
    model: ASTStageAFlow,
    batch: ASTTensorBatch,
    device: torch.device | str,
) -> torch.Tensor:
    local = batch.to(device)
    slots = torch.arange(local.node_mask.shape[-1], device=device)
    return (
        model.node_embedding(local.node_types)
        + model.edge_embedding(local.edge_fields)
        + model.lexical_embedding(local.lexemes)
        + model.gap_embedding(local.gaps)
        + model.level_embedding(local.levels)
        + model.slot_embedding(slots)[None]
    )


def sample_target_free(
    model: ASTStageAFlow,
    context: torch.Tensor,
    *,
    steps: int | None,
    max_steps: int | None,
    noise: torch.Tensor | None,
    generator: torch.Generator | None,
    max_nodes: int | None,
) -> ASTFlowOutput:
    if context.ndim != 2 or context.shape[-1] != model.hidden_size:
        raise ValueError("context must be [batch, hidden_size]")
    limit = model.max_nodes if max_nodes is None else int(max_nodes)
    if not 1 <= limit <= model.max_nodes:
        raise ValueError("max_nodes is outside the model capacity")
    budget_logits = model.node_count_head(context)
    counts = budget_logits.argmax(dim=-1).clamp(1, limit)
    node_mask = (
        torch.arange(limit, device=context.device)[None]
        < counts[:, None]
    )
    return model.sample(
        context,
        node_mask,
        steps=steps,
        max_steps=max_steps,
        noise=noise,
        generator=generator,
    )


__all__ = ["embed_ast_target", "placeholder_ast_batch", "sample_target_free"]
