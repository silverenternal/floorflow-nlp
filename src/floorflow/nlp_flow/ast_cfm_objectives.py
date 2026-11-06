"""Joint flow and categorical objectives for Stage-A AST generation."""

from __future__ import annotations

import torch

from floorflow.nlp_flow.ast_cfm_data import (
    ASTFlowOutput,
    ASTObjectiveLosses,
    ASTTensorBatch,
)
from floorflow.nlp_flow.ast_token_decoder import source_token_loss


def ast_cfm_objective_losses(
    output: ASTFlowOutput,
    target_state: torch.Tensor,
    noise: torch.Tensor,
    batch: ASTTensorBatch,
    *,
    categorical_output: ASTFlowOutput | None = None,
    weights: dict[str, float] | None = None,
) -> ASTObjectiveLosses:
    local = batch.to(output.velocity.device)
    categories = output if categorical_output is None else categorical_output
    mask = local.node_mask[..., None]
    denominator = mask.sum().clamp_min(1)
    velocity = (
        (output.velocity - (target_state - noise)).square() * mask
    ).sum() / denominator
    endpoint = ((output.endpoint - target_state).square() * mask).sum() / denominator
    node_type = torch.nn.functional.cross_entropy(
        categories.node_type_logits[local.node_mask],
        local.node_types[local.node_mask],
    )
    nonroot = local.node_mask & local.parent_indices.ge(0)
    edge_type = torch.nn.functional.cross_entropy(
        categories.edge_type_logits[nonroot],
        local.edge_fields[nonroot],
    )
    lexical = torch.nn.functional.cross_entropy(
        categories.lexical_logits[local.node_mask],
        local.lexemes[local.node_mask],
    )
    gap = torch.nn.functional.cross_entropy(
        categories.gap_logits[local.node_mask],
        local.gaps[local.node_mask],
    )
    level = torch.nn.functional.cross_entropy(
        categories.level_logits[local.node_mask],
        local.levels[local.node_mask],
    )
    parent = torch.nn.functional.cross_entropy(
        categories.parent_logits[nonroot],
        local.parent_indices[nonroot] + 1,
    )
    root_target = local.node_mask & local.parent_indices.lt(0)
    root = torch.nn.functional.binary_cross_entropy_with_logits(
        categories.root_logits[local.node_mask],
        root_target[local.node_mask].to(categories.root_logits.dtype),
    )
    node_count = torch.nn.functional.cross_entropy(
        categories.node_count_logits,
        local.node_mask.sum(dim=1).long(),
    )
    source_length = torch.nn.functional.cross_entropy(
        categories.source_length_logits,
        local.source_token_mask.sum(dim=1).long(),
    )
    remaining = (
        (categories.flow_endpoint - target_state).square() * mask
    ).sum(dim=(1, 2)) / local.node_mask.sum(dim=1).clamp_min(1)
    variance = (2.0 * categories.uncertainty_log_std).exp()
    uncertainty = (
        remaining / variance.clamp_min(1.0e-8)
        + 2.0 * categories.uncertainty_log_std
    ).mean()
    template = torch.nn.functional.cross_entropy(
        categories.template_logits,
        local.template_ids,
    )
    parseability = torch.nn.functional.binary_cross_entropy_with_logits(
        categories.parseability_logits,
        local.parseability_targets.to(categories.parseability_logits),
    )
    source_tokens = (
        categories.endpoint.new_zeros(())
        if categories.source_token_logits is None
        else source_token_loss(
            categories.source_token_logits,
            local.source_token_ids[:, 1:],
            local.source_token_mask[:, 1:],
        )
    )
    source_tokens_free = (
        categories.endpoint.new_zeros(())
        if categories.source_token_free_logits is None
        else source_token_loss(
            categories.source_token_free_logits,
            local.source_token_ids[:, 1:],
            local.source_token_mask[:, 1:],
        )
    )
    source_tokens_teacher = (
        categories.endpoint.new_zeros(())
        if categories.source_token_teacher_logits is None
        else source_token_loss(
            categories.source_token_teacher_logits,
            local.source_token_ids[:, 1:],
            local.source_token_mask[:, 1:],
        )
    )
    defaults = {
        "velocity": 1.0,
        "endpoint": 0.5,
        "node_type": 1.0,
        "edge_type": 1.0,
        "lexical": 1.0,
        "gap": 1.0,
        "level": 0.5,
        "parent": 1.0,
        "root": 1.0,
        "node_count": 0.5,
        "source_length": 0.5,
        "uncertainty": 0.1,
        "template": 1.0,
        "parseability": 0.25,
        "source_tokens": 1.0,
        "source_tokens_free": 1.0,
        "source_tokens_teacher": 1.0,
    }
    active = defaults if weights is None else {**defaults, **weights}
    total = (
        velocity * active["velocity"]
        + endpoint * active["endpoint"]
        + node_type * active["node_type"]
        + edge_type * active["edge_type"]
        + lexical * active["lexical"]
        + gap * active["gap"]
        + level * active["level"]
        + parent * active["parent"]
        + root * active["root"]
        + node_count * active["node_count"]
        + source_length * active["source_length"]
        + uncertainty * active["uncertainty"]
        + template * active["template"]
        + parseability * active["parseability"]
        + source_tokens * active["source_tokens"]
        + source_tokens_free * active["source_tokens_free"]
        + source_tokens_teacher * active["source_tokens_teacher"]
    )
    return ASTObjectiveLosses(
        total=total,
        velocity=velocity,
        endpoint=endpoint,
        node_type=node_type,
        edge_type=edge_type,
        lexical=lexical,
        gap=gap,
        level=level,
        parent=parent,
        root=root,
        node_count=node_count,
        source_length=source_length,
        uncertainty=uncertainty,
        template=template,
        parseability=parseability,
        source_tokens=source_tokens,
        source_tokens_free=source_tokens_free,
        source_tokens_teacher=source_tokens_teacher,
    )


__all__ = ["ast_cfm_objective_losses"]
