"""Padded causal-prefix batches for the transformer Seq2Tree policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_policy import encode_blueprint_targets
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import (
    ACTION_HISTORY,
    MAX_LEXEME,
    MAX_NODES,
    MAX_PARENT_CANDIDATES,
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    REAL_ACTIONS,
    STATE_FEATURES,
    RealPolicyBatch,
    action_from_record,
    encode_action_rows,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import OpenNodeAction, TreeActionDecoder


@dataclass(frozen=True)
class PrefixPolicyBatch:
    prompts: torch.Tensor
    prompt_masks: torch.Tensor
    prompt_token_ids: torch.Tensor
    prompt_token_mask: torch.Tensor
    node_budget: torch.Tensor
    step_mask: torch.Tensor
    state_features: torch.Tensor
    context_kind: torch.Tensor
    context_edge: torch.Tensor
    candidate_kind: torch.Tensor
    candidate_scope: torch.Tensor
    candidate_position: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_target: torch.Tensor
    history: dict[str, torch.Tensor]
    previous: dict[str, torch.Tensor]
    action: torch.Tensor
    node_kind: torch.Tensor
    edge_field: torch.Tensor
    parent: torch.Tensor
    leaf_node: torch.Tensor
    leaf_class: torch.Tensor
    copy_label: torch.Tensor
    copy_start: torch.Tensor
    copy_length: torch.Tensor
    lexical: torch.Tensor
    gap: torch.Tensor
    blueprint_node: torch.Tensor | None = None
    blueprint_edge: torch.Tensor | None = None
    blueprint_parent: torch.Tensor | None = None
    blueprint_presence: torch.Tensor | None = None
    node_kind_legal_mask: torch.Tensor | None = None
    node_budget_minimum: torch.Tensor | None = None


def _terminalizable_kinds(grammar: ASTTransitionGrammar) -> set[str]:
    kinds = {kind for kind, classes in grammar.leaf_classes.items() if classes}
    changed = True
    while changed:
        changed = False
        for transition in grammar.transitions:
            if transition.parent_kind not in kinds and transition.child_kind in kinds:
                kinds.add(transition.parent_kind)
                changed = True
    return kinds


def _node_kind_legal_mask(
    rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
    vocabulary: Mapping[str, object],
    max_steps: int,
) -> torch.Tensor:
    node_ids = vocabulary["node_ids"]
    mask = torch.zeros((len(rows), max_steps, len(node_ids) + 1), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        grammar = grammars[str(row["language"])]
        terminal = _terminalizable_kinds(grammar)
        decoder = TreeActionDecoder(grammar)
        for step, record in enumerate(row["actions"]):
            action = action_from_record(record)
            if isinstance(action, OpenNodeAction):
                if decoder.pending_edge is None:
                    candidates = set(grammar.legal_root_kinds)
                else:
                    pending = decoder.pending_edge
                    candidates = set(
                        decoder.state.allowed_child_kinds(
                            parent_index=pending.parent_index,
                            edge_field=pending.edge_field,
                        )
                    )
                for kind in candidates & terminal:
                    if kind in node_ids:
                        mask[row_index, step, node_ids[kind]] = True
            decoder.apply(action)
    return mask


def _pad_scalar_rows(
    flat: RealPolicyBatch,
    rows: Sequence[Mapping[str, object]],
    *,
    key: str,
    fill: float = 0,
) -> torch.Tensor:
    values = getattr(flat, key)
    length = max(len(row["actions"]) for row in rows)
    trailing = values.shape[1:]
    output = torch.full(
        (len(rows), length, *trailing),
        fill,
        dtype=values.dtype,
    )
    offset = 0
    for row_index, row in enumerate(rows):
        action_count = len(row["actions"])
        output[row_index, :action_count] = values[offset : offset + action_count]
        offset += action_count
    return output


def _pad_mapping_rows(
    flat: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, torch.Tensor]:
    output = {}
    for name, values in flat.items():
        length = max(len(row["actions"]) for row in rows)
        trailing = values.shape[1:]
        padded = torch.zeros(
            (len(rows), length, *trailing),
            dtype=values.dtype,
        )
        offset = 0
        for row_index, row in enumerate(rows):
            action_count = len(row["actions"])
            padded[row_index, :action_count] = values[offset : offset + action_count]
            offset += action_count
        output[name] = padded
    return output


def encode_prefix_rows(
    rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
    vocabulary: Mapping[str, object],
    *,
    budget_minimums: Mapping[str, int] | None = None,
) -> PrefixPolicyBatch:
    flat = encode_action_rows(rows, grammars, vocabulary)
    step_mask = torch.zeros(
        (len(rows), max(len(row["actions"]) for row in rows)),
        dtype=torch.bool,
    )
    node_budget = torch.zeros(len(rows), dtype=torch.long)
    node_budget_minimum = (
        torch.tensor(
            [int(budget_minimums[str(row["language"])]) for row in rows],
            dtype=torch.long,
        )
        if budget_minimums is not None
        else None
    )
    node_kind_mask = _node_kind_legal_mask(
        rows,
        grammars,
        vocabulary,
        max_steps=max(len(row["actions"]) for row in rows),
    )
    offset = 0
    for row_index, row in enumerate(rows):
        action_count = len(row["actions"])
        step_mask[row_index, :action_count] = True
        node_budget[row_index] = flat.node_budget[offset]
        offset += action_count
    return PrefixPolicyBatch(
        prompts=flat.prompts,
        prompt_masks=flat.prompt_masks,
        prompt_token_ids=flat.prompt_token_ids,
        prompt_token_mask=flat.prompt_token_mask,
        node_budget=node_budget,
        step_mask=step_mask,
        state_features=_pad_scalar_rows(
            flat,
            rows,
            key="state_features",
        ),
        context_kind=_pad_scalar_rows(flat, rows, key="context_kind"),
        context_edge=_pad_scalar_rows(flat, rows, key="context_edge"),
        candidate_kind=_pad_scalar_rows(flat, rows, key="candidate_kind"),
        candidate_scope=_pad_scalar_rows(flat, rows, key="candidate_scope"),
        candidate_position=_pad_scalar_rows(flat, rows, key="candidate_position"),
        candidate_mask=_pad_scalar_rows(flat, rows, key="candidate_mask"),
        candidate_target=_pad_scalar_rows(flat, rows, key="candidate_target"),
        history=_pad_mapping_rows(flat.history, rows),
        previous=_pad_mapping_rows(flat.previous, rows),
        action=_pad_scalar_rows(flat, rows, key="action"),
        node_kind=_pad_scalar_rows(flat, rows, key="node_kind"),
        edge_field=_pad_scalar_rows(flat, rows, key="edge_field"),
        parent=_pad_scalar_rows(flat, rows, key="parent"),
        leaf_node=_pad_scalar_rows(flat, rows, key="leaf_node"),
        leaf_class=_pad_scalar_rows(flat, rows, key="leaf_class"),
        copy_label=_pad_scalar_rows(flat, rows, key="copy_label"),
        copy_start=_pad_scalar_rows(flat, rows, key="copy_start"),
        copy_length=_pad_scalar_rows(flat, rows, key="copy_length"),
        lexical=_pad_scalar_rows(flat, rows, key="lexical"),
        gap=_pad_scalar_rows(flat, rows, key="gap"),
        **encode_blueprint_targets(rows, vocabulary, max_nodes=MAX_NODES),
        node_kind_legal_mask=node_kind_mask,
        node_budget_minimum=node_budget_minimum,
    )


def audit_prefix_batch(batch: PrefixPolicyBatch) -> dict[str, object]:
    checks = {
        "prompt_shapes": (
            batch.prompts.shape[:2]
            == (batch.prompt_masks.shape[0], MAX_PROMPT)
            and batch.prompt_token_ids.shape[1] == MAX_PROMPT_TOKENS
        ),
        "sequence_shapes": (
            batch.step_mask.shape
            == batch.action.shape
            == batch.state_features.shape[:2]
        ),
        "state_width": batch.state_features.shape[-1] == STATE_FEATURES,
        "candidate_width": (
            batch.candidate_kind.shape[-1] == MAX_PARENT_CANDIDATES
        ),
        "history_width": all(
            value.shape[-1] == ACTION_HISTORY
            for value in batch.history.values()
        ),
        "label_ranges": (
            int(batch.action.max()) < 6
            and int(batch.node_kind.min()) >= 0
            and int(batch.edge_field.min()) >= 0
            and int(batch.leaf_node.max()) < MAX_NODES
            and int(batch.copy_start.max()) < MAX_PROMPT
            and int(batch.copy_length.max()) <= MAX_LEXEME
        ),
        "node_budget_range": (
            int(batch.node_budget.min()) > 0
            and int(batch.node_budget.max()) <= MAX_NODES
        ),
        "node_kind_legal_targets": _legal_node_targets(batch),
        "node_budget_minimum": (
            batch.node_budget_minimum is None
            or (
                batch.node_budget_minimum.shape == batch.node_budget.shape
                and bool((batch.node_budget_minimum > 0).all())
                and bool((batch.node_budget_minimum <= batch.node_budget).all())
            )
        ),
        "blueprint_shapes": all(
            value is not None and value.shape == (batch.action.shape[0], MAX_NODES)
            for value in (
                batch.blueprint_node,
                batch.blueprint_edge,
                batch.blueprint_parent,
                batch.blueprint_presence,
            )
        ),
    }
    return {
        "version": "nlp-prefix-action-batch-audit-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "rows": batch.action.shape[0],
        "maximum_steps": batch.action.shape[1],
        "active_steps": int(batch.step_mask.sum()),
        "checks": checks,
        "evidence_boundary": "Padded action-prefix batch contract only.",
    }


def _legal_node_targets(batch: PrefixPolicyBatch) -> bool:
    mask = batch.node_kind_legal_mask
    if mask is None or mask.shape[:2] != batch.action.shape:
        return False
    active = batch.step_mask & batch.action.eq(REAL_ACTIONS.index("open_node"))
    if not active.any():
        return False
    return bool(mask[active].gather(1, batch.node_kind[active, None]).all())


__all__ = ["PrefixPolicyBatch", "audit_prefix_batch", "encode_prefix_rows"]
