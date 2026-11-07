"""Inspect the actual historical tree-policy masks at a teacher prefix."""

from __future__ import annotations

from collections.abc import Mapping

from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import MAX_LEXEME, MAX_PARENT_CANDIDATES, MAX_PROMPT, REAL_ACTIONS
from newera.nlp_flow.components.generation.trees.actions.tree_action_selection import _allowed_action_types
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TreeAction,
    TreeActionDecoder,
)


def inspect_teacher_mask(
    decoder: TreeActionDecoder,
    action: TreeAction,
    budget: int,
    parent_kinds: set[str],
    terminal_kinds: set[str],
    vocabulary: Mapping[str, object],
    prompt: str | None = None,
) -> dict[str, object]:
    """Report exclusions in the historical policy; do not change its choices.

    Lexical reachability is existential over possible logits, not a prediction.
    """
    names = {
        "AttachEdgeAction": "attach_edge",
        "OpenNodeAction": "open_node",
        "OpenScopeAction": "open_scope",
        "DeclareLeafAction": "declare_leaf",
        "CloseScopeAction": "close_scope",
        "TerminateAction": "terminate",
    }
    action_name = names[type(action).__name__]
    allowed_ids = _allowed_action_types(decoder, budget, parent_kinds, terminal_kinds)
    allowed_names = [name for index, name in enumerate(REAL_ACTIONS) if index in allowed_ids]
    reasons: list[str] = []
    lexical_check: dict[str, object] | None = None
    if action_name not in allowed_names:
        reasons.append("action_type_masked")
    if isinstance(action, OpenScopeAction):
        if decoder.expected_scope != (action.scope_id, action.node_index, action.scope_kind):
            reasons.append("scope_mismatch")
    elif isinstance(action, OpenNodeAction):
        if not decoder.nodes:
            allowed_nodes = {
                kind
                for kind in decoder.grammar.legal_root_kinds
                if kind in vocabulary["node_ids"]
                and kind in terminal_kinds
                and (budget > 1 or bool(decoder.grammar.leaf_classes.get(kind)))
            }
        elif decoder.pending_edge is not None:
            edge = decoder.pending_edge
            allowed_nodes = {
                kind
                for kind in decoder.state.allowed_child_kinds(
                    parent_index=edge.parent_index, edge_field=edge.edge_field
                )
                if kind in vocabulary["node_ids"] and kind in terminal_kinds
            }
        else:
            allowed_nodes = set()
        if action.node_kind not in allowed_nodes:
            reasons.append("node_kind_masked")
    elif isinstance(action, AttachEdgeAction):
        active = decoder.state.open_scopes[-1] if decoder.state.open_scopes else None
        parents = [i for i, node in enumerate(decoder.nodes) if node.scope_id == active and node.kind in parent_kinds]
        if 0 <= action.parent_index < len(decoder.nodes):
            node = decoder.nodes[action.parent_index]
            kind = node.kind
            if node.scope_id != active:
                reasons.append("parent_not_in_active_scope")
            if node.kind not in parent_kinds:
                reasons.append("parent_kind_nonviable")
            if action.parent_index in parents and action.parent_index not in parents[-MAX_PARENT_CANDIDATES:]:
                reasons.append("parent_candidate_window_truncated")
            edges = {
                t.edge_field
                for t in decoder.grammar.transitions
                if t.parent_kind == kind and t.edge_field in vocabulary["edge_ids"] and t.child_kind in terminal_kinds
            }
            if action.edge_field not in edges:
                reasons.append("edge_field_masked")
        else:
            reasons.append("parent_out_of_bounds")
    elif isinstance(action, DeclareLeafAction):
        if not 0 <= action.node_index < len(decoder.nodes):
            reasons.append("leaf_node_out_of_bounds")
        else:
            node = decoder.nodes[action.node_index]
            classes = decoder.grammar.leaf_classes.get(node.kind, ())
            if node.children or node.leaf_class is not None or not classes:
                reasons.append("leaf_node_masked")
            if action.leaf_class not in classes or action.leaf_class not in vocabulary["class_ids"]:
                reasons.append("leaf_class_masked")
            if prompt is not None:
                value = action.lexical
                copy_reachable = bool(
                    value
                    and len(value) < MAX_LEXEME
                    and classify_leaf(node.kind, value) == action.leaf_class
                    and any(
                        prompt[start : start + len(value)] == value for start in range(min(len(prompt), MAX_PROMPT))
                    )
                )
                value_ids = vocabulary["lexeme_ids"]
                value_id = value_ids.get(value) if value is not None else None
                allowed = vocabulary["lexemes_by_kind_class"].get(f"{node.kind}|{action.leaf_class}", [])
                vocabulary_reachable = value_id is not None and (not allowed or value_id in allowed)
                gap_id = vocabulary["gap_ids"].get(action.gap_before)
                allowed_gaps = vocabulary["gaps_by_kind"].get(node.kind, [])
                gap_reachable = gap_id is not None and (not allowed_gaps or gap_id in allowed_gaps)
                if not copy_reachable and not vocabulary_reachable:
                    reasons.append("lexical_value_unreachable")
                if not gap_reachable:
                    reasons.append("gap_value_unreachable")
                lexical_check = {
                    "copy_reachable": copy_reachable,
                    "vocabulary_reachable": vocabulary_reachable,
                    "gap_reachable": gap_reachable,
                }
    return {
        "allowed_action_types": allowed_names,
        "teacher_action": action_name,
        "reachable": not reasons,
        "reasons": reasons,
        "lexical_value_checked": lexical_check is not None,
        "lexical": lexical_check,
    }
