"""Selection policies for legal and certified AST action decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from floorflow.nlp_flow.ast_generation_trace import record_choice, record_event, tracing_enabled
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.ast_leaf_classifier import classify_leaf
from floorflow.nlp_flow.real_corpus_policy_data import REAL_ACTIONS
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
)
from floorflow.nlp_flow.tree_completion_runtime import TreeCompletionBudget


def _masked_choice(logits: torch.Tensor, allowed: set[int], *, head: str = "unspecified") -> int:
    if not allowed:
        record_choice(logits, allowed, None, head)
        raise ValueError("no legal action candidate")
    masked = torch.full_like(logits, -torch.inf)
    for index in allowed:
        masked[index] = logits[index]
    selected = int(masked.argmax())
    record_choice(logits, allowed, selected, head)
    return selected


def _allowed_action_types(
    decoder: TreeActionDecoder,
    node_budget: int,
    parent_kinds: set[str],
    terminal_kinds: set[str],
    completion_budget: TreeCompletionBudget | None = None,
    local_transition_mask: bool = True,
) -> set[int]:
    if decoder.expected_scope is not None:
        return {REAL_ACTIONS.index("open_scope")}
    if not decoder.nodes:
        return {REAL_ACTIONS.index("open_node")}
    if decoder.pending_edge is not None:
        return {REAL_ACTIONS.index("open_node")}
    if not local_transition_mask:
        return set(range(len(REAL_ACTIONS)))
    allowed = set()
    has_attach_candidate = False
    if len(decoder.nodes) < node_budget:
        active_scope = decoder.state.open_scopes[-1] if decoder.state.open_scopes else None
        for parent_index, node in enumerate(decoder.nodes):
            if node.scope_id != active_scope or node.kind not in parent_kinds:
                continue
            for transition in decoder.grammar.transitions:
                if transition.parent_kind != node.kind or transition.child_kind not in terminal_kinds:
                    continue
                if completion_budget is not None and not completion_budget.child_candidates(
                    decoder,
                    parent_index=parent_index,
                    field=transition.edge_field,
                    kinds=(transition.child_kind,),
                    remaining_nodes=node_budget - len(decoder.nodes),
                ):
                    continue
                has_attach_candidate = True
                break
            if has_attach_candidate:
                break
    if has_attach_candidate:
        allowed.add(REAL_ACTIONS.index("attach_edge"))
    if any(
        not node.children
        and node.leaf_class is None
        and bool(decoder.grammar.leaf_classes.get(node.kind))
        for node in decoder.nodes
    ):
        allowed.add(REAL_ACTIONS.index("declare_leaf"))
    active_scope = (
        decoder.state.open_scopes[-1]
        if decoder.state.open_scopes
        else None
    )
    scope_has_incomplete_node = any(
        node.scope_id == active_scope
        and not node.children
        and node.leaf_class is None
        for node in decoder.nodes
    )
    if (
        decoder.state.open_scopes
        and not scope_has_incomplete_node
        and (completion_budget is None or completion_budget.can_close_active_scope(decoder))
    ):
        allowed.add(REAL_ACTIONS.index("close_scope"))
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    if (
        decoder.state.eos_ready
        and not undeclared
        and (completion_budget is None or completion_budget.minimum_required_nodes(decoder) == 0)
    ):
        allowed.add(REAL_ACTIONS.index("terminate"))
    return allowed


def _select_real_action(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
    logits: Mapping[str, torch.Tensor],
    vocabulary: Mapping[str, object],
    prompt: str,
    parent_candidates: Sequence[int],
    node_budget: int,
    parent_kinds: set[str],
    terminal_kinds: set[str],
    completion_budget: TreeCompletionBudget | None = None,
    local_transition_mask: bool = True,
    lexical_candidate_mask: bool = True,
) -> TreeAction:
    if decoder.expected_scope is not None:
        scope_id, node_index, scope_kind = decoder.expected_scope
        return OpenScopeAction(scope_id, node_index, scope_kind)
    action_id = _masked_choice(
        logits["action"][0],
        _allowed_action_types(
            decoder,
            node_budget,
            parent_kinds,
            terminal_kinds,
            completion_budget,
            local_transition_mask,
        ),
        head="action",
    )
    action_name = REAL_ACTIONS[action_id]
    if action_name == "open_node":
        if not decoder.nodes:
            allowed_nodes = (
                set(vocabulary["node_ids"].values())
                if not local_transition_mask
                else {
                    vocabulary["node_ids"][value]
                    for value in grammar.legal_root_kinds
                    if value in vocabulary["node_ids"]
                    and value in terminal_kinds
                    and (1 < node_budget or bool(grammar.leaf_classes.get(value)))
                }
            )
            if completion_budget is not None:
                candidates = completion_budget.root_candidates(
                    grammar.legal_root_kinds,
                    remaining_nodes=node_budget - len(decoder.nodes),
                )
                allowed_nodes &= {
                    vocabulary["node_ids"][kind]
                    for kind in candidates
                    if kind in vocabulary["node_ids"]
                }
        else:
            pending = decoder.pending_edge
            if pending is None:
                raise ValueError("non-root node has no edge attachment")
            allowed_nodes = (
                set(vocabulary["node_ids"].values())
                if not local_transition_mask
                else {
                    vocabulary["node_ids"][value]
                    for value in decoder.state.allowed_child_kinds(
                        parent_index=pending.parent_index,
                        edge_field=pending.edge_field,
                    )
                    if value in vocabulary["node_ids"] and value in terminal_kinds
                }
            )
            if completion_budget is not None:
                candidates = completion_budget.child_candidates(
                    decoder,
                    parent_index=pending.parent_index,
                    field=pending.edge_field,
                    kinds=decoder.state.allowed_child_kinds(
                        parent_index=pending.parent_index,
                        edge_field=pending.edge_field,
                    ),
                    remaining_nodes=node_budget - len(decoder.nodes),
                )
                allowed_nodes &= {
                    vocabulary["node_ids"][kind]
                    for kind in candidates
                    if kind in vocabulary["node_ids"]
                }
        node_id = _masked_choice(logits["node_kind"][0], allowed_nodes, head="node_kind")
        return OpenNodeAction(vocabulary["node_names"][node_id])
    if action_name == "attach_edge":
        allowed_parent_positions = set(range(len(parent_candidates)))
        if completion_budget is not None and local_transition_mask:
            allowed_parent_positions = set()
            for position, parent in enumerate(parent_candidates):
                for transition in grammar.transitions:
                    if transition.parent_kind != decoder.nodes[parent].kind:
                        continue
                    children = decoder.state.allowed_child_kinds(
                        parent_index=parent,
                        edge_field=transition.edge_field,
                    )
                    if completion_budget.child_candidates(
                        decoder,
                        parent_index=parent,
                        field=transition.edge_field,
                        kinds=children,
                        remaining_nodes=node_budget - len(decoder.nodes),
                    ):
                        allowed_parent_positions.add(position)
                        break
        parent_position = _masked_choice(
            logits["parent"][0],
            allowed_parent_positions,
            head="parent",
        )
        parent = parent_candidates[parent_position]
        parent_kind = decoder.nodes[parent].kind
        allowed_edges = (
            set(vocabulary["edge_ids"].values())
            if not local_transition_mask
            else {
                vocabulary["edge_ids"][value.edge_field]
                for value in grammar.transitions
                if value.parent_kind == parent_kind
                and value.edge_field in vocabulary["edge_ids"]
                and value.child_kind in terminal_kinds
            }
        )
        if completion_budget is not None:
            allowed_edges = {
                edge_id
                for edge_id in allowed_edges
                if completion_budget.child_candidates(
                    decoder,
                    parent_index=parent,
                    field=vocabulary["edge_names"][edge_id],
                    kinds=decoder.state.allowed_child_kinds(
                        parent_index=parent,
                        edge_field=vocabulary["edge_names"][edge_id],
                    ),
                    remaining_nodes=node_budget - len(decoder.nodes),
                )
            }
        edge_id = _masked_choice(logits["edge_field"][0], allowed_edges, head="edge_field")
        return AttachEdgeAction(parent, vocabulary["edge_names"][edge_id])
    if action_name == "declare_leaf":
        allowed_leaves = {
            index
            for index, node in enumerate(decoder.nodes)
            if not node.children
            and node.leaf_class is None
            and (not local_transition_mask or bool(grammar.leaf_classes.get(node.kind)))
        }
        node_index = _masked_choice(logits["leaf_node"][0], allowed_leaves, head="leaf_node")
        node = decoder.nodes[node_index]
        allowed_classes = (
            set(vocabulary["class_ids"].values())
            if not lexical_candidate_mask
            else {
                vocabulary["class_ids"][value]
                for value in grammar.leaf_classes[node.kind]
                if value in vocabulary["class_ids"]
            }
        )
        class_id = _masked_choice(logits["leaf_class"][0], allowed_classes, head="leaf_class")
        leaf_class = vocabulary["class_names"][class_id]
        copy_selected = bool(torch.sigmoid(logits["copy"][0]).ge(0.5))
        if copy_selected:
            start = int(logits["copy_start"][0].argmax())
            length = int(logits["copy_length"][0].argmax())
            lexical = prompt[start : start + length]
            copy_class = classify_leaf(node.kind, lexical)
            if tracing_enabled():
                record_event("copy", selected=True, score=float(logits["copy"][0]), start=start, length=length,
                             value=lexical, observed_class=copy_class, expected_class=leaf_class)
                record_choice(logits["copy_start"][0], set(range(logits["copy_start"].shape[-1])), start, "copy_start")
                record_choice(
                    logits["copy_length"][0], set(range(logits["copy_length"].shape[-1])), length, "copy_length"
                )
            if copy_class != leaf_class:
                lexical = ""
        else:
            if tracing_enabled():
                record_event("copy", selected=False, score=float(logits["copy"][0]))
            lexical = ""
        if not lexical:
            lexical_allowed = (
                set(range(len(vocabulary["lexeme_names"])))
                if not lexical_candidate_mask
                else vocabulary["lexemes_by_kind_class"].get(f"{node.kind}|{leaf_class}", [])
            )
            if lexical_allowed:
                lexical_id = _masked_choice(
                    logits["lexical"][0],
                    set(lexical_allowed),
                    head="lexical",
                )
                lexical = vocabulary["lexeme_names"][lexical_id]
            else:
                lexical_id = int(logits["lexical"][0].argmax())
                if tracing_enabled():
                    record_choice(
                        logits["lexical"][0], set(range(logits["lexical"].shape[-1])), lexical_id, "lexical_fallback"
                    )
                lexical = vocabulary["lexeme_names"][lexical_id]
        gap_allowed = (
            set(range(len(vocabulary["gap_names"])))
            if not lexical_candidate_mask
            else vocabulary["gaps_by_kind"].get(node.kind, [])
        )
        gap_id = (
            _masked_choice(logits["gap"][0], set(gap_allowed), head="gap")
            if gap_allowed
            else int(logits["gap"][0].argmax())
        )
        gap = vocabulary["gap_names"][gap_id]
        if not gap_allowed and tracing_enabled():
            record_choice(logits["gap"][0], set(range(logits["gap"].shape[-1])), gap_id, "gap_fallback")
        return DeclareLeafAction(
            node_index,
            leaf_class,
            lexical,
            gap,
        )
    if action_name == "close_scope":
        return CloseScopeAction(decoder.state.open_scopes[-1])
    return TerminateAction()
