"""Shared tree-action transitions and state-safe teacher divergence alignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

from floorflow.nlp_flow.ast_discrete_flow import legal_action_candidates
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
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

StateSignature: TypeAlias = tuple[object, ...]


@dataclass(frozen=True)
class TeacherNextAlignment:
    compatible: tuple[bool, ...]
    reasons: tuple[str, ...]
    divergence_indices: tuple[int, ...]


def _signature(decoder: TreeActionDecoder) -> StateSignature:
    nodes = tuple(
        (
            node.kind,
            node.parent_index,
            node.edge_field,
            node.scope_id,
            tuple(node.children),
            node.leaf_class,
            node.lexical,
            node.gap_before,
        )
        for node in decoder.nodes
    )
    grammar_state = decoder.state
    return (
        nodes,
        grammar_state.nodes,
        grammar_state.scope_kinds,
        grammar_state.open_scopes,
        grammar_state.next_scope_id,
        grammar_state.finished,
        decoder.pending_edge,
        decoder.expected_scope,
        decoder.terminated,
    )


def _clone(decoder: TreeActionDecoder) -> TreeActionDecoder:
    clone = TreeActionDecoder(decoder.grammar)
    for action in decoder.actions:
        clone.apply(action)
    return clone


def _action_name(action: TreeAction) -> str:
    return {
        AttachEdgeAction: "attach_edge",
        OpenNodeAction: "open_node",
        OpenScopeAction: "open_scope",
        DeclareLeafAction: "declare_leaf",
        CloseScopeAction: "close_scope",
        TerminateAction: "terminate",
    }[type(action)]


def action_is_legal(decoder: TreeActionDecoder, action: TreeAction) -> bool:
    """Check category masks and the same decoder transition used at inference."""
    candidates = legal_action_candidates(decoder, decoder.grammar)
    name = _action_name(action)
    if name not in candidates["action_type"]:
        return False
    if isinstance(action, OpenNodeAction) and action.node_kind not in candidates["node_kind"]:
        return False
    # The legacy inference mask lists edge fields for one eligible parent only;
    # the decoder transition checks the selected parent/field pair exactly.
    if isinstance(action, DeclareLeafAction) and action.leaf_class not in candidates["leaf_class"]:
        return False
    trial = _clone(decoder)
    try:
        trial.apply(action)
    except (IndexError, TypeError, ValueError):
        return False
    return True


class TreeRollout:
    """A single shared decoder/mask state for teacher, sampled and replay actions."""

    def __init__(self, grammar: ASTTransitionGrammar) -> None:
        self.decoder = TreeActionDecoder(grammar)

    @property
    def legal_candidates(self) -> dict[str, tuple[str, ...]]:
        return {
            key: tuple(value)
            for key, value in legal_action_candidates(self.decoder, self.decoder.grammar).items()
        }

    @property
    def state_fingerprint(self) -> str:
        payload = repr(_signature(self.decoder)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def apply(self, action: TreeAction) -> TreeRollout:
        if not action_is_legal(self.decoder, action):
            raise ValueError(f"rollout action is outside the shared transition mask: {_action_name(action)}")
        self.decoder.apply(action)
        return self


def _state_history(
    grammar: ASTTransitionGrammar,
    actions: tuple[TreeAction, ...],
    *,
    stop_on_invalid: bool,
) -> tuple[list[TreeActionDecoder], list[StateSignature], int | None]:
    decoder = TreeActionDecoder(grammar)
    decoders = [decoder]
    signatures = [_signature(decoder)]
    invalid_index = None
    for index, action in enumerate(actions):
        if not action_is_legal(decoder, action):
            if stop_on_invalid:
                invalid_index = index
                break
            raise ValueError(f"teacher action {index} is outside the shared transition mask")
        decoder = _clone(decoder)
        decoder.apply(action)
        decoders.append(decoder)
        signatures.append(_signature(decoder))
    return decoders, signatures, invalid_index


def align_teacher_next_labels(
    grammar: ASTTransitionGrammar,
    teacher_actions: tuple[TreeAction, ...] | list[TreeAction],
    rollout_actions: tuple[TreeAction, ...] | list[TreeAction],
) -> TeacherNextAlignment:
    """Supervise a teacher next action only when the generated prefix state matches."""
    teacher = tuple(teacher_actions)
    rollout = tuple(rollout_actions)
    teacher_decoders, teacher_states, _ = _state_history(grammar, teacher, stop_on_invalid=False)
    rollout_decoders, rollout_states, invalid_index = _state_history(grammar, rollout, stop_on_invalid=True)
    compatible = []
    reasons = []
    divergences = []
    for index, action in enumerate(teacher):
        if index >= len(rollout_states) or (invalid_index is not None and index > invalid_index):
            compatible.append(False)
            reasons.append("rollout_stopped_or_invalid")
            divergences.append(index)
        elif teacher_states[index] != rollout_states[index]:
            compatible.append(False)
            reasons.append("prefix_state_diverged")
            divergences.append(index)
        elif teacher_states[index] != _signature(teacher_decoders[index]):
            raise AssertionError("teacher state history is inconsistent")
        elif not action_is_legal(rollout_decoders[index], action):
            compatible.append(False)
            reasons.append("teacher_action_outside_rollout_mask")
            divergences.append(index)
        else:
            compatible.append(True)
            reasons.append("compatible_prefix")
    return TeacherNextAlignment(tuple(compatible), tuple(reasons), tuple(divergences))


__all__ = [
    "TeacherNextAlignment",
    "TreeRollout",
    "action_is_legal",
    "align_teacher_next_labels",
]
