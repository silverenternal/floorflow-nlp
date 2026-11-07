"""Capture reproducible snapshots of tree action decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import TreeAction, TreeActionDecoder


@dataclass(frozen=True)
class DecoderSnapshot:
    """Immutable view of the information needed to diagnose an action prefix."""

    nodes: tuple[tuple[str, int, str, int, tuple[int, ...], str | None, str | None, str], ...]
    open_scopes: tuple[int, ...]
    scope_kinds: tuple[tuple[int, str], ...]
    next_scope_id: int
    grammar_finished: bool
    pending_edge: tuple[int, str] | None
    expected_scope: tuple[int, int, str] | None
    terminated: bool
    action_count: int


def snapshot(decoder: TreeActionDecoder) -> DecoderSnapshot:
    """Copy decoder state without retaining references to mutable nodes."""
    pending = decoder.pending_edge
    return DecoderSnapshot(
        nodes=tuple(
            (
                node.kind, node.parent_index, node.edge_field, node.scope_id, tuple(node.children),
                node.leaf_class, node.lexical, node.gap_before,
            )
            for node in decoder.nodes
        ),
        open_scopes=tuple(decoder.state.open_scopes),
        scope_kinds=decoder.state.scope_kinds,
        next_scope_id=decoder.state.next_scope_id,
        grammar_finished=decoder.state.finished,
        pending_edge=None if pending is None else (pending.parent_index, pending.edge_field),
        expected_scope=decoder.expected_scope,
        terminated=decoder.terminated,
        action_count=len(decoder.actions),
    )


def trace_action(decoder: TreeActionDecoder, action: TreeAction) -> dict[str, object]:
    """Apply one action and record its state transition or explicit failure.

    Failed actions are not retried. Callers must stop replay after a failure.
    """
    before = snapshot(decoder)
    try:
        decoder.apply(action)
    except (ValueError, TypeError) as error:
        return {
            "action_type": type(action).__name__,
            "action": asdict(action),
            "before": asdict(before),
            "after": asdict(snapshot(decoder)),
            "accepted": False,
            "error": str(error),
        }
    return {
        "action_type": type(action).__name__,
        "action": asdict(action),
        "before": asdict(before),
        "after": asdict(snapshot(decoder)),
        "accepted": True,
        "error": None,
    }
