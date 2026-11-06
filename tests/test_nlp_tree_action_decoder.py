from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeActionDecoder,
    decode_actions,
    render_tree_structure,
    trace_teacher_actions,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def _payload(source: str = "value = 1\n") -> dict[str, object]:
    return json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))


def test_teacher_trace_uses_explicit_typed_tree_actions() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")

    actions = trace_teacher_actions(grammar, _payload())
    structure = decode_actions(grammar, actions)

    assert isinstance(actions[0], OpenNodeAction)
    assert isinstance(actions[1], OpenScopeAction)
    assert any(isinstance(action, AttachEdgeAction) for action in actions)
    assert any(isinstance(action, DeclareLeafAction) for action in actions)
    assert any(isinstance(action, CloseScopeAction) for action in actions)
    assert isinstance(actions[-1], TerminateAction)
    assert structure["language"] == "python"
    assert structure["nodes"]
    assert render_tree_structure(structure) == "value = 1\n"


def test_decoder_rejects_missing_scope_open_and_premature_termination() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    decoder = TreeActionDecoder(grammar)
    decoder.apply(OpenNodeAction("module"))

    with pytest.raises(ValueError, match="pending"):
        decoder.apply(TerminateAction())
    with pytest.raises(ValueError, match="scope"):
        decoder.apply(OpenNodeAction("function_definition"))

    decoder.apply(
        OpenScopeAction(
            scope_id=0,
            node_index=0,
            scope_kind="module",
        )
    )
    with pytest.raises(ValueError, match="leaf"):
        decoder.apply(TerminateAction())


def test_decoder_rejects_illegal_edge_and_leaf_class() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    decoder = (
        TreeActionDecoder(grammar)
        .apply(OpenNodeAction("module"))
        .apply(
            OpenScopeAction(
                scope_id=0,
                node_index=0,
                scope_kind="module",
            )
        )
        .apply(AttachEdgeAction(parent_index=0, edge_field="invalid"))
    )
    with pytest.raises(ValueError, match="illegal AST transition"):
        decoder.apply(OpenNodeAction("expression_statement"))

    leaf_decoder = (
        TreeActionDecoder(grammar)
        .apply(OpenNodeAction("module"))
        .apply(
            OpenScopeAction(
                scope_id=0,
                node_index=0,
                scope_kind="module",
            )
        )
        .apply(AttachEdgeAction(parent_index=0, edge_field="child"))
        .apply(OpenNodeAction("expression_statement"))
        .apply(AttachEdgeAction(parent_index=1, edge_field="child"))
        .apply(OpenNodeAction("assignment"))
        .apply(AttachEdgeAction(parent_index=2, edge_field="left"))
        .apply(OpenNodeAction("identifier"))
    )
    with pytest.raises(ValueError, match="leaf class"):
        leaf_decoder.apply(
            DeclareLeafAction(
                node_index=3,
                leaf_class="numeric_literal",
            )
        )
