from __future__ import annotations

from pathlib import Path

import pytest
import torch

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTGrammarState, ASTTransition, ASTTransitionGrammar

ROOT = Path(__file__).resolve().parents[5]
GRAMMAR = ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json"


def _python_state() -> ASTGrammarState:
    return ASTGrammarState.from_file(GRAMMAR, "python")


def test_root_child_and_edge_logits_mask_illegal_actions() -> None:
    state = _python_state()
    kinds = ("module", "ERROR", "function_definition")
    root_logits = state.mask_root_logits(
        torch.zeros(len(kinds)),
        kinds,
    )

    assert root_logits[0].isfinite()
    assert root_logits[1] == -torch.inf

    state = state.start_root("module")
    child_logits = state.mask_child_logits(
        torch.zeros(len(kinds)),
        kinds,
        parent_index=0,
    )
    assert child_logits[2].isfinite()
    assert child_logits[0] == -torch.inf

    edge_fields = ("child", "alias")
    edge_logits = state.mask_edge_logits(
        torch.zeros(len(edge_fields)),
        edge_fields,
        parent_index=0,
        child_kind="function_definition",
    )
    assert edge_logits[0].isfinite()
    assert edge_logits[1] == -torch.inf


def test_scope_lifecycle_blocks_duplicate_closed_and_eos_actions() -> None:
    state = _python_state().start_root("module")
    eos = state.mask_eos_logits(torch.zeros(1))
    assert eos[0] == -torch.inf

    state = state.add_child(
        parent_index=0,
        edge_field="child",
        child_kind="function_definition",
    )
    assert state.open_scopes == (0, 1)
    with pytest.raises(ValueError, match="LIFO"):
        state.close_scope(0)

    state = state.close_scope(1)
    with pytest.raises(ValueError, match="closed scope"):
        state.add_child(
            parent_index=1,
            edge_field="child",
            child_kind="identifier",
        )
    state = state.close_scope(0)
    finished = state.finish()
    assert finished.finished
    with pytest.raises(ValueError, match="EOS"):
        finished.finish()


def test_parent_cycle_and_leaf_class_masks_are_structural() -> None:
    state = _python_state().start_root("module")
    parent_logits = state.mask_parent_logits(
        torch.zeros(1),
        child_kind="function_definition",
    )

    assert parent_logits[0].isfinite()
    assert not state.topology_is_acyclic([-1, 2, 1])
    assert state.topology_is_acyclic([-1, 0, 0, 1])
    cycle_logits = state.mask_cycle_edges(
        torch.zeros(3, 3),
        [-1, 2, 1],
    )
    assert not cycle_logits[1:].isfinite().all()

    scope_logits = state.mask_scope_close_logits(
        torch.zeros(2),
        (0, 1),
    )
    assert scope_logits[0].isfinite()
    assert scope_logits[1] == -torch.inf

    leaf_logits = state.mask_leaf_logits(
        torch.zeros(2),
        ("identifier", "numeric_literal"),
        node_kind="identifier",
    )
    assert leaf_logits[0].isfinite()
    assert leaf_logits[1] == -torch.inf


def test_scope_opening_uses_the_selected_parent_field_transition() -> None:
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=(
            ASTTransition("module", "keyword", "lambda", "module", "module", False),
            ASTTransition("module", "expression", "lambda", "module", "lambda", True),
        ),
        leaf_classes={"lambda": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    state = ASTGrammarState.initial(grammar).start_root("module")

    anonymous_keyword = state.add_child(
        parent_index=0,
        edge_field="keyword",
        child_kind="lambda",
    )
    named_expression = state.add_child(
        parent_index=0,
        edge_field="expression",
        child_kind="lambda",
    )

    assert anonymous_keyword.open_scopes == (0,)
    assert named_expression.open_scopes == (0, 1)
