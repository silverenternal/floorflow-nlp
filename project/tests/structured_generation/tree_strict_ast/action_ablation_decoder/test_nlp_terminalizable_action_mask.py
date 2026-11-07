from __future__ import annotations

import hashlib

import pytest

from newera.nlp_flow.components.ast.diagnostics.validation.ast_completion_contract import ASTCompletionContract
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransition, ASTTransitionGrammar
from newera.nlp_flow.components.generation.trees.actions.terminalizable_action_mask import (
    TerminalizableActionMask,
    UnsupportedCompletionGrammar,
)


def _contract(*, recursive_dead_end: bool = False) -> ASTCompletionContract:
    grammar = ASTTransitionGrammar(
        language="fixture",
        legal_root_kinds=("root",),
        transitions=(
            ASTTransition("root", "left", "integer", "root", "root", False),
            ASTTransition("root", "right", "branch", "root", "branch", True),
            ASTTransition("root", "right", "identifier", "root", "root", False),
            ASTTransition("branch", "next", "branch", "branch", "branch", True),
            ASTTransition("branch", "next", "identifier", "branch", "branch", False),
        ),
        leaf_classes={"integer": ("integer",), "identifier": ("identifier",)},
        corpus_sha256="c" * 64,
    )
    if recursive_dead_end:
        grammar = ASTTransitionGrammar(
            language="fixture",
            legal_root_kinds=("dead_root",),
            transitions=(ASTTransition("dead_root", "value", "loop", "dead_root", "loop", True),
                         ASTTransition("loop", "next", "loop", "loop", "loop", True)),
            leaf_classes={},
            corpus_sha256="c" * 64,
        )
        schema = [
            {"type": "dead_root", "named": True, "fields": {
                "value": {"multiple": False, "required": True, "types": [{"type": "loop", "named": True}]}
            }},
            {"type": "loop", "named": True, "fields": {
                "next": {"multiple": False, "required": True, "types": [{"type": "loop", "named": True}]}
            }},
        ]
    else:
        schema = [
            {"type": "root", "named": True, "fields": {
                "left": {"multiple": False, "required": True, "types": [{"type": "integer", "named": True}]},
                "right": {"multiple": False, "required": True, "types": [
                    {"type": "branch", "named": True}, {"type": "identifier", "named": True}
                ]},
            }},
            {"type": "branch", "named": True, "fields": {
                "next": {"multiple": False, "required": True, "types": [
                    {"type": "branch", "named": True}, {"type": "identifier", "named": True}
                ]}
            }},
            {"type": "integer", "named": True},
            {"type": "identifier", "named": True},
        ]
    return ASTCompletionContract.from_payload(
        grammar,
        schema,
        schema_sha256=hashlib.sha256(repr(schema).encode()).hexdigest(),
    )


def test_required_fields_are_anded_and_recursive_alternatives_get_minimum_cost() -> None:
    mask = TerminalizableActionMask(_contract())
    certificate = mask.certificate("root")
    assert certificate.minimum_nodes == 3
    assert [(choice.field, choice.child_kind, choice.minimum_nodes) for choice in certificate.required_choices] == [
        ("left", "integer", 1),
        ("right", "identifier", 1),
    ]
    branch = mask.certificate("branch")
    assert branch.minimum_nodes == 2
    assert branch.required_choices[0].child_kind == "identifier"


def test_masks_candidate_by_budget_and_keeps_gold_choice_reachable() -> None:
    contract = _contract()
    mask = TerminalizableActionMask(contract)
    state = contract.initial_state().start_root("root")
    assert not mask.can_close_node(state)
    too_small = mask.candidate_mask(
        state,
        field="left",
        candidates=(("integer", True),),
        remaining_node_budget=1,
    )
    enough = mask.candidate_mask(
        state,
        field="left",
        candidates=(("integer", True),),
        remaining_node_budget=2,
    )
    assert too_small.allowed == (False,)
    assert enough.allowed == (True,)
    assert mask.teacher_action_reachable(
        state,
        field="left",
        child_kind="integer",
        child_named=True,
        remaining_node_budget=2,
    )
    assert not mask.teacher_action_reachable(
        state,
        field="left",
        child_kind="integer",
        child_named=True,
        remaining_node_budget=1,
    )


def test_early_close_and_duplicate_singleton_field_are_rejected() -> None:
    contract = _contract()
    state = contract.initial_state().start_root("root")
    with pytest.raises(ValueError, match="required parser fields"):
        state.close_node()
    state = state.append_child("left", "integer").close_node()
    with pytest.raises(ValueError, match="non-repeated"):
        state.append_child("left", "integer")
    duplicate = TerminalizableActionMask(contract).candidate_mask(
        state,
        field="left",
        candidates=(("integer", True),),
        remaining_node_budget=2,
    )
    assert duplicate.allowed == (False,)


def test_recursive_grammar_without_a_base_case_fails_explicitly() -> None:
    contract = _contract(recursive_dead_end=True)
    mask = TerminalizableActionMask(contract)
    with pytest.raises(UnsupportedCompletionGrammar, match="no finite completion"):
        mask.certificate("loop")
    state = contract.initial_state().start_root("dead_root")
    with pytest.raises(UnsupportedCompletionGrammar, match="no finite completion"):
        mask.candidate_mask(
            state,
            field="value",
            candidates=(("loop", True),),
            remaining_node_budget=8,
        )
