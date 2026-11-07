from __future__ import annotations

import hashlib

import pytest
import torch

from newera.nlp_flow.components.ast.diagnostics.validation.ast_completion_contract import ASTCompletionContract, ASTCompletionState, ChildEdge
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransition, ASTTransitionGrammar
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_tree_runtime import terminalizable_kinds, viable_parent_kinds
from newera.nlp_flow.components.generation.trees.actions.terminalizable_action_mask import TerminalizableActionMask
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    OpenNodeAction,
    OpenScopeAction,
    TreeActionDecoder,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_selection import _select_real_action
from newera.nlp_flow.components.generation.trees.runtime.tree_completion_runtime import TreeCompletionBudget


def _completion_budget() -> TreeCompletionBudget:
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=(
            ASTTransition("module", "name", "identifier", "module", "module", False),
            ASTTransition("module", "body", "function_definition", "module", "function_definition", True),
            ASTTransition(
                "function_definition", "name", "identifier", "function_definition", "function_definition", False
            ),
        ),
        leaf_classes={"identifier": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    schema = [
        {
            "type": "module",
            "named": True,
            "fields": {
                "name": {"multiple": False, "required": True, "types": [{"type": "identifier", "named": True}]},
                "body": {
                    "multiple": True,
                    "required": False,
                    "types": [{"type": "function_definition", "named": True}],
                },
            },
        },
        {
            "type": "function_definition",
            "named": True,
            "fields": {
                "name": {"multiple": False, "required": True, "types": [{"type": "identifier", "named": True}]}
            },
        },
        {"type": "identifier", "named": True},
    ]
    contract = ASTCompletionContract.from_payload(
        grammar,
        schema,
        schema_sha256=hashlib.sha256(b"test-schema").hexdigest(),
    )
    return TreeCompletionBudget(contract)


def test_parser_schema_resolves_namedness_by_parent_field() -> None:
    budget = _completion_budget()

    assert budget.contract.namedness_for_kind("module") is True
    assert budget.contract.child_namedness("module", True, "body", "function_definition") is True


def test_completion_candidates_reserve_required_parent_and_child_nodes() -> None:
    budget = _completion_budget()
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=budget.contract.transitions,
        leaf_classes={"identifier": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    decoder = TreeActionDecoder(grammar)
    decoder.apply(OpenNodeAction("module"))
    decoder.apply(OpenScopeAction(0, 0, "module"))

    assert budget.child_candidates(
        decoder,
        parent_index=0,
        field="body",
        kinds=("function_definition",),
        remaining_nodes=2,
    ) == set()
    assert budget.child_candidates(
        decoder,
        parent_index=0,
        field="body",
        kinds=("function_definition",),
        remaining_nodes=3,
    ) == {"function_definition"}


def test_completion_budget_counts_existing_tree_obligations() -> None:
    budget = _completion_budget()
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=budget.contract.transitions,
        leaf_classes={"identifier": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    decoder = TreeActionDecoder(grammar)
    decoder.apply(OpenNodeAction("module"))
    decoder.apply(OpenScopeAction(0, 0, "module"))

    assert budget.minimum_required_nodes(decoder) == 1
    assert not budget.can_close_active_scope(decoder)


def test_completion_costs_are_cached_per_decoder_action(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = _completion_budget()
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=budget.contract.transitions,
        leaf_classes={"identifier": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    decoder = TreeActionDecoder(grammar)
    decoder.apply(OpenNodeAction("module"))
    decoder.apply(OpenScopeAction(0, 0, "module"))
    calls = 0
    original = budget.mask.minimum_additional_nodes

    def count_cost(context: ASTCompletionState) -> int:
        nonlocal calls
        calls += 1
        return original(context)

    monkeypatch.setattr(budget.mask, "minimum_additional_nodes", count_cost)
    assert budget.minimum_required_nodes(decoder) == 1
    assert budget.minimum_required_nodes(decoder) == 1
    assert calls == 1

    decoder.apply(AttachEdgeAction(0, "body"))
    assert budget.minimum_required_nodes(decoder) == 1
    assert calls == 2


def test_completion_contract_relation_indexes_are_reused() -> None:
    contract = _completion_budget().contract
    relation = ("module", True, "body")

    assert ("module", "body", "function_definition") in contract.transition_keys
    assert contract.field_rules_by_relation[relation].repeated is True
    assert contract.field_rules_by_relation is contract.field_rules_by_relation
    assert contract.alternatives_by_relation[relation] == frozenset(
        {("function_definition", True)}
    )


def test_anonymous_extra_child_does_not_consume_named_child_cardinality() -> None:
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=(
            ASTTransition("module", "child", "return", "module", "module", False),
            ASTTransition("module", "child", "identifier", "module", "module", False),
        ),
        leaf_classes={"identifier": ("identifier",), "return": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    schema = [
        {
            "type": "module",
            "named": True,
            "children": {
                "multiple": False,
                "required": False,
                "types": [{"type": "identifier", "named": True}],
            },
        },
        {"type": "identifier", "named": True},
        {"type": "return", "named": False},
    ]
    contract = ASTCompletionContract.from_payload(
        grammar,
        schema,
        schema_sha256=hashlib.sha256(b"anonymous-schema").hexdigest(),
    )
    assert contract.child_namedness("module", True, "child", "return") is False
    state = ASTCompletionState.open_parent_context(
        contract,
        "module",
        named=True,
        child_edges=(ChildEdge("child", "return", False),),
    )
    result = TerminalizableActionMask(contract).candidate_mask(
        state,
        field="child",
        candidates=(("identifier", True),),
        remaining_node_budget=1,
    )
    assert result.allowed_candidates == (("identifier", True),)


def test_action_selector_uses_certified_budget_mask_for_children() -> None:
    budget = _completion_budget()
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=budget.contract.transitions,
        leaf_classes={"identifier": ("identifier",)},
        corpus_sha256="a" * 64,
    )
    decoder = TreeActionDecoder(grammar)
    decoder.apply(OpenNodeAction("module"))
    decoder.apply(OpenScopeAction(0, 0, "module"))
    decoder.apply(AttachEdgeAction(0, "body"))
    vocabulary = {
        "node_ids": {"module": 0, "function_definition": 1, "identifier": 2},
        "node_names": ["module", "function_definition", "identifier"],
    }
    logits = {
        "action": torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]]),
        "node_kind": torch.tensor([[0.0, 10.0, 0.0]]),
    }
    args = (
        decoder,
        grammar,
        logits,
        vocabulary,
        "",
        [],
        2,
        viable_parent_kinds(grammar),
        terminalizable_kinds(grammar),
        budget,
    )
    with pytest.raises(ValueError, match="no legal action candidate"):
        _select_real_action(*args)

    args = (*args[:6], 4, *args[7:])
    assert _select_real_action(*args) == OpenNodeAction("function_definition")

