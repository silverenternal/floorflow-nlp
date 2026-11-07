from __future__ import annotations

import hashlib

import pytest

from newera.nlp_flow.components.ast.diagnostics.validation.ast_completion_contract import ASTCompletionContract
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransition, ASTTransitionGrammar


def _contract() -> ASTCompletionContract:
    grammar = ASTTransitionGrammar(
        language="python",
        legal_root_kinds=("module",),
        transitions=(
            ASTTransition("module", "name", "identifier", "module", "module", False),
            ASTTransition("module", "name", "integer", "module", "module", False),
            ASTTransition("module", "body", "function_definition", "module", "function_definition", True),
            ASTTransition(
                "function_definition", "name", "identifier", "function_definition", "function_definition", False
            ),
        ),
        leaf_classes={"identifier": ("identifier",), "integer": ("integer",)},
        corpus_sha256="a" * 64,
    )
    schema = [
        {
            "type": "module",
            "named": True,
            "fields": {
                "name": {
                    "multiple": False,
                    "required": True,
                    "types": [{"type": "identifier", "named": True}],
                },
                "body": {
                    "multiple": True,
                    "required": False,
                    "types": [{"type": "_statement", "named": True}],
                },
            },
        },
        {
            "type": "_statement",
            "named": True,
            "subtypes": [
                {"type": "function_definition", "named": True},
                {"type": "expression_statement", "named": True},
            ],
        },
        {"type": "function_definition", "named": True, "fields": {
            "name": {
                "multiple": False,
                "required": True,
                "types": [{"type": "identifier", "named": True}],
            }
        }},
        {"type": "identifier", "named": True},
        {"type": "expression_statement", "named": True},
    ]
    return ASTCompletionContract.from_payload(
        grammar,
        schema,
        schema_sha256=hashlib.sha256(b"fixture-schema").hexdigest(),
    )


def test_contract_expands_alternatives_and_reuses_same_immutable_state() -> None:
    contract = _contract()
    state = contract.initial_state()
    state.validate_lexical_class("identifier", "identifier")
    with pytest.raises(ValueError, match="lexical class"):
        state.validate_lexical_class("identifier", "integer")
    body = next(rule for rule in contract.fields if rule.parent_kind == "module" and rule.field == "body")
    assert {item.kind for item in body.alternatives} == {"expression_statement", "function_definition"}
    assert body.repeated and not body.required
    assert contract.training_state() == contract.inference_state()
    assert contract.training_state() is not contract.inference_state()


def test_required_fields_alternatives_cardinality_and_child_order_are_enforced() -> None:
    contract = _contract()
    state = contract.initial_state().start_root("module")
    with pytest.raises(ValueError, match="required parser fields"):
        state.close_node()

    with pytest.raises(ValueError, match="outside parser alternatives"):
        state.append_child("name", "integer")
    state = state.append_child("name", "identifier").close_node()
    with pytest.raises(ValueError, match="non-repeated"):
        state.append_child("name", "identifier")

    state = state.append_child("body", "function_definition")
    state = state.append_child("name", "identifier").close_node()
    state = state.close_node()
    state = state.close_node()
    module_children = next(edges for kind, edges in state.child_order if kind == "module")
    assert [(edge.field, edge.child_kind, edge.child_named) for edge in module_children] == [
        ("name", "identifier", True),
        ("body", "function_definition", True),
    ]
    assert state.eos_ready
    assert state.finish().finished


def test_scope_closure_and_eos_require_complete_nodes() -> None:
    contract = _contract()
    state = contract.initial_state().start_root("module")
    with pytest.raises(ValueError, match="EOS requires"):
        state.finish()
    state = state.append_child("name", "identifier").close_node()
    state = state.append_child("body", "function_definition")
    assert state.open_scopes == (0, 1)
    state = state.append_child("name", "identifier").close_node().close_node()
    assert state.open_scopes == (0,)
    state = state.close_node()
    assert state.open_scopes == ()
    assert state.finish().finished


def test_named_and_anonymous_forms_have_distinct_field_obligations() -> None:
    contract = _contract()
    state = contract.initial_state().start_root("module", named=False)
    with pytest.raises(ValueError, match="required parser fields"):
        contract.initial_state().start_root("module", named=True).close_node()
    assert state.close_node().finish().finished


def test_from_files_rejects_unpinned_parser_schema(tmp_path) -> None:
    contract = _contract()
    path = tmp_path / "nodes.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned identity"):
        ASTCompletionContract.from_files(
            ASTTransitionGrammar(
                language=contract.language,
                legal_root_kinds=contract.legal_root_kinds,
                transitions=contract.transitions,
                leaf_classes=dict(contract.lexical_classes),
                corpus_sha256="a" * 64,
            ),
            path,
            expected_schema_sha256="0" * 64,
        )
