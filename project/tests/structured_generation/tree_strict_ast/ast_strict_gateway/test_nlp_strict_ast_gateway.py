from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
    ModelQualificationBinding,
    canonicalizer_runtime_identity,
    full_ordered_graph_sha256,
    parser_runtime_identity,
)
from newera.nlp_flow.components.ast.diagnostics.strict.strict_ast_gateway import ASTGatewayFailure, ASTGatewaySuccess, commit_typed_ast
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import trace_teacher_actions
from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import TypedLexicalValue

SOURCE = "def add(left, right):\n    return left + right\n"
ROOT = Path(__file__).resolve().parents[5]


def _fixture():
    graph = parse_ast_graph(SOURCE, "python")
    payload = json.loads(serialize_ast_graph(graph))
    grammar = load_ast_grammar(ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json", "python")
    actions = tuple(trace_teacher_actions(grammar, payload))
    values = []
    node_id = 0

    def visit(node: ASTNode) -> None:
        nonlocal node_id
        current_id = node_id
        node_id += 1
        if node.children:
            for _field, child in node.children:
                visit(child)
            return
        assert node.lexical_value is not None
        values.append(
            TypedLexicalValue(
                node_id=current_id,
                node_kind=node.node_type,
                leaf_class=classify_leaf(node.node_type, node.lexical_value),
                scope_id=node.scope_id,
                value=node.lexical_value,
                origin="generated",
                minimum_length=0,
                maximum_length=max(1, len(node.lexical_value)),
            )
        )

    visit(graph.root)
    qualification = ModelQualificationBinding(
        checkpoint_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        parser_sha256="3" * 64,
        canonicalizer_sha256="4" * 64,
        evaluator_sha256="5" * 64,
        languages=("python",),
        tracks=("code",),
        freeze_manifest_sha256="6" * 64,
        mode="fixture",
    )
    return graph, tuple(values), actions, qualification


def _commit(graph, values, actions, qualification):
    return commit_typed_ast(
        graph,
        values,
        actions,
        task_id="gateway-case",
        track="code",
        input_sha256=hashlib.sha256(b"user request").hexdigest(),
        qualification=qualification,
        parser_version="fixture-parser",
        canonicalizer_version="fixture-canonicalizer",
        grammar_version="fixture-grammar",
    )


def test_committed_tree_emits_source_and_receipt_from_one_parser_invocation(monkeypatch) -> None:
    import newera.nlp_flow.components.generation.trees.typed.typed_source_renderer as renderer_module

    graph, values, actions, qualification = _fixture()
    parser = renderer_module.parse_ast_graph
    calls = []

    def count_parser(source: str, language: str):
        calls.append((source, language))
        return parser(source, language)

    monkeypatch.setattr(renderer_module, "parse_ast_graph", count_parser)
    output = _commit(graph, values, actions, qualification)

    assert isinstance(output, ASTGatewaySuccess)
    assert output.source == SOURCE
    assert output.parsed_graph is not graph
    assert output.receipt.full_ordered_canonical_graph_sha256 == full_ordered_graph_sha256(output.parsed_graph)
    assert calls == [(SOURCE, "python")]
    assert output.original_actions is actions


def test_invalid_lexical_projection_preserves_original_actions_without_repair() -> None:
    graph, values, actions, qualification = _fixture()
    changed = tuple(
        replace(value, value="*") if value.value == "+" else value
        for value in values
    )

    output = _commit(graph, changed, actions, qualification)

    assert isinstance(output, ASTGatewayFailure)
    assert output.kind == "parse_or_replay_failure"
    assert output.original_actions is actions
    assert "changed the committed AST" in output.message


def test_unsupported_language_and_wrong_scope_fail_before_rendering() -> None:
    graph, values, actions, qualification = _fixture()
    unsupported = _commit(replace(graph, language="unknown"), values, actions, qualification)
    wrong_track = _commit(
        graph,
        values,
        actions,
        replace(qualification, tracks=("engineering",)),
    )

    assert isinstance(unsupported, ASTGatewayFailure)
    assert unsupported.kind == "unsupported"
    assert unsupported.original_actions is actions
    assert isinstance(wrong_track, ASTGatewayFailure)
    assert wrong_track.kind == "receipt_failure"
    assert wrong_track.original_actions is actions


def test_production_gateway_requires_current_parser_and_canonicalizer_fingerprints() -> None:
    graph, values, actions, fixture_qualification = _fixture()
    parser_version, parser_hash = parser_runtime_identity("python")
    canonicalizer_version, canonicalizer_hash = canonicalizer_runtime_identity()
    qualification = replace(
        fixture_qualification,
        parser_sha256=parser_hash,
        canonicalizer_sha256=canonicalizer_hash,
        mode="production",
    )
    result = commit_typed_ast(
        graph,
        values,
        actions,
        task_id="gateway-prod-case",
        track="code",
        input_sha256=hashlib.sha256(b"request").hexdigest(),
        qualification=qualification,
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=parser_version.rsplit(":grammar-", maxsplit=1)[-1],
    )

    assert isinstance(result, ASTGatewaySuccess)
    assert result.receipt.mode == "production"
