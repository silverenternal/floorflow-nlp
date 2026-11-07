from __future__ import annotations

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_transition_grammar import extract_transition_grammar


def _row(split: str, source: str) -> dict[str, object]:
    graph = parse_ast_graph(source, "python")
    return {
        "case_id": f"python-{split}",
        "language": "python",
        "source_sha256": graph.source_sha256,
        "split": split,
        "ast": serialize_ast_graph(graph),
    }


def test_transition_grammar_extracts_train_roots_edges_scopes_and_leaves() -> None:
    rows = [
        _row("train", "def add(a, b):\n    return a + b\n"),
        _row("validation", "def subtract(a, b):\n    return a - b\n"),
        _row("test", "class Box:\n    value = 1\n"),
    ]

    grammar, report = extract_transition_grammar(
        rows,
        corpus_sha256="a" * 64,
    )

    language = grammar["languages"]["python"]
    assert report["status"] == "pass"
    assert report["train_rows"] == 1
    assert report["excluded_validation_test_rows"] == 2
    assert language["legal_root_kinds"] == ["module"]
    assert any(
        transition["parent_kind"] == "module"
        and transition["edge_field"] == "child"
        for transition in language["transitions"]
    )
    assert any(
        transition["opens_scope"]
        for transition in language["transitions"]
    )
    assert any(rule["leaf_class"] == "identifier" for rule in language["leaf_classes"])
    assert report["validation_test_hash_overlap"] == []


def test_transition_grammar_is_deterministic() -> None:
    rows = [_row("train", "value = 1\n")]

    first, first_report = extract_transition_grammar(
        rows,
        corpus_sha256="b" * 64,
    )
    second, second_report = extract_transition_grammar(
        rows,
        corpus_sha256="b" * 64,
    )

    assert first == second
    assert first_report == second_report
