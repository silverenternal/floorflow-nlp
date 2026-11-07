from __future__ import annotations

import pytest

from newera.nlp_flow.components.ast.representation.codec.ast_codec import (
    ast_metrics,
    deserialize_ast,
    normalized_tree_edit_distance,
    parse_ast,
    parser_fingerprint,
    render_source,
    serialize_ast,
    validate_ast,
)


@pytest.mark.parametrize(
    ("language", "source", "root"),
    [
        ("python", "def add(a, b):\n    return a + b\n", "module"),
        ("rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n", "source_file"),
        ("cpp", "int add(int a, int b) { return a + b; }\n", "translation_unit"),
    ],
)
def test_multilanguage_ast_round_trip(language: str, source: str, root: str) -> None:
    node = parse_ast(source, language)
    restored = deserialize_ast(serialize_ast(node))

    assert node.node_type == root
    assert restored == node


def test_ast_similarity_is_perfect_for_identity_and_lower_for_change() -> None:
    reference = parse_ast("def add(a, b):\n    return a + b\n", "python")
    predicted = parse_ast("def add(a, b):\n    return a - b\n", "python")

    identity = ast_metrics(reference, reference)
    changed = ast_metrics(predicted, reference)

    assert identity == {
        "lexical_f1": 1.0,
        "node_type_f1": 1.0,
        "typed_edge_f1": 1.0,
        "scope_relation_f1": 1.0,
        "scope_accuracy": 1.0,
        "parse_validity": True,
        "normalized_tree_edit_distance": 0.0,
    }
    assert changed["node_type_f1"] < 1.0
    assert changed["normalized_tree_edit_distance"] > 0.0


def test_graph_round_trip_preserves_lexical_scope_and_symbol_relations() -> None:
    source = "def add(a, b):\n    total = a + b\n    return total\n"
    graph = parse_ast(source, "python")
    restored = deserialize_ast(serialize_ast(graph))

    assert restored == graph
    validate_ast(restored, source=source)
    assert render_source(restored) == source
    symbols = [
        (node.symbol_name, node.symbol_kind)
        for _path, node in _walk_for_test(restored)
        if node.symbol_name is not None
    ]
    assert ("add", "function") in symbols


def test_tree_edit_distance_is_zero_for_identity_and_positive_for_relabel() -> None:
    reference = parse_ast("def add(a, b):\n    return a + b\n", "python")
    changed = parse_ast("def add(a, b):\n    return a - b\n", "python")

    assert normalized_tree_edit_distance(reference, reference) == 0.0
    assert normalized_tree_edit_distance(changed, reference) > 0.0


def test_parser_provenance_is_frozen() -> None:
    fingerprint = parser_fingerprint("python")

    assert fingerprint.package_version == "0.25.0"
    assert fingerprint.abi_version == 15
    assert len(fingerprint.grammar_sha256) == 64
    assert len(fingerprint.native_sha256) == 64


def _walk_for_test(node):
    yield (), node
    for index, (_field, child) in enumerate(node.children):
        yield from _walk_for_test(child)


def test_malformed_source_fails_parse() -> None:
    with pytest.raises(ValueError, match="parse error"):
        parse_ast("def broken(:\n", "python")
