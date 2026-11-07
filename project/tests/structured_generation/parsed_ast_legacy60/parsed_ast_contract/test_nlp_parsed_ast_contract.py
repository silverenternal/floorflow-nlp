from __future__ import annotations

from dataclasses import replace

import pytest

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
    ModelQualificationBinding,
    full_ordered_graph_sha256,
    issue_parsed_ast_receipt,
    verify_parsed_ast_receipt,
)

SOURCE = "def add(left, right):\n    return left + right\n"
INPUT_HASH = "a" * 64


def _graph(source: str = SOURCE) -> ASTGraph:
    return parse_ast_graph(source, "python")


def _qualification(**changes) -> ModelQualificationBinding:
    values = {
        "checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "parser_sha256": "3" * 64,
        "canonicalizer_sha256": "4" * 64,
        "evaluator_sha256": "5" * 64,
        "languages": ("python",),
        "tracks": ("code", "engineering"),
        "freeze_manifest_sha256": "6" * 64,
        "mode": "fixture",
    }
    return ModelQualificationBinding(**(values | changes))


def _receipt(graph: ASTGraph | None = None, qualification=None):
    graph = graph or _graph()
    return issue_parsed_ast_receipt(
        graph,
        SOURCE,
        task_id="task-17",
        track="code",
        input_sha256=INPUT_HASH,
        qualification=qualification or _qualification(),
        parser_version="tree-sitter-pinned-v1",
        canonicalizer_version="ast-canonical-v1",
        grammar_version="grammar-v2",
        mode="fixture",
    )


def _replace_node(node: ASTNode, path: tuple[int, ...], update) -> ASTNode:
    if not path:
        return update(node)
    child_index = path[0]
    children = list(node.children)
    field, child = children[child_index]
    children[child_index] = (field, _replace_node(child, path[1:], update))
    return replace(node, children=tuple(children))


def _first_lexical_path(node: ASTNode, prefix: tuple[int, ...] = ()) -> tuple[int, ...]:
    if node.lexical_value is not None:
        return prefix
    for index, (_field, child) in enumerate(node.children):
        try:
            return _first_lexical_path(child, prefix + (index,))
        except LookupError:
            continue
    raise LookupError


def _replace_graph_root(graph: ASTGraph, root: ASTNode) -> ASTGraph:
    return replace(graph, root=root)


def test_receipt_binds_source_task_versions_graph_and_fixture_mode() -> None:
    graph = _graph()
    qualification = _qualification()
    receipt = _receipt(graph, qualification)

    verify_parsed_ast_receipt(
        receipt,
        graph,
        SOURCE,
        task_id="task-17",
        track="code",
        input_sha256=INPUT_HASH,
        qualification=qualification,
        parser_version="tree-sitter-pinned-v1",
        canonicalizer_version="ast-canonical-v1",
        grammar_version="grammar-v2",
    )
    assert receipt.mode == "fixture"
    assert len(receipt.full_ordered_canonical_graph_sha256) == 64
    with pytest.raises(ValueError, match="stale"):
        verify_parsed_ast_receipt(
            receipt, graph, SOURCE, task_id="other-task", track="code",
            input_sha256=INPUT_HASH, qualification=qualification,
            parser_version="tree-sitter-pinned-v1", canonicalizer_version="ast-canonical-v1",
            grammar_version="grammar-v2",
        )
    with pytest.raises(ValueError, match="differs from the supplied source"):
        issue_parsed_ast_receipt(
            graph, SOURCE + "# changed\n", task_id="task-17", track="code",
            input_sha256=INPUT_HASH, qualification=qualification,
            parser_version="tree-sitter-pinned-v1", canonicalizer_version="ast-canonical-v1",
            grammar_version="grammar-v2", mode="fixture",
        )


def test_full_graph_hash_detects_descendant_lexical_scope_order_and_topology_edits() -> None:
    graph = _graph()
    original = full_ordered_graph_sha256(graph)
    leaf_path = _first_lexical_path(graph.root)
    lexical_edit = _replace_graph_root(
        graph,
        _replace_node(graph.root, leaf_path, lambda node: replace(node, lexical_value=(node.lexical_value or "") + "x")),
    )
    scope_edit = _replace_graph_root(
        graph,
        _replace_node(graph.root, leaf_path, lambda node: replace(node, scope_id=node.scope_id + 100)),
    )
    assert full_ordered_graph_sha256(lexical_edit) != original
    assert full_ordered_graph_sha256(scope_edit) != original

    multi_child_path = None
    stack = [(graph.root, ())]
    while stack:
        node, path = stack.pop()
        if len(node.children) > 1:
            multi_child_path = path
            break
        stack.extend((child, path + (index,)) for index, (_field, child) in enumerate(node.children))
    assert multi_child_path is not None
    parent = graph.root
    for index in multi_child_path:
        parent = parent.children[index][1]
    reordered = list(parent.children)
    reordered.reverse()
    order_edit = _replace_graph_root(
        graph,
        _replace_node(graph.root, multi_child_path, lambda node: replace(node, children=tuple(reordered))),
    )
    assert full_ordered_graph_sha256(order_edit) != original

    topology = _replace_graph_root(
        graph,
        _replace_node(graph.root, leaf_path[:-1], lambda node: replace(node, children=())),
    )
    assert full_ordered_graph_sha256(topology) != original


def test_qualification_id_changes_with_weights_parser_or_track_scope() -> None:
    base = _qualification()
    changed_checkpoint = replace(base, checkpoint_sha256="7" * 64)
    changed_parser = replace(base, parser_sha256="8" * 64)
    changed_track_scope = replace(base, tracks=("engineering",))
    assert base.qualification_id != changed_checkpoint.qualification_id
    assert base.qualification_id != changed_parser.qualification_id
    assert base.qualification_id != changed_track_scope.qualification_id
    with pytest.raises(ValueError, match="does not cover this track"):
        _receipt(qualification=changed_track_scope)


def test_receipt_rejects_changed_parser_canonicalizer_or_grammar_versions() -> None:
    graph = _graph()
    qualification = _qualification()
    receipt = _receipt(graph, qualification)
    for values in (
        {"parser_version": "tree-sitter-updated"},
        {"canonicalizer_version": "canonical-updated"},
        {"grammar_version": "grammar-updated"},
    ):
        with pytest.raises(ValueError, match="stale"):
            verify_parsed_ast_receipt(
                receipt, graph, SOURCE, task_id="task-17", track="code",
                input_sha256=INPUT_HASH, qualification=qualification,
                parser_version=values.get("parser_version", "tree-sitter-pinned-v1"),
                canonicalizer_version=values.get("canonicalizer_version", "ast-canonical-v1"),
                grammar_version=values.get("grammar_version", "grammar-v2"),
            )


def test_fixture_qualification_cannot_issue_a_production_receipt() -> None:
    with pytest.raises(ValueError, match="mode differs"):
        issue_parsed_ast_receipt(
            _graph(), SOURCE, task_id="task-17", track="code", input_sha256=INPUT_HASH,
            qualification=_qualification(), parser_version="p1", canonicalizer_version="c1",
            grammar_version="g1", mode="production",
        )
