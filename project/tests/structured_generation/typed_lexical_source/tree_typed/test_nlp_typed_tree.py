from __future__ import annotations

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast
from newera.nlp_flow.components.generation.outputs.json.json_tree import build_json_tree
from newera.nlp_flow.components.generation.trees.typed.typed_tree import (
    from_ast_node,
    from_json_node,
    render_typed_tree,
    tree_edge_count,
    tree_node_count,
    tree_signature,
    validity_for_tree,
)


def test_ast_and_json_share_the_same_typed_tree_contract() -> None:
    ast_tree = from_ast_node(
        parse_ast("def add(a, b):\n    return a + b\n", "python")
    )
    json_tree = from_json_node(
        build_json_tree({"status": "ok", "count": 2})
    )

    assert ast_tree.domain == "ast"
    assert json_tree.domain == "json"
    assert tree_node_count(ast_tree) > tree_node_count(json_tree)
    assert tree_edge_count(ast_tree) > 0
    assert tree_signature(json_tree)[0][0] == "json"
    assert tree_signature(json_tree)[1][0].startswith("object.entry[")
    assert render_typed_tree(ast_tree) == "def add(a, b):\n    return a + b\n"
    assert render_typed_tree(json_tree) == '{"status":"ok","count":2}'


def test_typed_tree_validity_is_explicit_and_domain_checked() -> None:
    tree = from_json_node(build_json_tree({"ok": True}))
    validity = validity_for_tree(
        tree,
        structural=True,
        schema=True,
        executable=True,
    )

    assert validity.all_valid
