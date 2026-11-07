"""Parser recovery inspection kept separate from production AST conversion."""

from __future__ import annotations

from tree_sitter import Node, Parser

from newera.nlp_flow.components.ast.representation.codec.ast_codec import language_for

_EXPECTED_ROOTS = {
    "python": "module",
    "rust": "source_file",
    "cpp": "translation_unit",
}


def inspect_parse_recovery(source: str, language: str) -> dict[str, object]:
    """Inspect parser recovery nodes for evaluation without repairing source."""
    if language not in _EXPECTED_ROOTS:
        raise ValueError(f"unsupported AST language: {language}")
    tree = Parser(language_for(language)).parse(source.encode("utf-8"))
    error_node_count = 0
    missing_node_count = 0

    def visit(node: Node) -> None:
        nonlocal error_node_count, missing_node_count
        error_node_count += int(node.type == "ERROR")
        missing_node_count += int(node.is_missing)
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return {
        "root_type": tree.root_node.type,
        "expected_root_type": _EXPECTED_ROOTS[language],
        "has_error": bool(tree.root_node.has_error),
        "error_node_count": error_node_count,
        "missing_node_count": missing_node_count,
    }


__all__ = ["inspect_parse_recovery"]
