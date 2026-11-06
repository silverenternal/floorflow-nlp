"""Shared typed-tree representation for AST, JSON, patches, and tool actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from floorflow.nlp_flow._generated_protocol import (
    OUTPUT_MODE_NAMES,
    TREE_DOMAIN_NAMES,
)
from floorflow.nlp_flow.ast_codec import ASTNode
from floorflow.nlp_flow.json_tree import JSONNode

TREE_DOMAINS = TREE_DOMAIN_NAMES
MODES = OUTPUT_MODE_NAMES


@dataclass(frozen=True)
class TypedTreeEdge:
    field: str
    node: TypedTreeNode


@dataclass(frozen=True)
class TypedTreeNode:
    domain: str
    kind: str
    value: Any = None
    children: tuple[TypedTreeEdge, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.domain not in TREE_DOMAINS:
            raise ValueError(f"unsupported tree domain: {self.domain}")
        if not self.kind:
            raise ValueError("tree node kind is required")


@dataclass(frozen=True)
class TypedTreeValidity:
    structural: bool
    schema: bool
    executable: bool

    @property
    def all_valid(self) -> bool:
        return self.structural and self.schema and self.executable


def from_ast_node(node: ASTNode) -> TypedTreeNode:
    """Convert a parsed source AST into the shared typed-tree representation."""

    metadata = (
        ("scope_id", node.scope_id),
        ("symbol_name", node.symbol_name),
        ("symbol_kind", node.symbol_kind),
        ("gap_before", node.gap_before),
    )
    return TypedTreeNode(
        domain="ast",
        kind=node.node_type,
        value=node.lexical_value,
        children=tuple(
            TypedTreeEdge(field, from_ast_node(child))
            for field, child in node.children
        ),
        metadata=metadata,
    )


def from_json_node(
    node: JSONNode,
    *,
    domain: str = "json",
) -> TypedTreeNode:
    """Convert strict JSON structure into the same typed-tree representation."""

    if node.kind == "object":
        return TypedTreeNode(
            domain=domain,
            kind="object",
            children=tuple(
                TypedTreeEdge(
                    f"entry[{index}]",
                    TypedTreeNode(
                        domain=domain,
                        kind="entry",
                        children=(
                            TypedTreeEdge("key", from_json_node(key)),
                            TypedTreeEdge("value", from_json_node(value)),
                        ),
                    ),
                )
                for index, (key, value) in enumerate(node.children)
            ),
        )
    if node.kind == "array":
        return TypedTreeNode(
            domain=domain,
            kind="array",
            children=tuple(
                TypedTreeEdge(f"item[{index}]", from_json_node(child))
                for index, child in enumerate(node.children)
            ),
        )
    return TypedTreeNode(
        domain=domain,
        kind=node.kind,
        value=node.value,
    )


def tree_node_count(root: TypedTreeNode) -> int:
    return 1 + sum(tree_node_count(edge.node) for edge in root.children)


def tree_edge_count(root: TypedTreeNode) -> int:
    return sum(1 + tree_edge_count(edge.node) for edge in root.children)


def tree_signature(root: TypedTreeNode) -> tuple[tuple[str, str, str], ...]:
    rows = [(root.domain, root.kind, str(root.value))]
    for edge in root.children:
        rows.extend(
            (f"{root.kind}.{edge.field}", domain, kind)
            for domain, kind, _value in tree_signature(edge.node)
        )
    return tuple(rows)


def validity_for_tree(
    root: TypedTreeNode,
    *,
    structural: bool,
    schema: bool,
    executable: bool,
) -> TypedTreeValidity:
    if root.domain not in TREE_DOMAINS:
        structural = False
    return TypedTreeValidity(
        structural=structural,
        schema=schema,
        executable=executable,
    )


def _render_ast_tree(node: TypedTreeNode) -> str:
    metadata = dict(node.metadata)
    gap = str(metadata.get("gap_before", ""))
    value = "" if node.value is None else str(node.value)
    return (
        gap
        + value
        + "".join(_render_ast_tree(edge.node) for edge in node.children)
    )


def _to_json_value(node: TypedTreeNode) -> Any:
    if node.kind == "object":
        output = {}
        for entry_edge in node.children:
            entry = entry_edge.node
            key_edge = next(edge for edge in entry.children if edge.field == "key")
            value_edge = next(
                edge for edge in entry.children if edge.field == "value"
            )
            output[str(key_edge.node.value)] = _to_json_value(value_edge.node)
        return output
    if node.kind == "array":
        return [_to_json_value(edge.node) for edge in node.children]
    return node.value


def render_typed_tree(root: TypedTreeNode) -> str:
    if root.domain == "ast":
        return _render_ast_tree(root)
    if root.domain == "json":
        import json

        return json.dumps(
            _to_json_value(root),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    if root.domain in {"patch", "tool_action"}:
        import json

        return json.dumps(
            _to_json_value(root),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    raise ValueError(f"cannot render tree domain: {root.domain}")


__all__ = [
    "MODES",
    "TREE_DOMAINS",
    "TypedTreeEdge",
    "TypedTreeNode",
    "TypedTreeValidity",
    "from_ast_node",
    "from_json_node",
    "render_typed_tree",
    "tree_edge_count",
    "tree_node_count",
    "tree_signature",
    "validity_for_tree",
]
