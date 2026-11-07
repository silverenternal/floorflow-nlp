"""Strict serialization for canonical AST nodes and graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from newera.nlp_flow.components.ast.representation.codec.ast_codec import (
    _MODULES,
    SERIALIZATION_VERSION,
    SUPPORTED_LANGUAGES,
    ASTGraph,
    ASTNode,
    ASTScope,
    ASTSymbol,
    _validate_ast_or_raise,
)


def _payload(node: ASTNode) -> dict[str, Any]:
    return {
        "type": node.node_type,
        "start": node.start_byte,
        "end": node.end_byte,
        "named": node.named,
        "lexical": node.lexical_value,
        "gap_before": node.gap_before,
        "scope_id": node.scope_id,
        "symbol_name": node.symbol_name,
        "symbol_kind": node.symbol_kind,
        "language": node.language,
        "children": [
            {"field": field, "node": _payload(child)}
            for field, child in node.children
        ],
    }


def serialize_ast(node: ASTNode) -> str:
    """Serialize a deterministic graph payload after validating invariants."""

    _validate_ast_or_raise(node)
    payload = {"version": SERIALIZATION_VERSION, "root": _payload(node)}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _scope_payload(scope: ASTScope) -> dict[str, Any]:
    return {
        "scope_id": scope.scope_id,
        "node_type": scope.node_type,
        "start": scope.start_byte,
        "end": scope.end_byte,
        "parent_scope_id": scope.parent_scope_id,
        "path": list(scope.path),
    }


def _symbol_payload(symbol: ASTSymbol) -> dict[str, Any]:
    return {
        "name": symbol.name,
        "kind": symbol.kind,
        "node_type": symbol.node_type,
        "scope_id": symbol.scope_id,
        "scope_path": list(symbol.scope_path),
        "start": symbol.start_byte,
        "end": symbol.end_byte,
    }


def serialize_ast_graph(graph: ASTGraph) -> str:
    """Serialize a complete reversible AST, scope, symbol, and provenance graph."""

    _validate_ast_or_raise(graph)
    payload = {
        "version": SERIALIZATION_VERSION,
        "language": graph.language,
        "source_sha256": graph.source_sha256,
        "root": _payload(graph.root),
        "scopes": [_scope_payload(scope) for scope in graph.scopes],
        "symbols": [_symbol_payload(symbol) for symbol in graph.symbols],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate serialized AST key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite serialized AST number: {value}")


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _from_payload(payload: Mapping[str, Any], *, is_root: bool = False) -> ASTNode:
    expected = {
        "type",
        "start",
        "end",
        "named",
        "lexical",
        "gap_before",
        "scope_id",
        "symbol_name",
        "symbol_kind",
        "language",
        "children",
    }
    if set(payload) != expected:
        raise ValueError(f"AST node fields must be exactly {sorted(expected)}")
    node_type = payload["type"]
    if not isinstance(node_type, str) or not node_type:
        raise TypeError("AST node type must be a non-empty string")
    named = payload["named"]
    if not isinstance(named, bool):
        raise TypeError("AST named must be boolean")
    children_payload = payload["children"]
    if not isinstance(children_payload, list):
        raise TypeError("AST children must be a list")
    children = []
    for item in children_payload:
        if not isinstance(item, Mapping) or set(item) != {"field", "node"}:
            raise ValueError("AST child must contain exactly field and node")
        field = item["field"]
        if not isinstance(field, str) or not field:
            raise TypeError("AST child field must be a non-empty string")
        if not isinstance(item["node"], Mapping):
            raise TypeError("AST child node must be an object")
        children.append((field, _from_payload(item["node"])))
    language = _optional_string(payload["language"], "language")
    if is_root and language not in _MODULES:
        raise ValueError(f"root language must be one of {SUPPORTED_LANGUAGES}")
    if not is_root and language is not None:
        raise ValueError("language is only valid on the AST root")
    return ASTNode(
        node_type=node_type,
        start_byte=_integer(payload["start"], "start"),
        end_byte=_integer(payload["end"], "end"),
        named=named,
        children=tuple(children),
        lexical_value=_optional_string(payload["lexical"], "lexical"),
        gap_before=_optional_string(payload["gap_before"], "gap_before") or "",
        scope_id=_integer(payload["scope_id"], "scope_id"),
        symbol_name=_optional_string(payload["symbol_name"], "symbol_name"),
        symbol_kind=_optional_string(payload["symbol_kind"], "symbol_kind"),
        language=language,
    )


def deserialize_ast(serialized: str) -> ASTNode:
    payload = json.loads(
        serialized,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, Mapping) or set(payload) != {"version", "root"}:
        raise ValueError("serialized AST must contain exactly version and root")
    if payload["version"] != SERIALIZATION_VERSION:
        raise ValueError(f"unsupported AST serialization: {payload['version']}")
    if not isinstance(payload["root"], Mapping):
        raise TypeError("serialized AST root must be an object")
    node = _from_payload(payload["root"], is_root=True)
    _validate_ast_or_raise(node)
    return node


def _from_scope_payload(payload: Mapping[str, Any]) -> ASTScope:
    expected = {"scope_id", "node_type", "start", "end", "parent_scope_id", "path"}
    if set(payload) != expected:
        raise ValueError(f"AST scope fields must be exactly {sorted(expected)}")
    node_type = payload["node_type"]
    if not isinstance(node_type, str) or not node_type:
        raise TypeError("AST scope node type must be a non-empty string")
    path = payload["path"]
    if not isinstance(path, list) or any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in path
    ):
        raise TypeError("AST scope path must be a list of integers")
    parent_scope_id = payload["parent_scope_id"]
    if parent_scope_id is not None:
        parent_scope_id = _integer(parent_scope_id, "parent_scope_id")
    return ASTScope(
        scope_id=_integer(payload["scope_id"], "scope_id"),
        node_type=node_type,
        start_byte=_integer(payload["start"], "start"),
        end_byte=_integer(payload["end"], "end"),
        parent_scope_id=parent_scope_id,
        path=tuple(path),
    )


def _from_symbol_payload(payload: Mapping[str, Any]) -> ASTSymbol:
    expected = {
        "name",
        "kind",
        "node_type",
        "scope_id",
        "scope_path",
        "start",
        "end",
    }
    if set(payload) != expected:
        raise ValueError(f"AST symbol fields must be exactly {sorted(expected)}")
    strings = []
    for field in ("name", "kind", "node_type"):
        value = payload[field]
        if not isinstance(value, str) or not value:
            raise TypeError(f"AST symbol {field} must be a non-empty string")
        strings.append(value)
    scope_path = payload["scope_path"]
    if not isinstance(scope_path, list) or any(
        not isinstance(item, str) or not item
        for item in scope_path
    ):
        raise TypeError("AST symbol scope_path must be a string list")
    return ASTSymbol(
        name=strings[0],
        kind=strings[1],
        node_type=strings[2],
        scope_id=_integer(payload["scope_id"], "scope_id"),
        scope_path=tuple(scope_path),
        start_byte=_integer(payload["start"], "start"),
        end_byte=_integer(payload["end"], "end"),
    )


def deserialize_ast_graph(serialized: str) -> ASTGraph:
    payload = json.loads(
        serialized,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    expected = {"version", "language", "source_sha256", "root", "scopes", "symbols"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"serialized AST graph fields must be exactly {sorted(expected)}")
    if payload["version"] != SERIALIZATION_VERSION:
        raise ValueError(f"unsupported AST serialization: {payload['version']}")
    language = payload["language"]
    if language not in _MODULES:
        raise ValueError(f"unsupported AST language: {language}")
    source_sha256 = payload["source_sha256"]
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(payload["root"], Mapping):
        raise TypeError("serialized AST graph root must be an object")
    if not isinstance(payload["scopes"], list) or not isinstance(payload["symbols"], list):
        raise TypeError("serialized AST graph scopes and symbols must be lists")
    graph = ASTGraph(
        language=language,
        root=_from_payload(payload["root"], is_root=True),
        scopes=tuple(_from_scope_payload(scope) for scope in payload["scopes"]),
        symbols=tuple(_from_symbol_payload(symbol) for symbol in payload["symbols"]),
        source_sha256=source_sha256,
    )
    _validate_ast_or_raise(graph)
    return graph


__all__ = [
    "deserialize_ast",
    "deserialize_ast_graph",
    "serialize_ast",
    "serialize_ast_graph",
]
