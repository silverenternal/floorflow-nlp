"""Strict typed JSON tree serialization and schema validation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JSONNode:
    kind: str
    value: Any
    children: tuple[Any, ...] = ()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_object_pairs,
    )


def build_json_tree(value: Any) -> JSONNode:
    if value is None:
        return JSONNode("null", None)
    if isinstance(value, bool):
        return JSONNode("boolean", value)
    if isinstance(value, int):
        return JSONNode("integer", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON numbers are forbidden")
        return JSONNode("number", value)
    if isinstance(value, str):
        return JSONNode("string", value)
    if isinstance(value, list):
        children = tuple(build_json_tree(item) for item in value)
        return JSONNode("array", None, children)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        children = tuple(
            (build_json_tree(key), build_json_tree(item))
            for key, item in value.items()
        )
        return JSONNode("object", None, children)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def tree_to_python(node: JSONNode) -> Any:
    if node.kind == "object":
        output = {}
        for key_node, value_node in node.children:
            key = tree_to_python(key_node)
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = tree_to_python(value_node)
        return output
    if node.kind == "array":
        return [tree_to_python(item) for item in node.children]
    return node.value


def serialize_json_tree(root: JSONNode) -> str:
    value = tree_to_python(root)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type: {expected}")


def validate_schema(value: Any, schema: Mapping[str, Any]) -> None:
    expected = schema.get("type")
    if expected is not None and not _matches_type(value, str(expected)):
        raise ValueError(f"expected {expected}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"value is not in enum: {value!r}")
    if expected in {"integer", "number"}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON numbers are forbidden")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError("number is below schema minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError("number is above schema maximum")
    if expected == "object":
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"object is missing required keys: {missing}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key])
            elif additional is False:
                raise ValueError(f"unexpected object key: {key}")
            elif isinstance(additional, Mapping):
                validate_schema(item, additional)
    if expected == "array" and "items" in schema:
        for item in value:
            validate_schema(item, schema["items"])


def load_schema(path: str | Path) -> dict[str, Any]:
    payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON schema must be an object")
    return payload


def parse_and_validate(text: str, schema: Mapping[str, Any]) -> JSONNode:
    value = strict_json_loads(text)
    validate_schema(value, schema)
    return build_json_tree(value)


__all__ = [
    "JSONNode",
    "build_json_tree",
    "load_schema",
    "parse_and_validate",
    "serialize_json_tree",
    "strict_json_loads",
    "tree_to_python",
    "validate_schema",
]
