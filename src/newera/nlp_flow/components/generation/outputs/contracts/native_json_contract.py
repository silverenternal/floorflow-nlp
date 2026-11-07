"""Schema-constrained JSON decisions and commit-only serialization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from newera.nlp_flow.model.components.tasks.output.json_grammar import SchemaGrammarState, SchemaNode
from newera.nlp_flow.components.generation.outputs.contracts.structured_generation import (
    StructuredGenerationResult,
    TypedTreeGateway,
)
from newera.nlp_flow.components.generation.trees.typed.typed_tree import TypedTreeNode

JSON_OUTPUT_KINDS = ("ast", "patch", "tool_action", "diagnostic", "engineering_result")
TOOL_NAMES = ("run_tests", "read_file", "write_file")


@dataclass(frozen=True)
class NativeJSONDecision:
    action: str
    key: str | None = None
    value: Any = None


@dataclass(frozen=True)
class NativeJSONCommit:
    value: dict[str, Any]
    text: str
    decisions: tuple[NativeJSONDecision, ...]
    typed_tree: TypedTreeNode
    structured: StructuredGenerationResult


class InvalidNativeJSONDecision(ValueError):
    """A model decision is invalid under the selected output schema."""


def _object(properties: dict[str, SchemaNode], required: tuple[str, ...] | None = None) -> SchemaNode:
    return SchemaNode("object", tuple(properties) if required is None else required, properties)


def _array(item: SchemaNode) -> SchemaNode:
    return SchemaNode("array", items=item)


def _string(*, enum: tuple[str, ...] = ()) -> SchemaNode:
    return SchemaNode("string", enum_values=enum)


def _integer(*, minimum: float | None = None, maximum: float | None = None) -> SchemaNode:
    return SchemaNode("integer", minimum=minimum, maximum=maximum)


def output_schema(kind: str, *, tool_name: str = "run_tests") -> SchemaNode:
    if kind not in JSON_OUTPUT_KINDS:
        raise ValueError(f"unsupported native JSON output kind: {kind}")
    if kind == "ast":
        payload = _object(
            {
                "language": _string(enum=("python", "rust", "cpp")),
                "nodes": _array(_object({"index": _integer(minimum=0), "kind": _string()})),
                "edges": _array(_object({"parent": _integer(minimum=0), "field": _string(), "child": _integer(minimum=0)})),
            }
        )
    elif kind == "patch":
        payload = _object(
            {
                "files": _array(
                    _object(
                        {
                            "path": _string(),
                            "operation": _string(enum=("add", "update", "delete")),
                            "content": _string(),
                        }
                    )
                ),
                "summary": _string(),
            }
        )
    elif kind == "tool_action":
        if tool_name not in TOOL_NAMES:
            raise ValueError(f"unsupported tool schema: {tool_name}")
        arguments = {
            "run_tests": _object({"command": _array(_string()), "timeout_seconds": _integer(minimum=1, maximum=300)}),
            "read_file": _object({"path": _string()}),
            "write_file": _object({"path": _string(), "content": _string()}),
        }[tool_name]
        payload = _object({"tool": _string(enum=(tool_name,)), "arguments": arguments})
    elif kind == "diagnostic":
        payload = _object(
            {
                "diagnosis": _string(),
                "recommended_action": _string(),
                "evidence": _array(_string()),
            }
        )
    else:
        payload = _object(
            {
                "status": _string(enum=("passed", "failed", "unsupported", "timed_out")),
                "tests": _array(_string()),
                "measurements": _object({"runtime_ms": SchemaNode("number"), "peak_bytes": _integer(minimum=0)}),
            }
        )
    return _object(
        {
            "schema_version": _string(enum=("nlp-json-output-v1",)),
            "kind": _string(enum=(kind,)),
            "payload": payload,
        }
    )


def _current_value_schema(state: SchemaGrammarState) -> SchemaNode:
    frame = state.grammar.frames[-1]
    current = state.schemas[-1]
    if frame.kind == "root":
        return current
    if frame.kind == "object":
        if state.pending_key is None:
            raise InvalidNativeJSONDecision("object value has no committed key")
        return current.properties[state.pending_key]
    if current.items is None:
        raise InvalidNativeJSONDecision("array schema has no item contract")
    return current.items


def _check_scalar(action: NativeJSONDecision, schema: SchemaNode) -> Any:
    value = action.value
    kinds = {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "true": value is True,
        "false": value is False,
        "null": value is None,
    }
    if not kinds.get(action.action, False):
        raise InvalidNativeJSONDecision(f"{action.action} decision has an incompatible value")
    expected = "boolean" if action.action in {"true", "false"} else action.action
    if not (expected == schema.kind or (expected == "number" and schema.kind == "integer")):
        raise InvalidNativeJSONDecision(f"schema expects {schema.kind}, got {expected}")
    if schema.enum_values and value not in schema.enum_values:
        raise InvalidNativeJSONDecision(f"value {value!r} is outside the committed schema enum")
    if schema.kind == "integer" and not isinstance(value, int):
        raise InvalidNativeJSONDecision("integer schema requires an integer value")
    if schema.kind in {"integer", "number"}:
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidNativeJSONDecision("non-finite JSON numbers are forbidden")
        if schema.minimum is not None and value < schema.minimum:
            raise InvalidNativeJSONDecision("number is below the schema minimum")
        if schema.maximum is not None and value > schema.maximum:
            raise InvalidNativeJSONDecision("number is above the schema maximum")
    return value


def commit_native_json_decisions(
    kind: str,
    decisions: tuple[NativeJSONDecision, ...] | list[NativeJSONDecision],
    *,
    tool_name: str = "run_tests",
) -> NativeJSONCommit:
    """Validate schema transitions, then serialize exactly the committed value tree."""
    schema = output_schema(kind, tool_name=tool_name)
    state = SchemaGrammarState.initial(schema)
    stack: list[dict[str, Any] | list[Any]] = []
    pending_key: str | None = None
    root: Any = None

    def attach(value: Any) -> None:
        nonlocal root, pending_key
        if not stack:
            if root is not None:
                raise InvalidNativeJSONDecision("root value was committed more than once")
            root = value
            return
        parent = stack[-1]
        if isinstance(parent, dict):
            if pending_key is None:
                raise InvalidNativeJSONDecision("object value is missing its committed key")
            parent[pending_key] = value
            pending_key = None
        else:
            parent.append(value)

    try:
        for decision in decisions:
            before = state
            state = state.transition(decision.action, key=decision.key)
            if decision.action == "key":
                pending_key = decision.key
            elif decision.action == "start_object":
                value: dict[str, Any] = {}
                attach(value)
                stack.append(value)
            elif decision.action == "start_array":
                value_list: list[Any] = []
                attach(value_list)
                stack.append(value_list)
            elif decision.action in {"end_object", "end_array"}:
                if not stack:
                    raise InvalidNativeJSONDecision("container close has no open container")
                stack.pop()
            elif decision.action in {"comma", "colon"}:
                continue
            else:
                value = _check_scalar(decision, _current_value_schema(before))
                attach(value)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InvalidNativeJSONDecision):
            raise
        raise InvalidNativeJSONDecision(str(error)) from error

    if not state.complete or stack or not isinstance(root, dict):
        raise InvalidNativeJSONDecision("native JSON decision sequence is incomplete")
    if root.get("schema_version") != "nlp-json-output-v1" or root.get("kind") != kind:
        raise InvalidNativeJSONDecision("schema version or output kind is not the committed contract")
    structured = TypedTreeGateway.from_json(root, domain="json", schema_valid=True)
    TypedTreeGateway.require_contract(structured)
    text = structured.text
    if json.loads(text) != root:
        raise InvalidNativeJSONDecision("typed-tree serialization changed committed JSON values")
    return NativeJSONCommit(root, text, tuple(decisions), structured.tree, structured)


__all__ = [
    "JSON_OUTPUT_KINDS",
    "TOOL_NAMES",
    "InvalidNativeJSONDecision",
    "NativeJSONCommit",
    "NativeJSONDecision",
    "commit_native_json_decisions",
    "output_schema",
]
