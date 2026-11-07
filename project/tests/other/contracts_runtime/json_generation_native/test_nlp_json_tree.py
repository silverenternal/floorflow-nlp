from __future__ import annotations

import json
from pathlib import Path

import pytest

from newera.nlp_flow.components.generation.outputs.json.json_tree import (
    build_json_tree,
    load_schema,
    parse_and_validate,
    serialize_json_tree,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.skip(reason="Requires the retired root schema/nlp_json_output_v1.schema.json artifact.")
def test_json_tree_round_trip_and_schema() -> None:
    schema = load_schema(ROOT / "schema/nlp_json_output_v1.schema.json")
    text = json.dumps(
        {
            "schema_version": "nlp-json-output-v1",
            "kind": "diagnostic",
            "payload": {"diagnosis": "ok", "recommended_action": "none", "evidence": []},
            "constraints": {"timeout": 30},
        }
    )

    node = parse_and_validate(text, schema)

    assert strict_json_loads(serialize_json_tree(node))["kind"] == "diagnostic"


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":1} trailing',
    ],
)
def test_strict_parser_rejects_ambiguous_or_nonfinite_json(payload: str) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        strict_json_loads(payload)


def test_tree_builder_rejects_nonfinite_number() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        build_json_tree({"value": float("nan")})
