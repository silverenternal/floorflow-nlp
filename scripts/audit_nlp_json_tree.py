#!/usr/bin/env python3
"""Audit strict JSON tree/schema behavior without repair paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.json_tree import (
    load_schema,
    parse_and_validate,
    serialize_json_tree,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "schema/nlp_json_output_v1.schema.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_json_schema_attack_v1.json",
    )
    args = parser.parse_args()
    schema = load_schema(args.schema)
    golden = json.dumps(
        {
            "schema_version": "nlp-json-output-v1",
            "kind": "tool_action",
            "payload": {"tool": "pytest", "arguments": ["-q"]},
        }
    )
    node = parse_and_validate(golden, schema)
    serialized = serialize_json_tree(node)
    attacks = {
        "duplicate_key": '{"a":1,"a":2}',
        "nan": '{"a":NaN}',
        "infinity": '{"a":Infinity}',
        "trailing_data": '{"a":1} trailing',
        "wrong_root_type": "[]",
        "missing_required": '{"schema_version":"nlp-json-output-v1"}',
        "extra_root_field": (
            '{"schema_version":"nlp-json-output-v1","kind":"ast",'
            '"payload":{},"extra":1}'
        ),
    }
    results = {}
    for name, payload in attacks.items():
        try:
            parse_and_validate(payload, schema)
        except (ValueError, TypeError, json.JSONDecodeError):
            results[name] = "rejected"
        else:
            results[name] = "accepted"
    errors = [name for name, result in results.items() if result != "rejected"]
    report = {
        "version": "nlp-json-schema-attack-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "golden_serialized": serialized,
        "attacks": results,
        "evidence_boundary": "Strict JSON codec contract only; no model generation result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
