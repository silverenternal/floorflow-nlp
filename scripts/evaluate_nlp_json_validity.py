#!/usr/bin/env python3
"""Evaluate strict JSON validity over supplied train/validation/test/engineering rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.json_tree import load_schema
from floorflow.nlp_flow.json_validity import (
    audit_json_outputs,
    load_records,
    load_schema_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "schema/nlp_json_output_v1.schema.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/nlp_json_schemas_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_json_outputs(
        load_records(args.records),
        schema=load_schema(args.schema),
        schema_registry=load_schema_registry(args.registry),
        required_suites={"train", "validation", "test", "engineering"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
