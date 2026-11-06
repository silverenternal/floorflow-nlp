from __future__ import annotations

import json
from pathlib import Path

from floorflow.nlp_flow.json_tree import load_schema
from floorflow.nlp_flow.json_validity import (
    JSONOutputRecord,
    audit_json_outputs,
    load_schema_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def output(kind: str = "tool_action") -> str:
    return json.dumps(
        {
            "schema_version": "nlp-json-output-v1",
            "kind": kind,
            "payload": {
                "tool": "pytest",
                "arguments": ["-q"],
                "language": "python",
                "nodes": [],
                "edges": [],
                "files": [],
                "summary": "",
                "diagnosis": "ok",
                "recommended_action": "none",
                "evidence": [],
                "status": "ok",
                "tests": [],
                "measurements": {},
            },
        }
    )


def audit(records: list[JSONOutputRecord]) -> dict:
    return audit_json_outputs(
        records,
        schema=load_schema(ROOT / "schema/nlp_json_output_v1.schema.json"),
        schema_registry=load_schema_registry(ROOT / "configs/nlp_json_schemas_v1.json"),
        required_suites={"train", "validation", "test", "engineering"},
    )


def test_all_suite_json_validity_passes_with_zero_repairs() -> None:
    records = [
        JSONOutputRecord(suite, suite, output())
        for suite in ("train", "validation", "test", "engineering")
    ]

    report = audit(records)

    assert report["status"] == "pass"
    assert report["strict_json_parse_validity"] == 1.0
    assert report["claim_gate"] == "allow_C1_C4"


def test_json_validity_blocks_claim_on_failure_repair_or_missing_suite() -> None:
    records = [
        JSONOutputRecord("train", "train", output()),
        JSONOutputRecord("validation", "validation", "{", repaired=True),
    ]

    report = audit(records)

    assert report["status"] == "fail"
    assert report["repair_count"] == 1
    assert {"test", "engineering"}.issubset(report["missing_suites"])
    assert report["claim_gate"] == "block_C1_C4"


def test_synthetic_all_suite_validity_cannot_unlock_claim() -> None:
    records = [
        JSONOutputRecord(suite, suite, output(), synthetic=True)
        for suite in ("train", "validation", "test", "engineering")
    ]

    report = audit(records)

    assert report["status"] == "pass"
    assert report["synthetic_count"] == 4
    assert report["claim_gate"] == "block_C1_C4"
