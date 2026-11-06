"""All-suite strict JSON validity certification."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.json_tree import strict_json_loads, validate_schema


@dataclass(frozen=True)
class JSONOutputRecord:
    case_id: str
    suite: str
    output: str
    repaired: bool = False
    retried: bool = False
    synthetic: bool = False


def load_records(path: str | Path) -> list[JSONOutputRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = strict_json_loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"record {line_number} must be an object")
            records.append(
                JSONOutputRecord(
                    case_id=str(payload["case_id"]),
                    suite=str(payload["suite"]),
                    output=str(payload["output"]),
                    repaired=bool(payload.get("repaired", False)),
                    retried=bool(payload.get("retried", False)),
                    synthetic=bool(payload.get("synthetic", False)),
                )
            )
    return records


def audit_json_outputs(
    records: Iterable[JSONOutputRecord],
    *,
    schema: Mapping[str, Any],
    schema_registry: Mapping[str, Any],
    required_suites: set[str],
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("JSON validity audit needs at least one record")
    suites = {row.suite for row in rows}
    missing_suites = sorted(required_suites - suites)
    parse_valid = 0
    schema_valid = 0
    repair_count = 0
    retry_count = 0
    synthetic_count = 0
    failures = []
    for row in rows:
        repair_count += int(row.repaired)
        retry_count += int(row.retried)
        synthetic_count += int(row.synthetic)
        try:
            value = strict_json_loads(row.output)
            validate_schema(value, schema)
            parse_valid += 1
            kind = str(value.get("kind", ""))
            spec = schema_registry.get("outputs", {}).get(kind)
            if not isinstance(spec, Mapping):
                raise TypeError(f"unknown JSON output kind: {kind}")
            payload = value.get("payload")
            if not isinstance(payload, Mapping):
                raise TypeError("payload must be an object")
            missing = [key for key in spec.get("payload_required", []) if key not in payload]
            if missing:
                raise ValueError(f"{kind} payload is missing {missing}")
            schema_valid += 1
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            failures.append({"case_id": row.case_id, "error": str(error)})
    total = len(rows)
    parse_rate = parse_valid / total
    schema_rate = schema_valid / total
    validity_passed = (
        not missing_suites
        and parse_rate == 1.0
        and schema_rate == 1.0
        and repair_count == 0
        and retry_count == 0
    )
    claim_passed = validity_passed and synthetic_count == 0
    return {
        "version": "nlp-json-validity-all-suites-v1",
        "status": "pass" if validity_passed else "fail",
        "cases": total,
        "suites": sorted(suites),
        "missing_suites": missing_suites,
        "strict_json_parse_validity": parse_rate,
        "schema_compliance": schema_rate,
        "repair_count": repair_count,
        "retry_count": retry_count,
        "synthetic_count": synthetic_count,
        "failures": failures,
        "claim_gate": "allow_C1_C4" if claim_passed else "block_C1_C4",
        "evidence_boundary": "This reports the supplied records only.",
    }


def load_schema_registry(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("schema registry must be an object")
    return payload


__all__ = [
    "JSONOutputRecord",
    "audit_json_outputs",
    "load_records",
    "load_schema_registry",
]
