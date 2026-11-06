#!/usr/bin/env python3
"""Inventory existing validation failures without rerunning a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from floorflow.nlp_flow.ast_failure_taxonomy import classify_historical_case

ROOT = Path(__file__).resolve().parents[1]


def inventory(paths: dict[str, Path]) -> dict[str, object]:
    """Validate and summarize both historical arms by immutable case identity."""
    arms = {}
    case_sets = []
    for arm, path in paths.items():
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes)
        if payload.get("split") not in {"validation", "validation_only"} or payload.get("test_read") is not False:
            raise ValueError(f"{arm} is not a validation-only report")
        rows = [classify_historical_case(case) for case in payload["cases_report"]]
        case_ids = {row["case_id"] for row in rows}
        if len(rows) != 60 or len(case_ids) != 60:
            raise ValueError(f"{arm} does not contain 60 unique validation cases")
        case_sets.append(case_ids)
        arms[arm] = {
            "source": str(path),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "cases": rows,
            "observed_categories": dict(Counter(row["observed_category"] for row in rows)),
            "trace_missing": sum(row["trace_status"] == "missing" for row in rows),
        }
    if len(case_sets) != 2 or case_sets[0] != case_sets[1]:
        raise ValueError("historical arms do not share the same validation cases")
    return {
        "version": "nlp-ast-failure-inventory-v11",
        "status": "partial_missing_free_running_traces",
        "split": "validation",
        "case_count_per_arm": 60,
        "test_read": False,
        "arms": arms,
        "required_followup": "Capture action, mask, lexical and parser traces in deferred checkpoint replay.",
        "evidence_boundary": "Observed outcomes only; first-failure steps and causal diagnoses remain unresolved.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_ast_failure_inventory_v11.json")
    args = parser.parse_args()
    result = inventory(
        {
            "prefix_transformer": ROOT / "outputs/reports/nlp_prefix_tree_ast_gate_v1.json",
            "event_mlp": ROOT / "outputs/reports/nlp_real_corpus_ast_gate_v1.json",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: value["observed_categories"] for arm, value in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
