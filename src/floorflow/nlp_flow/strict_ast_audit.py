"""Independent immutable-input audit for strict AST recovery evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def audit_strict_ast_closure(
    frozen_manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    expected_seeds: list[int],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_cases = frozen_manifest.get("ordered_case_ids", [])
    reported_cases = report.get("ordered_case_ids", [])
    if reported_cases != expected_cases:
        errors.append("case_ids_changed_or_reordered")
    if report.get("thresholds") != frozen_manifest.get("thresholds"):
        errors.append("thresholds_changed")
    if report.get("oracle_inputs") not in ([], None):
        errors.append("oracle_inputs_present")
    if report.get("split") != "validation" or report.get("test_content_evaluated") is True:
        errors.append("non_validation_or_test_content_used")
    if report.get("seed_ids") != expected_seeds:
        errors.append("seed_set_changed_or_dropped")
    cases = {row.get("case_id"): row for row in frozen_manifest.get("cases", [])}
    evidence_rows = report.get("evidence", [])
    by_identity = {
        (row.get("case_id"), row.get("seed")): row
        for row in evidence_rows
        if isinstance(row, dict)
    }
    expected_pairs = {(case_id, seed) for case_id in expected_cases for seed in expected_seeds}
    if len(by_identity) != len(evidence_rows) or set(by_identity) != expected_pairs:
        errors.append("evidence_case_seed_set_mismatch")
    for case_id, seed in sorted(expected_pairs):
        frozen = cases.get(case_id)
        observed = by_identity.get((case_id, seed))
        if frozen is None or observed is None:
            continue
        expected_identity = {
            "case_id": case_id,
            "seed": seed,
            "gold_source_sha256": frozen.get("gold_source_sha256"),
            "gold_graph_sha256": frozen.get("gold_graph_sha256"),
            "parser_grammar_sha256": frozen.get("parser_grammar_sha256"),
        }
        evaluator_identity = frozen_manifest.get("evaluator_sha256")
        implementations = frozen_manifest.get("implementation_sha256")
        if evaluator_identity is None and implementations is not None:
            encoded = json.dumps(implementations, sort_keys=True, separators=(",", ":"))
            evaluator_identity = hashlib.sha256(encoded.encode()).hexdigest()
        if evaluator_identity is not None:
            expected_identity["evaluator_sha256"] = evaluator_identity
        if any(observed.get(key) != value for key, value in expected_identity.items()):
            errors.append(f"evidence_identity_mismatch:{case_id}")
    status = "passed" if not errors else "rejected"
    return {
        "status": status,
        "independent": True,
        "errors": errors,
        "expected_case_count": len(expected_cases),
        "observed_evidence_count": len(evidence_rows),
        "expected_seeds": list(expected_seeds),
        "observed_seeds": report.get("seed_ids", []),
        "quality_claim": "none",
    }


__all__ = ["audit_strict_ast_closure"]
