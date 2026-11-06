from __future__ import annotations

from copy import deepcopy

from floorflow.nlp_flow.strict_ast_audit import audit_strict_ast_closure


def _inputs():
    case = {
        "case_id": "case-1",
        "gold_source_sha256": "a" * 64,
        "gold_graph_sha256": "b" * 64,
        "parser_grammar_sha256": "c" * 64,
    }
    manifest = {
        "ordered_case_ids": ["case-1"],
        "thresholds": {"minimum_f1": 0.95},
        "split": "validation",
        "cases": [case],
    }
    report = {
        "ordered_case_ids": ["case-1"],
        "thresholds": {"minimum_f1": 0.95},
        "split": "validation",
        "oracle_inputs": [],
        "seed_ids": [11, 17],
        "evidence": [
            {"case_id": "case-1", "seed": seed, **{key: value for key, value in case.items() if key != "case_id"}}
            for seed in (11, 17)
        ],
    }
    return manifest, report


def test_independent_audit_accepts_exact_case_threshold_seed_and_evidence_identity() -> None:
    manifest, report = _inputs()
    result = audit_strict_ast_closure(manifest, report, expected_seeds=[11, 17])
    assert result["status"] == "passed"
    assert result["independent"]
    assert result["quality_claim"] == "none"


def test_independent_audit_rejects_changed_cases_thresholds_oracle_and_dropped_seeds() -> None:
    manifest, report = _inputs()
    changed = deepcopy(report)
    changed["ordered_case_ids"] = ["other-case"]
    changed["thresholds"]["minimum_f1"] = 0.8
    changed["oracle_inputs"] = ["gold AST"]
    changed["seed_ids"] = [11]
    result = audit_strict_ast_closure(manifest, changed, expected_seeds=[11, 17])
    assert result["status"] == "rejected"
    assert {
        "case_ids_changed_or_reordered",
        "thresholds_changed",
        "oracle_inputs_present",
        "seed_set_changed_or_dropped",
    } <= set(result["errors"])


def test_independent_audit_rejects_missing_seed_rows_and_evidence_hash_mismatch() -> None:
    manifest, report = _inputs()
    report["evidence"].pop()
    report["evidence"][0]["gold_graph_sha256"] = "f" * 64
    result = audit_strict_ast_closure(manifest, report, expected_seeds=[11, 17])
    assert result["status"] == "rejected"
    assert "evidence_case_seed_set_mismatch" in result["errors"]
    assert "evidence_identity_mismatch:case-1" in result["errors"]


def test_independent_audit_rejects_test_content_and_evaluator_identity_drift() -> None:
    manifest, report = _inputs()
    manifest["evaluator_sha256"] = "d" * 64
    report["test_content_evaluated"] = True
    report["evidence"][0]["evaluator_sha256"] = "e" * 64
    report["evidence"][1]["evaluator_sha256"] = "d" * 64
    result = audit_strict_ast_closure(manifest, report, expected_seeds=[11, 17])
    assert result["status"] == "rejected"
    assert "non_validation_or_test_content_used" in result["errors"]
    assert "evidence_identity_mismatch:case-1" in result["errors"]
