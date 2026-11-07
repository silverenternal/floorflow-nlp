from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.research.evaluation.benchmarks.benchmark_metrics import (
    CaseResult,
    attack_report,
    benchmark_metrics,
    validate_metric_contract,
)

ROOT = Path(__file__).resolve().parents[4]


def test_deterministic_metric_attacks_are_sensitive() -> None:
    report = attack_report()

    assert report["status"] == "pass"
    assert all(report["checks"].values())


def test_metric_contract_has_hard_json_and_no_best_of_n() -> None:
    payload = json.loads((ROOT / "project/configs/evaluation/metrics/nlp_metric_contract.json").read_text())

    assert validate_metric_contract(payload) == ()
    payload["primary_sampling"]["best_of_n_allowed"] = True
    payload["hard_guards"]["strict_json_parse_validity"] = 0.99

    errors = validate_metric_contract(payload)

    assert "best-of-N and retries are forbidden" in errors
    assert "JSON validity hard guard must equal one" in errors


def test_metrics_are_perfect_for_identity_case() -> None:
    row = CaseResult(
        "identity",
        True,
        2,
        2,
        frozenset({"a", "b"}),
        frozenset({"a", "b"}),
        frozenset({"a>b"}),
        frozenset({"a>b"}),
        True,
        True,
    )

    metrics = benchmark_metrics([row])

    assert metrics["pass_at_1"] == 1.0
    assert metrics["ast_node_f1"] == 1.0
    assert metrics["strict_json_parse_validity"] == 1.0
