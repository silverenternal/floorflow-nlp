"""Deterministic benchmark metrics and attack tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    compiled: bool | None
    tests_passed: int
    tests_total: int
    ast_predicted_nodes: frozenset[str]
    ast_reference_nodes: frozenset[str]
    ast_predicted_edges: frozenset[str]
    ast_reference_edges: frozenset[str]
    json_valid: bool
    schema_valid: bool
    repaired: bool = False
    retried: bool = False

    @property
    def passed(self) -> bool:
        return self.tests_total > 0 and self.tests_passed == self.tests_total


def _mean(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _f1(predicted: frozenset[str], reference: frozenset[str]) -> float:
    if not predicted and not reference:
        return 1.0
    true_positive = len(predicted & reference)
    precision = true_positive / max(1, len(predicted))
    recall = true_positive / max(1, len(reference))
    return 2.0 * precision * recall / max(1.0e-12, precision + recall)


def benchmark_metrics(rows: list[CaseResult]) -> dict[str, float]:
    if not rows:
        raise ValueError("at least one case is required")
    compiled = [row.compiled for row in rows if row.compiled is not None]
    total_tests = sum(row.tests_total for row in rows)
    passed_tests = sum(row.tests_passed for row in rows)
    return {
        "pass_at_1": _mean(row.passed for row in rows),
        "compilation_rate": _mean(compiled) if compiled else 1.0,
        "unit_test_pass_rate": passed_tests / max(1, total_tests),
        "repository_all_tests_pass_rate": _mean(row.passed for row in rows),
        "ast_node_f1": sum(
            _f1(row.ast_predicted_nodes, row.ast_reference_nodes) for row in rows
        ) / len(rows),
        "ast_edge_f1": sum(
            _f1(row.ast_predicted_edges, row.ast_reference_edges) for row in rows
        ) / len(rows),
        "strict_json_parse_validity": _mean(row.json_valid for row in rows),
        "json_schema_compliance": _mean(row.schema_valid for row in rows),
        "repair_count": float(sum(row.repaired for row in rows)),
        "retry_count": float(sum(row.retried for row in rows)),
    }


def weighted_track_score(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if abs(sum(weights.values()) - 1.0) > 1.0e-9:
        raise ValueError("track weights must sum to 1")
    missing = set(weights) - set(metrics)
    if missing:
        raise ValueError(f"metrics are missing tracks: {sorted(missing)}")
    return sum(float(metrics[track]) * weight for track, weight in weights.items())


def validate_metric_contract(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    sampling = payload.get("primary_sampling")
    if not isinstance(sampling, Mapping):
        errors.append("primary_sampling must be an object")
    else:
        if sampling.get("temperature") != 0.0:
            errors.append("primary temperature must be zero")
        if sampling.get("samples_per_task") != 1:
            errors.append("primary evaluation must use one sample")
        if sampling.get("best_of_n_allowed") is not False or sampling.get("retry_allowed") is not False:
            errors.append("best-of-N and retries are forbidden")
    tracks = payload.get("tracks")
    if not isinstance(tracks, Mapping) or abs(sum(tracks.values()) - 1.0) > 1.0e-9:
        errors.append("track weights must sum to one")
    guards = payload.get("hard_guards")
    if not isinstance(guards, Mapping):
        errors.append("hard_guards must be an object")
    else:
        if guards.get("strict_json_parse_validity") != 1.0:
            errors.append("JSON validity hard guard must equal one")
        if guards.get("json_schema_compliance") != 1.0:
            errors.append("JSON schema hard guard must equal one")
        if guards.get("posthoc_repair_count") != 0 or guards.get("retry_count") != 0:
            errors.append("repair and retry counts must be zero")
    return tuple(errors)


def attack_report() -> dict[str, object]:
    correct = {
        "nodes": frozenset({"root", "function", "return"}),
        "edges": frozenset({"root>function", "function>return"}),
    }
    good = CaseResult(
        "good",
        True,
        3,
        3,
        correct["nodes"],
        correct["nodes"],
        correct["edges"],
        correct["edges"],
        True,
        True,
    )
    broken = CaseResult(
        "broken",
        False,
        0,
        3,
        frozenset({"root"}),
        correct["nodes"],
        frozenset(),
        correct["edges"],
        False,
        False,
        repaired=False,
        retried=False,
    )
    baseline = benchmark_metrics([good])
    attacked = benchmark_metrics([good, broken])
    checks = {
        "pass_at_1_drops": attacked["pass_at_1"] < baseline["pass_at_1"],
        "compilation_drops": attacked["compilation_rate"] < baseline["compilation_rate"],
        "unit_tests_drop": attacked["unit_test_pass_rate"] < baseline["unit_test_pass_rate"],
        "ast_node_f1_drops": attacked["ast_node_f1"] < baseline["ast_node_f1"],
        "ast_edge_f1_drops": attacked["ast_edge_f1"] < baseline["ast_edge_f1"],
        "json_validity_drops": (
            attacked["strict_json_parse_validity"]
            < baseline["strict_json_parse_validity"]
        ),
        "repair_and_retry_zero": (
            attacked["repair_count"] == 0.0 and attacked["retry_count"] == 0.0
        ),
    }
    return {
        "version": "nlp-metric-attack-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "baseline_metrics": baseline,
        "attacked_metrics": attacked,
        "evidence_boundary": "Metric sensitivity attack only; no evaluated model.",
    }


__all__ = [
    "CaseResult",
    "attack_report",
    "benchmark_metrics",
    "validate_metric_contract",
    "weighted_track_score",
]
