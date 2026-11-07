"""Verifier-grounded eligibility and explicit no-winner Pareto selection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchDecision:
    candidate_id: str
    eligible: bool
    reasons: tuple[str, ...]
    objectives: tuple[float, float, float, float]
    costs: tuple[float, float, float]


def assess_candidate(row: dict[str, Any], *, minimum_cases: int = 30) -> SearchDecision:
    reasons: list[str] = []
    if not row.get("candidate_id"):
        reasons.append("missing_candidate_id")
    if not row.get("checkpoint_id"):
        reasons.append("missing_checkpoint_id")
    if row.get("split") != "validation":
        reasons.append("non_validation_split")
    if row.get("provenance") != "external_verified_validation":
        reasons.append("unqualified_or_fixture_provenance")
    if row.get("verifier_sha256") is None:
        reasons.append("missing_verifier")
    if row.get("evidence_sha256") is None or row.get("identity_sha256") is None:
        reasons.append("missing_evidence_or_identity_digest")
    support = int(row.get("support_count", 0))
    if support <= 0 or support < minimum_cases:
        reasons.append("zero_or_under_supported_coverage")
    total = int(row.get("case_count", 0))
    if total < support or total <= 0 or support / max(total, 1) < 0.8:
        reasons.append("insufficient_case_coverage")
    family = row.get("family")
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        reasons.append("metrics_must_be_track_specific_object")
    if family == "reconstruction_recovery":
        required = ("reconstruction_f1", "ast_validity", "native_validity", "coverage")
    elif family == "natural_task_executable_success":
        required = ("executable_success", "ast_validity", "native_validity", "coverage")
        if not row.get("execution_verified"):
            reasons.append("missing_execution_verification")
    else:
        required = ()
        reasons.append("unknown_track_family")
    if any(not isinstance(metrics.get(name), (int, float)) for name in required):
        reasons.append("missing_track_metrics")
    if row.get("fixture_execution") or row.get("oracle_only") or row.get("test_split_label"):
        reasons.append("forbidden_evidence_origin")
    def number(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    objectives = tuple(
        number(metrics.get(name))
        for name in (
            "reconstruction_f1" if family == "reconstruction_recovery" else "executable_success",
            "ast_validity",
            "native_validity",
            "coverage",
        )
    )
    costs = tuple(number(row.get(name)) for name in ("token_exposure", "resident_parameters", "total_compute"))
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in objectives):
        reasons.append("invalid_quality_metric_range")
    if objectives[1] < 1.0:
        reasons.append("ast_validity_gate_failed")
    if objectives[2] < 1.0:
        reasons.append("native_validity_gate_failed")
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        reasons.append("missing_or_invalid_cost_measurement")
    return SearchDecision(str(row.get("candidate_id", "")), not reasons, tuple(reasons), objectives, costs)


def _dominates(left: SearchDecision, right: SearchDecision) -> bool:
    quality_better_or_equal = all(a >= b for a, b in zip(left.objectives, right.objectives, strict=True))
    cost_better_or_equal = all(a <= b for a, b in zip(left.costs, right.costs, strict=True))
    strict = left.objectives != right.objectives or left.costs != right.costs
    return quality_better_or_equal and cost_better_or_equal and strict


def select_pareto_candidates(rows: Iterable[dict[str, Any]], *, minimum_cases: int = 30) -> dict[str, Any]:
    candidate_rows = list(rows)
    identifiers = [row.get("candidate_id") for row in candidate_rows]
    if len(identifiers) != len(set(identifiers)):
        return {
            "status": "no_winner_duplicate_candidate_ids",
            "winner": None,
            "frontier": [],
            "family_results": {},
            "decisions": [],
        }
    decisions = [assess_candidate(row, minimum_cases=minimum_cases) for row in candidate_rows]
    eligible = [decision for decision in decisions if decision.eligible]
    if eligible:
        by_family: dict[str, list[SearchDecision]] = {}
        for row in candidate_rows:
            if row.get("candidate_id") in {item.candidate_id for item in eligible}:
                by_family.setdefault(str(row["family"]), []).append(
                    next(item for item in eligible if item.candidate_id == row["candidate_id"])
                )
        family_results = {}
        for family, group in by_family.items():
            candidate_ids = {item.candidate_id for item in group}
            family_rows = [row for row in candidate_rows if row.get("candidate_id") in candidate_ids]
            if len({row.get("checkpoint_id") for row in family_rows}) < 5:
                family_results[family] = {
                    "status": "no_winner_insufficient_checkpoints",
                    "winner": None,
                    "frontier": [],
                }
                continue
            if len({item.objectives for item in group}) < 2:
                family_results[family] = {"status": "no_winner_constant_labels", "winner": None, "frontier": []}
                continue
            frontier = [item for item in group if not any(_dominates(other, item) for other in group if other != item)]
            winner = frontier[0].candidate_id if len(frontier) == 1 else None
            family_results[family] = {
                "status": "winner_selected" if winner else "no_winner_pareto_tie",
                "winner": winner,
                "frontier": [item.candidate_id for item in frontier],
            }
        winners = [result["winner"] for result in family_results.values() if result["winner"]]
        status = next(iter(family_results.values()))["status"] if len(family_results) == 1 else "family_results"
        return {
            "status": status,
            "winner": winners[0] if len(family_results) == 1 and winners else None,
            "frontier": next(iter(family_results.values()))["frontier"] if len(family_results) == 1 else [],
            "family_results": family_results,
            "decisions": decisions,
        }
    return {"status": "no_winner_ineligible", "winner": None, "frontier": [], "decisions": decisions}


__all__ = ["SearchDecision", "assess_candidate", "select_pareto_candidates"]
