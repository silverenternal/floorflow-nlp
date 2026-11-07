#!/usr/bin/env python3
"""Bind the implementation decision to scoped AST diagnostic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "outputs/reports"


def _load(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise TypeError("evidence report must be an object")
    return payload, hashlib.sha256(data).hexdigest()


def build_decision(paths: dict[str, Path]) -> dict[str, Any]:
    """Build a design decision, rejecting inconsistent or out-of-scope evidence."""
    required = {"reachability", "failures", "oracle_fixture", "comparison"}
    if set(paths) != required:
        raise ValueError("all four named evidence reports are required")
    loaded = {name: _load(path) for name, path in paths.items()}
    reach, failures, oracle, comparison = (
        loaded[name][0] for name in ("reachability", "failures", "oracle_fixture", "comparison")
    )
    cases = reach.get("cases")
    if reach.get("split") != "validation" or reach.get("test_evaluated") is not False or not isinstance(cases, list):
        raise ValueError("reachability report is not validation-only")
    case_ids = {case["case_id"] for case in cases}
    if len(cases) != 60 or len(case_ids) != 60 or reach.get("case_count") != 60:
        raise ValueError("reachability report must cover 60 unique validation cases")
    counts = Counter(
        reason for case in cases for failure in case["mask_exclusions"] for reason in failure["mask"]["reasons"]
    )
    if dict(counts) != reach.get("reasons"):
        raise ValueError("reachability aggregate does not match case evidence")
    if any(case.get("replay_error") is not None or case.get("terminated") is not True for case in cases):
        raise ValueError("teacher action replay is incomplete")
    arms = failures.get("arms")
    if failures.get("split") != "validation" or failures.get("test_read") is not False or not isinstance(arms, dict):
        raise ValueError("failure inventory is not validation-only")
    if set(arms) != {"event_mlp", "prefix_transformer"}:
        raise ValueError("failure inventory must contain both historical arms")
    for name, arm in arms.items():
        rows = arm.get("cases", [])
        categories = Counter(row["observed_category"] for row in rows)
        if len(rows) != 60 or {row["case_id"] for row in rows} != case_ids:
            raise ValueError(f"{name} case coverage differs")
        if dict(categories) != arm.get("observed_categories"):
            raise ValueError(f"{name} aggregate differs from case evidence")
    if oracle.get("status") != "pass" or oracle.get("split") != "synthetic_fixture_only":
        raise ValueError("oracle controls lack scoped synthetic fixture evidence")
    if comparison.get("status") != "rejected" or comparison.get("selected_policy") is not None:
        raise ValueError("historical negative comparison must remain unpromoted")
    case_counts = Counter(
        reason
        for case in cases
        for reason in {reason for failure in case["mask_exclusions"] for reason in failure["mask"]["reasons"]}
    )
    decisions = [
        {
            "mechanism": "scope_and_parent_candidate_coverage",
            "evidence": {
                key: counts[key] for key in ("parent_not_in_active_scope", "parent_candidate_window_truncated")
            },
            "implementation": "Preserve eligible open-scope parents and explicitly detect candidate-window truncation.",
            "control": "Historical active-scope-only mask under matched budgets.",
            "limitation": "Teacher-prefix exclusions do not establish the root cause of free-running failures.",
        },
        {
            "mechanism": "typed_lexical_and_gap_projection",
            "evidence": {key: counts[key] for key in ("lexical_value_unreachable", "gap_value_unreachable")},
            "implementation": "Separate lexical/trivia decisions and add provenance-checked copy and open vocabulary.",
            "control": "Historical copy/closed-vocabulary/gap policy at matched exposure and compute.",
            "limitation": "Teacher-prefix reachability does not measure the model's chosen tokens or quality.",
        },
        {
            "mechanism": "global_completion_and_hierarchical_blueprint",
            "evidence": {
                "historical_no_candidate": {
                    name: arm["observed_categories"].get("no_candidate", 0) for name, arm in arms.items()
                }
            },
            "implementation": "Implement bounded completion and planning with shared train/inference state.",
            "control": "Historical local mask and flat-prefix policy with identical data and resource limits.",
            "limitation": "The missing free-running traces leave the contribution of global dead ends unresolved.",
        },
        {
            "mechanism": "offline_oracle_decomposition",
            "evidence": {"fixture_status": oracle["status"], "scope": oracle["split"]},
            "implementation": "Isolate topology and lexical controls with explicit typed leaf alignment and coverage.",
            "control": "Teacher/teacher and candidate/candidate arms alongside both cross-projections.",
            "limitation": "Only synthetic wiring is verified; model upper bounds and unalignable cases are unmeasured.",
        },
    ]
    return {
        "version": "nlp-ast-recovery-design-v11",
        "status": "implementation_decision_only",
        "evidence_sha256": {name: value[1] for name, value in loaded.items()},
        "teacher_prefix": {"cases": len(cases), "occurrences": dict(counts), "affected_cases": dict(case_counts)},
        "historical_outcomes": {name: arm["observed_categories"] for name, arm in arms.items()},
        "decisions": decisions,
        "deferred": [
            "Capture and independently bind free-running traces before assigning a primary cause.",
            "Measure matched ablations and oracle controls after the implementation freeze.",
            "Do not promote a model, alter the frozen validation gate, or train on validation failures.",
        ],
        "evidence_boundary": "CPU design decision only; no target-free capability or causal effectiveness claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORTS / "nlp_ast_recovery_design_v11.json")
    args = parser.parse_args()
    result = build_decision(
        {
            "reachability": REPORTS / "nlp_ast_state_reachability_v11.json",
            "failures": REPORTS / "nlp_ast_failure_inventory_v11.json",
            "oracle_fixture": REPORTS / "nlp_ast_oracle_fixture_v11.json",
            "comparison": REPORTS / "nlp_prefix_tree_comparison_v1.json",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "teacher_prefix", "historical_outcomes")}, indent=2))


if __name__ == "__main__":
    main()
