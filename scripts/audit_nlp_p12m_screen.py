#!/usr/bin/env python3
"""Audit matched two-seed P12M control/candidate stack runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_p12m_screen_v1.json",
    )
    parser.add_argument(
        "--decision-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_scale_decision_v1.json",
    )
    args = parser.parse_args()
    roots = {
        "seed19_control": ROOT / "outputs/train/nlp_p12m_control_cpu_smoke_v1/report.json",
        "seed19_candidate": ROOT / "outputs/train/nlp_p12m_candidate_cpu_smoke_v1/report.json",
        "seed29_control": ROOT / "outputs/train/nlp_p12m_control_cpu_smoke_seed29_v1/report.json",
        "seed29_candidate": ROOT / "outputs/train/nlp_p12m_candidate_cpu_smoke_seed29_v1/report.json",
    }
    reports = {name: _read(path) for name, path in roots.items()}
    checks = {}
    for seed in ("seed19", "seed29"):
        control = reports[f"{seed}_control"]
        candidate = reports[f"{seed}_candidate"]
        checks[f"{seed}_status"] = control["status"] == candidate["status"] == "pass"
        checks[f"{seed}_tokenizer"] = control["tokenizer_sha256"] == candidate["tokenizer_sha256"]
        checks[f"{seed}_data"] = control["data"] == candidate["data"]
        checks[f"{seed}_steps"] = control["steps"] == candidate["steps"]
        checks[f"{seed}_samples"] = control["samples_seen"] == candidate["samples_seen"]
        checks[f"{seed}_restore"] = (
            control["checkpoint_restore_exact"]
            and candidate["checkpoint_restore_exact"]
        )
        checks[f"{seed}_budget"] = (
            control["budget"]["passed"]
            and candidate["budget"]["passed"]
        )
    checks["two_distinct_seeds"] = (
        reports["seed19_control"]["spec"]["seed"]
        != reports["seed29_control"]["spec"]["seed"]
    )
    checks["same_runtime_device"] = len(
        {report["runtime"]["device"] for report in reports.values()}
    ) == 1
    checks["test_not_read"] = all(
        not report["data"]["test_read_during_smoke"]
        for report in reports.values()
    )
    checks["target_not_used"] = all(
        not report["data"]["target_used_at_inference"]
        for report in reports.values()
    )
    report = {
        "version": "nlp-p12m-screen-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "screen_type": "bounded_cpu_stack_screen",
        "capability_measured": False,
        "checks": checks,
        "seeds": {
            seed: {
                "control_parameters": reports[f"{seed}_control"]["parameters"],
                "candidate_parameters": reports[f"{seed}_candidate"]["parameters"],
                "control_validation_loss": reports[f"{seed}_control"]["validation_losses"][0],
                "candidate_validation_loss": reports[f"{seed}_candidate"]["validation_losses"][0],
            }
            for seed in ("seed19", "seed29")
        },
        "working_reports": {name: str(path.relative_to(ROOT)) for name, path in roots.items()},
        "evidence_boundary": (
            "Matched data/tokenizer/budget/restore contract only; one-step CPU losses "
            "are not capability evidence."
        ),
    }
    decision = {
        "version": "nlp-scale-decision-v1",
        "status": "blocked",
        "selected_tier": None,
        "p5m_contract_screen": _read(
            ROOT / "outputs/reports/nlp_p5m_screen_v1.json"
        )["status"],
        "p12m_contract_screen": report["status"],
        "reason": "Two-seed capability runs and paired intervals are not available.",
        "claim_decision": "do_not_claim_scale_survival",
        "evidence_boundary": "Negative/unresolved evidence retained.",
    }
    for path, payload in ((args.output, report), (args.decision_output, decision)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(payload, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
