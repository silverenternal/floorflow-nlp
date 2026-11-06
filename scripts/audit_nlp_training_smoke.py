#!/usr/bin/env python3
"""Combine control/candidate P2M CPU smoke reports into one gate report."""

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
        "--control",
        type=Path,
        default=ROOT / "outputs/train/nlp_p2m_control_cpu_smoke_v1/report.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "outputs/train/nlp_p2m_candidate_cpu_smoke_v1/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_p2m_smoke_v1.json",
    )
    parser.add_argument(
        "--executor",
        type=Path,
        default=ROOT / "outputs/reports/nlp_p2m_executor_integration_v1.json",
    )
    args = parser.parse_args()
    control = _read(args.control)
    candidate = _read(args.candidate)
    executor = _read(args.executor)
    checks = {
        "both_status_pass": control["status"] == candidate["status"] == "pass",
        "both_checkpoint_restore_exact": (
            control["checkpoint_restore_exact"]
            and candidate["checkpoint_restore_exact"]
        ),
        "both_budget_pass": (
            control["budget"]["passed"]
            and candidate["budget"]["passed"]
        ),
        "same_seed": control["spec"]["seed"] == candidate["spec"]["seed"],
        "same_tokenizer": control["tokenizer_sha256"] == candidate["tokenizer_sha256"],
        "same_data": control["data"] == candidate["data"],
        "test_not_read": not control["data"]["test_read_during_smoke"],
        "target_not_used": not control["data"]["target_used_at_inference"],
        "control_runtime_reached": bool(control["gradient_reached_prefixes"]),
        "candidate_runtime_reached": bool(candidate["gradient_reached_prefixes"]),
        "coding_and_engineering_executor_passed": executor["status"] == "pass",
    }
    report = {
        "version": "nlp-p2m-smoke-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "control": {
            "report": str(args.control.relative_to(ROOT)),
            "parameters": control["parameters"],
            "budget": control["budget"],
            "train_losses": control["train_losses"],
            "gradient_reached_prefixes": control["gradient_reached_prefixes"],
        },
        "candidate": {
            "report": str(args.candidate.relative_to(ROOT)),
            "parameters": candidate["parameters"],
            "budget": candidate["budget"],
            "train_losses": candidate["train_losses"],
            "gradient_reached_prefixes": candidate["gradient_reached_prefixes"],
        },
        "executor": {
            "report": str(args.executor.relative_to(ROOT)),
            "coding_patch_executor_passed": executor["coding_patch_executor_passed"],
            "engineering_linux_diagnostic_passed": executor[
                "engineering_linux_diagnostic_passed"
            ],
        },
        "evidence_boundary": (
            "Two-step CPU training-stack smoke only; losses are not capability metrics "
            "and the optional CFM/JSON/factory branches are not fully trained."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
