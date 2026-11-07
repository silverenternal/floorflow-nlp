#!/usr/bin/env python3
"""Audit zero-tuning exposure without promoting incomplete evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer-report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_zero_hp_optimizer_ablation_v1.json",
    )
    parser.add_argument(
        "--loss-report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_zero_hp_loss_balance_v1.json",
    )
    parser.add_argument(
        "--ast-gate",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_upstream_quality_gate_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_zero_tuning_certificate_v1.json",
    )
    args = parser.parse_args()
    optimizer = json.loads(args.optimizer_report.read_text(encoding="utf-8"))
    balance = json.loads(args.loss_report.read_text(encoding="utf-8"))
    ast_gate = json.loads(args.ast_gate.read_text(encoding="utf-8"))
    checks = {
        "learned_loss_balance": (
            balance["status"] == "pass"
            and balance["manual_scalar_weights"] is False
            and balance["validity_objective_present"] is True
        ),
        "schedule_free_two_horizons": (
            optimizer["status"] == "pass"
            and optimizer["horizon_specific_retuning"] is False
        ),
        "upstream_ast_quality_gate": ast_gate.get("status") == "pass",
        "matched_end_to_end_tuned_control": False,
        "frozen_external_evaluation": False,
    }
    report = {
        "version": "nlp-zero-tuning-certificate-v1",
        "status": "blocked",
        "certified": False,
        "checks": checks,
        "missing": [
            name for name, passed in checks.items() if not passed
        ],
        "learned_budgets": {
            "ast_node_count": True,
            "source_length": True,
            "stage_a_nfe": True,
            "stage_b_nfe": True,
            "serial_depth": True,
            "generated_e_birth": True,
        },
        "safety_caps_are_quality_parameters": False,
        "validation_selects_models": True,
        "test_access": "closed",
        "evidence_boundary": (
            "Partial zero-tuning implementation only. Certification is blocked by "
            "the failing AST quality gate and missing matched end-to-end control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
