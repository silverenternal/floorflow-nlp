#!/usr/bin/env python3
"""Audit the architecture-search probe against its frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.architecture_search import estimate_inference_flops
from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_architecture_search_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_architecture_search_probe_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_architecture_search_audit_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    selected_config_path = _resolve(config["freeze"]["selected_config"])
    selected_config = json.loads(
        selected_config_path.read_text(encoding="utf-8")
    )
    selected = report["selected_candidate"]
    model_config = NLPModelConfig(**selected["config"])
    model = CompactNLPDecoder(model_config)
    actual_parameters = count_unique_trainable_parameters(model)
    expected_flops = estimate_inference_flops(
        model_config,
        batch_size=int(config["measurement"]["batch_size"]),
        sequence_length=int(config["measurement"]["sequence_length"]),
        expected_depth_fraction=float(
            config["measurement"]["expected_depth_fraction"]
        ),
    )
    expected_budgets = tuple(config["halving"]["token_budgets"])
    round_budgets = tuple(round_["budget_tokens"] for round_ in report["rounds"])
    evaluations = [
        result
        for round_ in report["rounds"]
        for result in round_["results"]
    ]
    checks = {
        "report_status_complete": report["status"] == "probe_complete",
        "test_not_read": report["test_read"] is False,
        "all_splits_are_validation": all(
            item["evaluation_split"] in {"validation", "heldout_validation"}
            for item in evaluations
        ),
        "round_budgets_match_protocol": round_budgets == expected_budgets,
        "all_round_budgets_matched": all(
            item["training_tokens"] >= item["budget_tokens"]
            for item in evaluations
        ),
        "selected_in_pareto_front": (
            selected["candidate_id"] in report["pareto_front"]
        ),
        "selected_parameters_recount": (
            selected["parameters"] == actual_parameters
        ),
        "selected_inference_flops_recount": (
            selected["estimated_inference_flops"] == expected_flops
        ),
        "selected_within_parameter_cap": (
            actual_parameters <= int(config["measurement"]["max_parameters"])
        ),
        "selected_within_inference_budget": (
            expected_flops <= int(config["measurement"]["max_inference_flops"])
        ),
        "frozen_config_matches_selection": (
            selected_config["candidate_id"] == selected["candidate_id"]
            and selected_config["model"] == selected["config"]
        ),
        "promotion_not_claimed_without_verifiers": (
            report["promotion_eligible"] is False
            and bool(report["promotion_blockers"])
        ),
    }
    output = {
        "version": "nlp-architecture-search-audit-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selected_candidate_id": selected["candidate_id"],
        "actual_parameters": actual_parameters,
        "estimated_inference_flops": expected_flops,
        "promotion_blockers": report["promotion_blockers"],
        "evidence_boundary": (
            "Protocol and accounting audit only; no coding capability, "
            "SOTA, or smallest-model claim."
        ),
    }
    rendered = json.dumps(output, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if output["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
