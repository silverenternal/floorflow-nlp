#!/usr/bin/env python3
"""Compare existing AST generation families without promoting unmatched arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continuous",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_cfm_nfe_curve_v1.json",
    )
    parser.add_argument(
        "--discrete",
        type=Path,
        default=ROOT / "outputs/reports/nlp_tree_action_generation_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/nlp_ast_generation_matched_comparison_v1.json",
    )
    args = parser.parse_args()
    continuous = json.loads(args.continuous.read_text(encoding="utf-8"))
    discrete = json.loads(args.discrete.read_text(encoding="utf-8"))
    continuous_primary = continuous["values"][str(continuous["primary_nfe"])]
    report = {
        "version": "nlp-ast-generation-matched-comparison-v1",
        "status": "fail",
        "comparison_status": "rejected_unmatched_continuous_arm",
        "arms": {
            "continuous_cfm": {
                "source": str(args.continuous.relative_to(ROOT)),
                "nfe": continuous["primary_nfe"],
                "parser_validity": continuous_primary["token_path_parse_rate"],
                "grammar_shell_parse_rate": continuous_primary[
                    "grammar_shell_parse_rate"
                ],
                "domain": "three_sample_stock_ast_cfm_probe",
                "eligible_for_promotion": False,
            },
            "discrete_tree_actions": {
                "source": str(args.discrete.relative_to(ROOT)),
                "nfe": discrete["metrics"].get(
                    "mean_action_steps",
                    None,
                ),
                "parser_validity": discrete["metrics"]["parser_validity"],
                "execution_success": discrete["metrics"]["execution_success"],
                "domain": discrete["domain"],
                "eligible_for_promotion": False,
            },
            "hybrid": {
                "source": None,
                "parser_validity": None,
                "eligible_for_promotion": False,
                "reason": "No separately trained continuous+discrete arm at matched budgets.",
            },
        },
        "matching_audit": {
            "same_training_data": False,
            "same_parameter_count": False,
            "same_total_nfe": False,
            "same_parser_fingerprint": True,
            "same_safety_cap": True,
        },
        "selected_family": None,
        "selection_boundary": (
            "No family is promoted. The discrete action result is a bounded "
            "synthetic contract result, while the continuous stock CFM arm has "
            "different training data, parameters, and NFE."
        ),
        "evidence_boundary": (
            "Negative comparison result. Parser-valid discrete output does not "
            "establish matched superiority over AST CFM."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
