#!/usr/bin/env python3
"""Record the validation-only AST generation-family decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT
        / "outputs/reports/nlp_ast_generation_matched_comparison_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_generation_decision_v1.json",
    )
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    discrete = comparison["arms"]["discrete_tree_actions"]
    report = {
        "version": "nlp-ast-generation-decision-v1",
        "status": "rejected",
        "decision": "no_family_promoted",
        "validation_only": True,
        "test_read": False,
        "candidate_results": {
            "continuous_cfm": {
                "parser_validity": comparison["arms"]["continuous_cfm"][
                    "parser_validity"
                ],
                "eligible": False,
            },
            "discrete_tree_actions": {
                "parser_validity": discrete["parser_validity"],
                "execution_success": discrete["execution_success"],
                "domain": discrete["domain"],
                "eligible": False,
            },
            "hybrid": {
                "parser_validity": None,
                "eligible": False,
            },
        },
        "failed_construct_families": [
            "matched_continuous_cfm",
            "matched_hybrid",
            "open_domain_python",
            "rust",
            "cpp",
        ],
        "retained_result": (
            "Keep grammar-constrained tree actions as an exploratory contract "
            "path. Do not claim they beat continuous CFM without a matched run."
        ),
        "next_action": (
            "Build a matched continuous/discrete/hybrid harness before reopening "
            "generation-family promotion."
        ),
        "evidence_boundary": comparison["evidence_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
