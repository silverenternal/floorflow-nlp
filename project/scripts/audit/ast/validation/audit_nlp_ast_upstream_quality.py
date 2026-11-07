#!/usr/bin/env python3
"""Enforce strict token-endpoint AST quality before Stage B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast
from newera.nlp_flow.components.ast.diagnostics.validation.ast_quality_gate import evaluate_ast_upstream_source

ROOT = Path(__file__).resolve().parents[5]
SAMPLES = (
    (
        "python",
        "def add(a, b):\n    return a + b\n",
    ),
    (
        "rust",
        "fn add(a: i32, b: i32) -> i32 { a + b }\n",
    ),
    (
        "cpp",
        "int add(int a, int b) { return a + b; }\n",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nfe-report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_cfm_nfe_curve_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_upstream_quality_gate_v1.json",
    )
    args = parser.parse_args()
    nfe = json.loads(args.nfe_report.read_text(encoding="utf-8"))
    primary = str(nfe["primary_nfe"])
    samples = nfe["values"][primary]["samples"]
    rows = []
    for sample, (language, reference_source) in zip(
        samples,
        SAMPLES,
        strict=True,
    ):
        report = evaluate_ast_upstream_source(
            str(sample.get("token_path_source", "")),
            parse_ast(reference_source, language),
            language,
            source_kind="token_endpoint",
        )
        report.update(
            {
                "language": language,
                "grammar_shell_parseable": sample[
                    "grammar_shell_parseable"
                ],
                "grammar_shell_excluded_from_gate": True,
            }
        )
        rows.append(report)
    passed = sum(row["status"] == "pass" for row in rows)
    report = {
        "version": "nlp-ast-upstream-quality-gate-v1",
        "status": "pass" if passed == len(rows) else "blocked",
        "primary_nfe": int(primary),
        "strict_token_endpoint_pass_rate": passed / len(rows),
        "grammar_shell_parse_rate": nfe["values"][primary][
            "grammar_shell_parse_rate"
        ],
        "grammar_shell_gate_policy": (
            "Grammar-template selection cannot satisfy upstream AST synthesis."
        ),
        "samples": rows,
        "claim_level": "none",
        "evidence_boundary": (
            "Three synthetic samples only; pass is necessary but not sufficient "
            "for corpus-scale AST quality."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
