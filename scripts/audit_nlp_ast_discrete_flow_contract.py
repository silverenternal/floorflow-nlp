#!/usr/bin/env python3
"""Audit AST-aware discrete corruption dimensions and transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorflow.nlp_flow.ast_discrete_flow import (
    TOKEN_DIMENSIONS,
    contract_fingerprint,
    corruption_schedule,
    extract_discrete_tokens,
)
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_holdout import load_ast_corpus
from floorflow.nlp_flow.tree_action_decoder import trace_teacher_actions

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grammar",
        type=Path,
        default=ROOT / "configs/nlp_ast_transition_grammar_v1.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_discrete_flow_contract_v1.json",
    )
    args = parser.parse_args()
    grammars = {
        language: load_ast_grammar(args.grammar, language)
        for language in ("python", "rust", "cpp")
    }
    rows = [
        row
        for row in load_ast_corpus(args.corpus)
        if row["split"] == "validation"
    ]
    dimension_counts: Counter[str] = Counter()
    masked_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    cases = 0
    for row in rows:
        language = str(row["language"])
        payload = json.loads(str(row["ast"]))
        actions = trace_teacher_actions(grammars[language], payload)
        tokens = extract_discrete_tokens(actions)
        corrupt = corruption_schedule(
            tokens,
            mask_fraction=0.25,
            seed=20261102 + cases,
        )
        cases += 1
        for token in tokens:
            dimension_counts[token.dimension] += 1
            if token.dimension == "action_type":
                action_counts[token.value] += 1
        for token in corrupt:
            if token.masked:
                masked_counts[token.dimension] += 1
    covered = all(dimension_counts[value] > 0 for value in TOKEN_DIMENSIONS)
    report = {
        "version": "nlp-ast-discrete-flow-contract-v1",
        "status": "pass" if covered else "fail",
        "split": "validation_only",
        "test_read": False,
        "cases": cases,
        "token_dimensions": list(TOKEN_DIMENSIONS),
        "dimension_counts": dict(dimension_counts),
        "masked_counts_at_25_percent": dict(masked_counts),
        "action_counts": dict(action_counts),
        "corruption_target": "typed_ast_actions",
        "undifferentiated_source_tokens": False,
        "grammar_state_required": True,
        "mask_fraction_role": "diagnostic_contract_only",
        "repair_or_retry": False,
        "evidence_boundary": (
            "Corruption contract only; no denoising quality, parser validity, "
            "or executable generation result."
        ),
    }
    report["contract_sha256"] = contract_fingerprint(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
