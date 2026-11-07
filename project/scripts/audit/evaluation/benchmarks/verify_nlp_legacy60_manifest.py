#!/usr/bin/env python3
"""Verify the frozen 60-case manifest from validation rows only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "project/scripts/data/benchmarks"))

from freeze_nlp_legacy60_manifest import freeze_manifest, load_validation_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument(
        "--grammar",
        type=Path,
        default=ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_legacy60_eval_manifest.json",
    )
    args = parser.parse_args()
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    observed = freeze_manifest(load_validation_rows(args.corpus), grammar_path=args.grammar)
    if observed != frozen:
        expected_cases = {case["case_id"]: case for case in frozen.get("cases", [])}
        observed_cases = {case["case_id"]: case for case in observed["cases"]}
        changed_cases = sorted(
            case_id
            for case_id in expected_cases.keys() | observed_cases.keys()
            if expected_cases.get(case_id) != observed_cases.get(case_id)
        )
        changed_top_level = sorted(
            key for key in set(frozen) | set(observed) if frozen.get(key) != observed.get(key) and key != "cases"
        )
        print(
            json.dumps(
                {
                    "status": "fail",
                    "changed_cases": changed_cases,
                    "changed_manifest_fields": changed_top_level,
                },
                indent=2,
            )
        )
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "pass",
                "case_count": observed["case_count"],
                "language_counts": observed["language_counts"],
                "test_content_evaluated": observed["test_content_evaluated"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
