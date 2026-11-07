#!/usr/bin/env python3
"""Audit lossless teacher AST-to-action-to-source replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.platform.data.splits.ast_holdout import load_ast_corpus
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    decode_actions,
    render_tree_structure,
    trace_teacher_actions,
)

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grammar",
        type=Path,
        default=ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_tree_action_teacher_replay_v1.json",
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
    case_rows = []
    for row in rows:
        language = str(row["language"])
        payload = json.loads(str(row["ast"]))
        try:
            actions = trace_teacher_actions(grammars[language], payload)
            structure = decode_actions(grammars[language], actions)
            rendered = render_tree_structure(structure)
            parsed = parse_ast_graph(rendered, language)
            roundtrip = json.loads(serialize_ast_graph(parsed)) == payload
            exact_source = rendered == row["source"]
            error = None
        except (KeyError, TypeError, ValueError, UnicodeError) as failure:
            actions = []
            exact_source = False
            roundtrip = False
            error = str(failure)
        case_rows.append(
            {
                "case_id": row["case_id"],
                "language": language,
                "actions": len(actions),
                "exact_source": exact_source,
                "canonical_ast_roundtrip": roundtrip,
                "error": error,
            }
        )
    report = {
        "version": "nlp-tree-action-teacher-replay-v1",
        "status": (
            "pass"
            if all(
                row["exact_source"] and row["canonical_ast_roundtrip"]
                for row in case_rows
            )
            else "fail"
        ),
        "split": "validation_only",
        "test_read": False,
        "cases": len(case_rows),
        "exact_source_rate": sum(
            row["exact_source"] for row in case_rows
        )
        / max(1, len(case_rows)),
        "canonical_ast_roundtrip_rate": sum(
            row["canonical_ast_roundtrip"] for row in case_rows
        )
        / max(1, len(case_rows)),
        "repair_or_retry": False,
        "case_rows": case_rows,
        "evidence_boundary": (
            "Teacher-forced representation replay only; no target-free AST "
            "synthesis or coding capability claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
