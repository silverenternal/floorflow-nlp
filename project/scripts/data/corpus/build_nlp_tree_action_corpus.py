#!/usr/bin/env python3
"""Build real-corpus prompt/teacher-action supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.platform.data.splits.ast_holdout import load_ast_corpus
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_action_data import (
    build_real_corpus_action_dataset,
    write_real_corpus_action_dataset,
)

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/structured_generation/tree_action_corpus/nlp_tree_action_corpus_all_splits.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_tree_action_corpus_audit_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_ast_corpus(ROOT / str(config["ast_corpus"]))
    grammars = {
        language: load_ast_grammar(
            ROOT / str(config["grammar"]),
            language,
        )
        for language in ("python", "rust", "cpp")
    }
    action_rows, report = build_real_corpus_action_dataset(
        rows,
        grammars,
        config["prompt"],
        config["fragments"],
    )
    output_path = ROOT / str(config["output"])
    corpus_sha256 = write_real_corpus_action_dataset(
        action_rows,
        output=output_path,
    )
    report["output"] = str(output_path.relative_to(ROOT))
    report["corpus_sha256"] = corpus_sha256
    report["prompt_contract"] = config["prompt_policy"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
