#!/usr/bin/env python3
"""Run the training-only source and declaration-fragment provenance audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_trace_provenance import file_sha256
from floorflow.nlp_flow.training_source_lineage import audit_training_sources

ROOT = Path(__file__).resolve().parents[1]


def load_train(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            # Split metadata is checked before parsing a complete row.
            if re.search(r'"split"\s*:\s*"train"', line) is None:
                continue
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("split metadata mismatch")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_training_source_lineage_v11.json")
    args = parser.parse_args()
    parents = ROOT / ".data/nlp/ast_core_v1/corpus.jsonl"
    fragments = ROOT / ".data/nlp/tree_action_v1/corpus.jsonl"
    config = ROOT / "configs/nlp_tree_action_corpus_v1.json"
    grammar = ROOT / "configs/nlp_ast_transition_grammar_v1.json"
    report = audit_training_sources(
        load_train(parents),
        load_train(fragments),
        root=ROOT,
        source_root=ROOT / ".data/nlp",
        fragment_config=json.loads(config.read_text(encoding="utf-8"))["fragments"],
        grammars={language: load_ast_grammar(grammar, language) for language in ("python", "rust", "cpp")},
    )
    report["version"] = "nlp-training-source-lineage-v11"
    report["input_sha256"] = {
        name: file_sha256(path)
        for name, path in {"parents": parents, "fragments": fragments, "config": config, "grammar": grammar}.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
