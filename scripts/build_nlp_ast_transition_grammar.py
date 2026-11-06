#!/usr/bin/env python3
"""Extract train-only AST transition grammar and write its audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.ast_transition_grammar import (
    extract_source_transition_grammar,
    grammar_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_ast_corpus_v1.json",
    )
    parser.add_argument(
        "--freeze-report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_corpus_freeze_v1.json",
    )
    parser.add_argument(
        "--schema-config",
        type=Path,
        default=ROOT / "configs/nlp_ast_schema_v1.json",
    )
    parser.add_argument(
        "--grammar-output",
        type=Path,
        default=ROOT / "configs/nlp_ast_transition_grammar_v1.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_transition_grammar_audit_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schema_config = json.loads(args.schema_config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_report.read_text(encoding="utf-8"))
    corpus_hash = freeze["corpus"]["sha256"]
    grammar, report = extract_source_transition_grammar(
        config,
        root=ROOT,
        corpus_sha256=corpus_hash,
        ast_schema_config=schema_config,
    )
    grammar_payload = json.dumps(grammar, indent=2, sort_keys=True) + "\n"
    args.grammar_output.parent.mkdir(parents=True, exist_ok=True)
    args.grammar_output.write_text(grammar_payload, encoding="utf-8")
    report["grammar_path"] = str(args.grammar_output.relative_to(ROOT))
    report["source_config"] = str(args.config.relative_to(ROOT))
    report["ast_schema_config"] = str(args.schema_config.relative_to(ROOT))
    report["grammar_sha256"] = grammar_sha256(grammar)
    report["freeze_report_sha256"] = grammar_sha256(freeze)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
