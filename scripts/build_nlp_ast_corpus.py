#!/usr/bin/env python3
"""Build the bounded parser-valid AST/source corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.ast_corpus import build_ast_corpus, write_ast_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_ast_corpus_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_corpus_audit_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, summary = build_ast_corpus(config, root=ROOT)
    report = write_ast_corpus(
        rows,
        output=ROOT / str(config["output"]),
        manifest=ROOT / str(config["manifest"]),
        summary=summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if (
        not report["source_integrity"]
        or not report["split_integrity"]
        or int(report["rows"]) == 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
