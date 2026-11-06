#!/usr/bin/env python3
"""Build deterministic AST grammar and semantic negatives from train rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.ast_holdout import load_ast_corpus
from floorflow.nlp_flow.ast_negative_data import (
    build_ast_negative_suite,
    suite_sha256,
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
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_negative_generation_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    negatives = config["negatives"]
    rows = load_ast_corpus(args.corpus)
    cases, report = build_ast_negative_suite(
        rows,
        grammar_cases_per_language=int(
            negatives["grammar_cases_per_language"]
        ),
    )
    suite_path = ROOT / str(negatives["output"])
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_payload = "".join(
        json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n"
        for case in cases
    )
    suite_path.write_text(suite_payload, encoding="utf-8")
    report["suite_path"] = str(suite_path.relative_to(ROOT))
    report["suite_sha256"] = suite_sha256(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
