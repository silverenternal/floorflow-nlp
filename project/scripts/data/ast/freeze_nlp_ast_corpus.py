#!/usr/bin/env python3
"""Freeze AST corpus, parser, tokenizer, prompt, and holdout hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.splits.ast_freeze import build_ast_freeze_manifest

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/structured_generation/ast_corpus_diagnostic/ast_diagnostic_freeze/nlp_ast_freeze.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_corpus_freeze_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_ast_freeze_manifest(ROOT, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
