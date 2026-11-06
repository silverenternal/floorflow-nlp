#!/usr/bin/env python3
"""Audit padded causal-prefix action batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.prefix_policy_data import (
    audit_prefix_batch,
    encode_prefix_rows,
)
from floorflow.nlp_flow.real_corpus_policy_data import (
    load_action_rows,
)
from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_prefix_tree_policy_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_prefix_action_batch_audit_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_action_rows(ROOT / str(config["action_corpus"]))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    grammars = {
        language: load_ast_grammar(
            ROOT / str(config["grammar"]),
            language,
        )
        for language in ("python", "rust", "cpp")
    }
    vocabulary = build_vocabulary(train, grammars)
    reports = {
        "train": audit_prefix_batch(
            encode_prefix_rows(train, grammars, vocabulary)
        ),
        "validation": audit_prefix_batch(
            encode_prefix_rows(validation, grammars, vocabulary)
        ),
    }
    report = {
        "version": "nlp-prefix-action-batch-audit-v1",
        "status": (
            "pass"
            if all(value["status"] == "pass" for value in reports.values())
            else "fail"
        ),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "splits": reports,
        "target_ast_in_prompt": False,
        "repair_or_retry": False,
        "evidence_boundary": "Batch layout only; no model quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
