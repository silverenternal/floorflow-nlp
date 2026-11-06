#!/usr/bin/env python3
"""Train and audit typed leaf selection with context copy pointers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_holdout import load_ast_corpus
from floorflow.nlp_flow.leaf_pointer import (
    build_leaf_examples,
    train_leaf_pointer_head,
)

ROOT = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


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
        default=ROOT / "outputs/reports/nlp_tree_leaf_pointer_audit_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    grammars = {
        language: load_ast_grammar(args.grammar, language)
        for language in ("python", "rust", "cpp")
    }
    rows = load_ast_corpus(args.corpus)
    train, validation = build_leaf_examples(rows)
    device = _device()
    _model, metrics = train_leaf_pointer_head(
        train,
        validation,
        grammars,
        device=device,
        seed=args.seed,
    )
    report = {
        "version": "nlp-tree-leaf-pointer-audit-v1",
        "status": (
            "pass"
            if metrics["class_grammar_validity"] == 1.0
            and not metrics["parser_repair"]
            and not metrics["retry_or_best_of_n"]
            else "fail"
        ),
        "device": str(device),
        "split": "train_fit_validation_audit",
        "test_read": False,
        "metrics": metrics,
        "evidence_boundary": (
            "Leaf-selection contract only; copy success does not establish "
            "open-domain AST or executable source generation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
