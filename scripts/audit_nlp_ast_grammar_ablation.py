#!/usr/bin/env python3
"""Run the matched grammar-mask versus unstructured-head action ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_grammar_ablation import evaluate_action_ablation
from floorflow.nlp_flow.ast_grammar_training import (
    build_grammar_examples,
    encode_grammar_examples,
    train_grammar_legality_head,
)
from floorflow.nlp_flow.ast_holdout import load_ast_corpus

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
        "--negatives",
        type=Path,
        default=ROOT / ".data/nlp/ast_train_v1/grammar_negatives.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_grammar_matched_ablation_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    grammars = {
        language: load_ast_grammar(args.grammar, language)
        for language in ("python", "rust", "cpp")
    }
    negatives = [
        json.loads(line)
        for line in args.negatives.read_text(encoding="utf-8").splitlines()
        if line
    ]
    examples = build_grammar_examples(grammars, negatives)
    encoded = encode_grammar_examples(examples)
    device = _device()
    model, training = train_grammar_legality_head(
        examples,
        device=device,
        seed=args.seed,
        epochs=4,
        batch_size=1024,
    )
    rows = load_ast_corpus(args.corpus)
    ablation = evaluate_action_ablation(
        model,
        encoded.category_indices,
        rows,
        device=device,
    )
    report = {
        "version": "nlp-ast-grammar-matched-ablation-v1",
        "device": str(device),
        "training": training,
        **ablation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
