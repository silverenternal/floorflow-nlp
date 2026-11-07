#!/usr/bin/env python3
"""Train the causal-prefix transformer Seq2Tree policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import train_prefix_tree_policy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import (
    event_distribution,
    load_action_rows,
)
from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[5]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/structured_generation/transition_grammar_ast/lexical_matched_policy/nlp_prefix_tree_policy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/nlp_prefix_tree_policy_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_prefix_tree_policy_train_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261102)
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
    train_batch = encode_prefix_rows(train, grammars, vocabulary)
    validation_batch = encode_prefix_rows(validation, grammars, vocabulary)
    device = _device()
    model, metrics = train_prefix_tree_policy(
        train_batch,
        validation_batch,
        vocabulary,
        device=device,
        seed=args.seed,
        epochs=int(config["epochs"]),
        batch_rows=int(config["batch_rows"]),
        hidden_size=int(config["hidden_size"]),
        layers=int(config["decoder_layers"]),
        heads=int(config["decoder_heads"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "version": "nlp-prefix-tree-policy-v1",
            "model": model.state_dict(),
            "vocabulary": vocabulary,
            "metrics": metrics,
            "config": config,
        },
        checkpoint,
    )
    report = {
        "version": "nlp-prefix-tree-policy-train-v1",
        "status": "pass",
        "device": str(device),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_events": event_distribution(train),
        "validation_events": event_distribution(validation),
        "metrics": metrics,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "target_ast_input": False,
        "repair_or_retry": False,
        "evidence_boundary": "Teacher-forced transformer training only; strict generation gate follows.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
