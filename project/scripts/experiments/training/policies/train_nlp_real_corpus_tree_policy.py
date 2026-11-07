#!/usr/bin/env python3
"""Train the real-corpus event-conditioned Seq2Tree policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import (
    event_distribution,
    load_action_rows,
)
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_tree_policy import (
    train_real_corpus_tree_policy,
)
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_tree_runtime import save_policy_metadata

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
        default=ROOT / "project/configs/structured_generation/tree_action_corpus/nlp_tree_action_corpus_all_splits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/nlp_real_corpus_tree_policy_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_real_corpus_policy_train_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_action_rows(ROOT / str(config["output"]))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    grammars = {
        language: load_ast_grammar(
            ROOT / str(config["grammar"]),
            language,
        )
        for language in ("python", "rust", "cpp")
    }
    device = _device()
    model, metrics = train_real_corpus_tree_policy(
        train_rows,
        validation_rows,
        grammars,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "version": "nlp-real-corpus-tree-policy-v1",
            "model": model.state_dict(),
            "vocabulary": model.vocabulary,
            "metrics": metrics,
            "seed": args.seed,
        },
        checkpoint,
    )
    metadata_path = args.output_dir / "metadata.json"
    save_policy_metadata(
        metadata_path,
        vocabulary=model.vocabulary,
        metrics=metrics,
    )
    report = {
        "version": "nlp-real-corpus-policy-train-v1",
        "status": "pass",
        "device": str(device),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_event_distribution": event_distribution(train_rows),
        "validation_event_distribution": event_distribution(validation_rows),
        "metrics": metrics,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "metadata": str(metadata_path.relative_to(ROOT)),
        "target_ast_input": False,
        "repair_or_retry": False,
        "evidence_boundary": (
            "Teacher-forced real-corpus policy training only; target-free "
            "parser validity is audited in the next work package."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
