#!/usr/bin/env python3
"""Consolidate mixed-corpus, tokenizer, and contamination evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1/corpus_v1/manifest.json",
    )
    parser.add_argument(
        "--tokenizer-report",
        type=Path,
        default=ROOT / ".data/nlp/tokenizer_v1/report.json",
    )
    parser.add_argument(
        "--contamination",
        type=Path,
        default=ROOT / "outputs/reports/nlp_benchmark_contamination_base.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    tokenizer = json.loads(args.tokenizer_report.read_text(encoding="utf-8"))
    contamination = json.loads(args.contamination.read_text(encoding="utf-8"))
    train_tokens = tokenizer["ledger"]["train"]["tokens_by_family"]
    total_train = sum(train_tokens.values())
    shares = {
        family: count / max(1, total_train)
        for family, count in sorted(train_tokens.items())
    }
    target = {
        "general_language": 0.25,
        "code": 0.35,
        "model_architecture": 0.20,
        "engineering_acceleration": 0.20,
    }
    blockers = []
    if not tokenizer["p5m_target_met"]:
        blockers.append(
            f"P5M target short by {92_000_000 - tokenizer['total_tokens']} tokens"
        )
    for family, share in target.items():
        if shares.get(family, 0.0) < share - 0.02:
            blockers.append(
                f"{family} share {shares.get(family, 0.0):.4f} is below target {share:.2f}"
            )
    if contamination.get("status") != "pass":
        blockers.append("benchmark contamination audit failed")
    if not tokenizer.get("roundtrip_pass", False):
        blockers.append("tokenizer roundtrip failed")
    report = {
        "version": "nlp-corpus-collection-v1",
        "status": "pass" if not blockers else "partial",
        "blockers": blockers,
        "splits": corpus["splits"],
        "tokenizer": {
            "path": tokenizer["tokenizer"],
            "sha256": tokenizer["sha256"],
            "vocab_size": tokenizer["vocab_size"],
            "fit_split": tokenizer["fit_split"],
            "total_tokens": tokenizer["total_tokens"],
            "p5m_target_met": tokenizer["p5m_target_met"],
            "roundtrip_pass": tokenizer["roundtrip_pass"],
        },
        "train_family_shares": shares,
        "train_family_targets": target,
        "contamination": {
            "status": contamination["status"],
            "training_rows": contamination["training_rows"],
            "benchmark_scans": contamination["benchmark_scans"],
        },
        "evidence_boundary": "Data preparation only; benchmark/model quality untested.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
