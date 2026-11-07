#!/usr/bin/env python3
"""Audit the mixed NLP corpus and token-to-parameter budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.corpus.corpus import corpus_tiers, validate_corpus_plan

ROOT = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/data/corpus/nlp_corpus_plan.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text())
    errors = validate_corpus_plan(payload)
    report = {
        "version": "nlp-corpus-plan-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "config": str(args.config.relative_to(ROOT)),
        "ratio": payload["ratio_contract"]["tokens_per_parameter"],
        "tiers": [
            {
                "id": tier.tier_id,
                "total_tokens": tier.total_tokens,
                "target_parameters": tier.target_parameters,
                "train_tokens": tier.train_tokens,
                "validation_tokens": tier.validation_tokens,
                "test_tokens": tier.test_tokens,
            }
            for tier in (corpus_tiers(payload) if not errors else ())
        ],
        "evidence_boundary": "Corpus and budget contract only; no data-quality or model-quality claim.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
