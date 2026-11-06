#!/usr/bin/env python3
"""Audit the nlp_flow coding/engineering target and claim ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.target import target_tiers, validate_target_plan

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_coding_target_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text())
    errors = validate_target_plan(payload)
    report = {
        "version": "nlp-coding-target-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "config": str(args.config.relative_to(ROOT)),
        "claim": payload["claim"]["display"],
        "promotion_ladder": [row["id"] for row in payload["promotion_ladder"]],
        "parameter_ladder": [
            {
                "id": tier.tier_id,
                "parameters": tier.parameters,
                "corpus_tokens": tier.corpus_tokens,
                "ratio": tier.ratio,
            }
            for tier in (target_tiers(payload) if not errors else ())
        ],
        "evidence_boundary": payload["evidence_boundary"],
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
