#!/usr/bin/env python3
"""Audit the neuron-as-submodel evidence and growth experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.research.architecture.growth.growth import validate_growth_plan

ROOT = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/learning/growth/nlp_neuron_growth.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text())
    errors = validate_growth_plan(payload)
    report = {
        "version": "nlp-growth-plan-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "config": str(args.config.relative_to(ROOT)),
        "route": payload["research_route"],
        "sources": len(payload["evidence_ledger"]),
        "peer_reviewed_sources": sum(
            str(row["source_type"]).startswith("peer_reviewed")
            for row in payload["evidence_ledger"]
        ),
        "arms": [row["id"] for row in payload["ablation_arms"]],
        "expert_factory_method": payload["expert_factory"]["minimal_generation_method"],
        "fixed_pool_role": payload["moe_boundary"]["fixed_expert_pool"],
        "meta_episode_steps": len(payload["meta_growth_learning"]["episode"]),
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
