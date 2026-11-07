#!/usr/bin/env python3
"""Audit the compact coding SOTA architecture and staged-training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.research.architecture.search.architecture import validate_architecture_plan

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/architecture/definitions/nlp_architecture.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_architecture_plan(payload)
    report = {
        "version": "nlp-architecture-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "ambition": payload["ambition"],
        "channels": [row["id"] for row in payload["typed_channels"]],
        "attention": sorted(payload["attention"]),
        "training_scopes": [row["id"] for row in payload["training_scopes"]],
        "arms": [row["id"] for row in payload["experimental_arms"]],
        "evidence_boundary": payload["evidence_boundary"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
