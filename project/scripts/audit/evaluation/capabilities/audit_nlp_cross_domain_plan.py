#!/usr/bin/env python3
"""Audit the natural-language cross-domain plan without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.platform.contracts.protocol.protocol import protocol_from_payload
from newera.nlp_flow.components.research.experiments.studies.todo import validate_todo

REPO_ROOT = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "project/configs/other/general/chat_sft_canonical/nlp_cross_domain.json",
    )
    parser.add_argument(
        "--todo",
        type=Path,
        default=REPO_ROOT / "nlptodo.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    todo = json.loads(args.todo.read_text())
    protocol = protocol_from_payload(config)

    errors = list(validate_todo(todo, REPO_ROOT, check_artifacts=True))
    task_ids = [task.get("id") for task in todo.get("tasks", [])]

    root = Path(config["engineering"]["source_root"])
    if not root.exists():
        errors.append(f"source root does not exist: {root}")
    if protocol.inference_inputs & protocol.forbidden_inputs:
        errors.append("target-free boundary is open")
    report = {
        "version": "nlp-cross-domain-plan-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "config": str(args.config.relative_to(REPO_ROOT)),
        "todo": str(args.todo.relative_to(REPO_ROOT)),
        "workspace": str(root),
        "task_ids": task_ids,
        "current_round": todo.get("current_round", {}).get("id"),
        "hierarchy_levels": [level.level_id for level in protocol.hierarchy],
        "target_free": protocol.target_free,
        "evidence_boundary": "Protocol audit only; no model-quality evidence.",
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
