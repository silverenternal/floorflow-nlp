#!/usr/bin/env python3
"""Audit nlp_flow task dependencies, budgets, status, and evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorflow.nlp_flow.todo import ready_task_ids, validate_todo

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=Path, default=ROOT / "nlptodo.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.todo.read_text())
    errors = validate_todo(payload, ROOT, check_artifacts=True)
    tasks = payload.get("tasks", [])
    status_counts = Counter(task.get("status") for task in tasks)
    report = {
        "version": "nlp-todo-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "todo": str(args.todo.relative_to(ROOT)),
        "todo_version": payload.get("version"),
        "current_round": payload.get("current_round", {}).get("id"),
        "active_tasks": payload.get("current_round", {}).get("active_tasks", []),
        "active_work_packages": payload.get("current_round", {}).get("active_work_packages", []),
        "ready_tasks": list(ready_task_ids(payload)),
        "task_count": len(tasks),
        "work_package_count": len(payload.get("work_packages", [])),
        "milestone_count": len(payload.get("milestones", [])),
        "decision_point_count": len(payload.get("decision_points", [])),
        "status_counts": dict(sorted(status_counts.items())),
        "phases": payload.get("phases", []),
        "evidence_boundary": "Research task graph audit only; no model-quality evidence.",
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
