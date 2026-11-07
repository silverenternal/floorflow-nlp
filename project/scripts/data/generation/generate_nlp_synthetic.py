#!/usr/bin/env python3
"""Materialize deterministic target-free synthetic mechanism tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.synthetic.synthetic import generate_tasks, write_tasks

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".data/nlp/synthetic_v1/tasks.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_synthetic_generator_v1.json",
    )
    args = parser.parse_args()
    tasks = generate_tasks(args.seed, args.count)
    digest = write_tasks(tasks, args.output)
    report = {
        "version": "nlp-synthetic-generator-v1",
        "status": "pass",
        "seed": args.seed,
        "count": len(tasks),
        "kinds": sorted({task.kind for task in tasks}),
        "task_sha256": digest,
        "path": str(args.output.relative_to(ROOT)),
        "oracle_targets_excluded_from_model_input": True,
        "evidence_boundary": "Mechanism probe data only; not a public capability benchmark.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
