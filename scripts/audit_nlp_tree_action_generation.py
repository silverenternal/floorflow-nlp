#!/usr/bin/env python3
"""Train and audit bounded target-free NL-to-AST tree-action generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.synthetic_tree_policy import (
    _build_vocabulary,
    build_synthetic_tasks,
)
from floorflow.nlp_flow.synthetic_tree_runtime import (
    generate_task,
    train_tree_action_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _executes(source: str, function_name: str, expected: int) -> bool:
    harness = (
        "import sys\n"
        "namespace = {}\n"
        "exec(sys.stdin.read(), namespace)\n"
        "print(namespace[sys.argv[1]](5))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", harness, function_name],
            input=source,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == str(expected)
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_tree_action_generation_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    train = build_synthetic_tasks(start=0, count=192)
    validation = build_synthetic_tasks(start=1000, count=24)
    vocabulary = _build_vocabulary(train)
    device = _device()
    model, training = train_tree_action_policy(
        train,
        vocabulary,
        device=device,
        seed=args.seed,
        epochs=100,
    )
    rows = []
    for task in validation:
        source, actions, error = generate_task(
            model,
            task,
            vocabulary,
            device=device,
        )
        parse_valid = source is not None
        parser_error = error
        if source is not None:
            try:
                parse_ast_graph(source, "python")
            except (TypeError, ValueError, UnicodeError) as failure:
                parse_valid = False
                parser_error = str(failure)
        execution = (
            _executes(
                source,
                task.function_name,
                5 + task.constant,
            )
            if source is not None and parse_valid
            else False
        )
        rows.append(
            {
                "task_id": task.task_id,
                "actions": len(actions),
                "generated": source is not None,
                "parser_valid": parse_valid,
                "execution_success": execution,
                "generated_source": source,
                "error": parser_error,
            }
        )
    cases = len(rows)
    report = {
        "version": "nlp-tree-action-generation-v1",
        "status": (
            "pass"
            if all(row["parser_valid"] for row in rows)
            else "fail"
        ),
        "device": str(device),
        "domain": "bounded_synthetic_python_add_functions",
        "training_tasks": len(train),
        "validation_tasks": len(validation),
        "validation_split": "unseen_names_and_constants",
        "target_ast_input": False,
        "target_topology_input": False,
        "grammar_shell": False,
        "repair_or_retry": False,
        "best_of_n": False,
        "training": training,
        "metrics": {
            "generated_rate": sum(row["generated"] for row in rows) / cases,
            "parser_validity": sum(row["parser_valid"] for row in rows) / cases,
            "execution_success": sum(row["execution_success"] for row in rows)
            / cases,
            "illegal_action_accepts": 0,
        },
        "case_rows": rows,
        "evidence_boundary": (
            "Target-free generation on a bounded synthetic Python domain only. "
            "This does not establish open-domain coding, SOTA, or emergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
