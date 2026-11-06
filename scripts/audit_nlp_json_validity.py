#!/usr/bin/env python3
"""Audit strict JSON parser and grammar attack suites."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from floorflow.nlp_flow.model.json_grammar import ACTIONS, JSONActionHead, JSONGrammarState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_json_validity_attack_v1.json",
    )
    args = parser.parse_args()
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_nlp_json_tree.py",
            "tests/test_nlp_json_grammar.py",
            "-q",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    logits = JSONActionHead(16)(torch.randn(1, 16), [JSONGrammarState.initial()])
    invalid_actions = [
        action
        for action in ("comma", "colon", "end_object", "end_array")
        if not torch.isneginf(logits[0, ACTIONS.index(action)])
    ]
    errors = []
    if tests.returncode != 0:
        errors.append("strict JSON attack tests failed")
    if invalid_actions:
        errors.append(f"invalid actions have nonzero probability: {invalid_actions}")
    report = {
        "version": "nlp-json-validity-attack-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "test_returncode": tests.returncode,
        "test_output": tests.stdout[-1000:],
        "invalid_actions": invalid_actions,
        "repair_path": False,
        "retry_path": False,
        "evidence_boundary": "Parser/grammar attack contract only; not a trained-model all-suite validity result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
