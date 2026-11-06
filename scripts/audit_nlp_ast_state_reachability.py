#!/usr/bin/env python3
"""Audit historical policy masks on frozen validation teacher prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_reachability import inspect_teacher_mask
from floorflow.nlp_flow.ast_state_trace import snapshot, trace_action
from floorflow.nlp_flow.real_corpus_policy_data import action_from_record
from floorflow.nlp_flow.real_corpus_tree_runtime import terminalizable_kinds, viable_parent_kinds
from floorflow.nlp_flow.tree_action_decoder import TreeActionDecoder

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_case(row: dict, grammar: object, vocabulary: dict, parent: set[str], terminal: set[str]) -> dict:
    decoder = TreeActionDecoder(grammar)
    records = row["actions"]
    budget = sum(record["type"] == "open_node" for record in records)
    failures = []
    replay_error = None
    for step, record in enumerate(records):
        action = action_from_record(record)
        verdict = inspect_teacher_mask(decoder, action, budget, parent, terminal, vocabulary, prompt=str(row["prompt"]))
        if not verdict["reachable"]:
            failures.append({"step": step, "action": record, "mask": verdict, "state": asdict(snapshot(decoder))})
        transition = trace_action(decoder, action)
        if not transition["accepted"]:
            replay_error = {"step": step, "action": record, "error": transition["error"], "state": transition["before"]}
            break
    return {
        "case_id": row["case_id"],
        "language": row["language"],
        "gold_node_budget": budget,
        "steps": len(records),
        "replayed_steps": len(decoder.actions),
        "terminated": decoder.terminated,
        "mask_exclusions": failures,
        "replay_error": replay_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument("--grammar", type=Path, default=ROOT / "configs/nlp_ast_transition_grammar_v1.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs/train/nlp_prefix_tree_policy_v1/checkpoint.pt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_ast_state_reachability_v11.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    grammars = {language: load_ast_grammar(args.grammar, language) for language in ("python", "rust", "cpp")}
    parent = {language: viable_parent_kinds(grammar) for language, grammar in grammars.items()}
    terminal = {language: terminalizable_kinds(grammar) for language, grammar in grammars.items()}
    cases = []
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            # Filter on split metadata before decoding a row; test examples are not evaluated.
            if '"split": "validation"' not in line:
                continue
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("split metadata mismatch")
            language = row["language"]
            cases.append(audit_case(row, grammars[language], vocabulary, parent[language], terminal[language]))
    if len(cases) != 60 or len({row["case_id"] for row in cases}) != 60:
        raise ValueError("expected 60 unique frozen validation cases")
    counts = Counter(
        reason for case in cases for failure in case["mask_exclusions"] for reason in failure["mask"]["reasons"]
    )
    report = {
        "version": "nlp-ast-state-reachability-v11",
        "status": "pass"
        if not counts and all(case["replay_error"] is None and case["terminated"] for case in cases)
        else "fail",
        "split": "validation",
        "test_evaluated": False,
        "case_count": len(cases),
        "inputs_sha256": {
            "corpus": sha256(args.corpus),
            "grammar": sha256(args.grammar),
            "checkpoint": sha256(args.checkpoint),
        },
        "budget_source": "teacher target node count, diagnostic oracle only",
        "reasons": dict(counts),
        "cases": cases,
        "evidence_boundary": (
            "Historical masks on teacher prefixes with existential lexical reachability and oracle budgets; "
            "model lexical choice, predicted budgets and free-running quality are not measured."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
