#!/usr/bin/env python3
"""Run the strict real-corpus Stage-A AST quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_quality_gate import evaluate_ast_upstream_source
from floorflow.nlp_flow.real_corpus_policy_data import (
    action_from_record,
    load_action_rows,
)
from floorflow.nlp_flow.real_corpus_tree_policy import RealCorpusTreePolicy
from floorflow.nlp_flow.real_corpus_tree_runtime import generate_real_task
from floorflow.nlp_flow.tree_action_decoder import (
    decode_actions,
    render_tree_structure,
)

ROOT = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/nlp_real_corpus_tree_policy_v1/checkpoint.pt",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_real_corpus_ast_gate_v1.json",
    )
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    vocabulary = checkpoint["vocabulary"]
    device = _device()
    model = RealCorpusTreePolicy(vocabulary).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    grammars = {
        language: load_ast_grammar(
            ROOT / "configs/nlp_ast_transition_grammar_v1.json",
            language,
        )
        for language in ("python", "rust", "cpp")
    }
    rows = [
        row
        for row in load_action_rows(args.corpus)
        if row["split"] == "validation"
    ]
    cases = []
    for row in rows:
        language = str(row["language"])
        grammar = grammars[language]
        target_actions = [
            action_from_record(record) for record in row["actions"]
        ]
        target_source = render_tree_structure(
            decode_actions(grammar, target_actions)
        )
        reference = parse_ast(target_source, language)
        generated_source, actions, generation_error = generate_real_task(
            model,
            str(row["prompt"]),
            language,
            grammar,
            vocabulary,
            device=device,
            max_steps=1024,
        )
        if generated_source is None:
            gate = {
                "status": "fail",
                "parse_validity": False,
                "source_kind": "token_endpoint",
                "checks": {
                    "parse_validity": False,
                    "node_type_f1": False,
                    "typed_edge_f1": False,
                    "scope_relation_f1": False,
                    "lexical_f1": False,
                    "tree_edit_distance": False,
                },
                "metrics": None,
                "failure": generation_error,
            }
        else:
            gate = evaluate_ast_upstream_source(
                generated_source,
                reference,
                language,
                source_kind="token_endpoint",
            )
        cases.append(
            {
                "case_id": row["case_id"],
                "language": language,
                "actions": len(actions),
                "generated_source": generated_source,
                "generation_error": generation_error,
                **gate,
            }
        )
    passed = sum(case["status"] == "pass" for case in cases)
    parsed = sum(bool(case["parse_validity"]) for case in cases)
    metric_values: dict[str, list[float]] = {}
    for case in cases:
        if not isinstance(case.get("metrics"), dict):
            continue
        for name, value in case["metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values.setdefault(name, []).append(float(value))
    report = {
        "version": "nlp-real-corpus-ast-gate-v1",
        "status": "pass" if passed == len(cases) else "blocked",
        "split": "validation_only",
        "test_read": False,
        "cases": len(cases),
        "passed": passed,
        "parser_valid": parsed,
        "parser_validity": parsed / max(1, len(cases)),
        "mean_metrics_on_parsed": {
            name: mean(values) for name, values in metric_values.items()
        },
        "target_ast_input": False,
        "repair_or_retry": False,
        "best_of_n": False,
        "cases_report": cases,
        "evidence_boundary": (
            "Strict real-corpus validation AST gate; no test access or capability claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: report[name]
        for name in (
            "version",
            "status",
            "cases",
            "passed",
            "parser_valid",
            "parser_validity",
            "mean_metrics_on_parsed",
        )
    }, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
