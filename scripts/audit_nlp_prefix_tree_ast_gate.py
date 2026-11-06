#!/usr/bin/env python3
"""Run the strict validation gate for the prefix transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_quality_gate import evaluate_ast_upstream_source
from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy
from floorflow.nlp_flow.prefix_tree_runtime import generate_prefix_tree
from floorflow.nlp_flow.real_corpus_policy_data import (
    action_from_record,
    load_action_rows,
)
from floorflow.nlp_flow.tree_action_decoder import (
    decode_actions,
    render_tree_structure,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/nlp_prefix_tree_policy_v1/checkpoint.pt",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_prefix_tree_ast_gate_v1.json",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(
        args.device
        or (
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    config = checkpoint["config"]
    max_steps = int(checkpoint["model"]["step_position_embedding.weight"].shape[0])
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=int(config["hidden_size"]),
        layers=int(config["decoder_layers"]),
        heads=int(config["decoder_heads"]),
        max_steps=max_steps,
    ).to(device)
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
    for index, row in enumerate(rows, start=1):
        language = str(row["language"])
        grammar = grammars[language]
        target_source = render_tree_structure(
            decode_actions(
                grammar,
                [action_from_record(value) for value in row["actions"]],
            )
        )
        generated_source, actions, error = generate_prefix_tree(
            model,
            str(row["prompt"]),
            language,
            grammar,
            vocabulary,
            device=device,
            max_steps=max_steps,
        )
        if generated_source is None:
            gate = {
                "status": "fail",
                "parse_validity": False,
                "source_kind": "prefix_tree_endpoint",
                "checks": {
                    "parse_validity": False,
                    "node_type_f1": False,
                    "typed_edge_f1": False,
                    "scope_relation_f1": False,
                    "lexical_f1": False,
                    "tree_edit_distance": False,
                },
                "metrics": None,
                "failure": error,
            }
        else:
            reference = parse_ast(target_source, language)
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
                "generation_error": error,
                **gate,
            }
        )
        print(f"{index}/{len(rows)} {gate['status']} {error or 'parsed'}")
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
        "version": "nlp-prefix-tree-ast-gate-v1",
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
        "evidence_boundary": "Strict validation gate for the prefix transformer only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: report[name]
        for name in (
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
