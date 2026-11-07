#!/usr/bin/env python3
"""Rebuild generated trees offline from frozen selected-ID traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import MAX_PROMPT, action_from_record
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_tree_runtime import _inference_context, viable_parent_kinds
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeActionDecoder,
    decode_actions,
    render_tree_structure,
)

ROOT = Path(__file__).resolve().parents[4]


def _load_rows(path: Path, case_ids: list[str]) -> dict[str, dict[str, Any]]:
    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    wanted = set(case_ids)
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) != "validation":
                continue
            row = json.loads(line)
            if row.get("split") == "validation" and row.get("case_id") in wanted:
                rows[str(row["case_id"])] = row
    if set(rows) != wanted:
        raise ValueError("saved traces do not match the frozen validation case IDs")
    return rows


def _selected(decision: dict[str, Any]) -> dict[str, int]:
    selected = {}
    for item in decision.get("choices", []):
        head, selected_id = item.get("head"), item.get("selected_id")
        if head is not None and selected_id is not None:
            selected[str(head)] = int(selected_id)
    return selected


def replay_decisions(
    *, prompt: str,
    language: str,
    decisions: list[dict[str, Any]],
    grammar: Any,
    vocabulary: dict[str, Any],
) -> tuple[str, Any]:
    decoder = TreeActionDecoder(grammar)
    parent_kinds = viable_parent_kinds(grammar)
    device = torch.device("cpu")
    for decision in decisions:
        selected = _selected(decision)
        kind = decision.get("action")
        if kind == "OpenScopeAction":
            if decoder.expected_scope is None:
                raise ValueError("trace opens a scope without an expected scope")
            action = OpenScopeAction(*decoder.expected_scope)
        elif kind == "OpenNodeAction":
            action = OpenNodeAction(vocabulary["node_names"][selected["node_kind"]])
        elif kind == "AttachEdgeAction":
            *_, candidates = _inference_context(decoder, parent_kinds, vocabulary, device)
            parent = candidates[selected["parent"]]
            edge = vocabulary["edge_names"][selected["edge_field"]]
            action = AttachEdgeAction(parent, edge)
        elif kind == "DeclareLeafAction":
            node_index = selected["leaf_node"]
            leaf_class = vocabulary["class_names"][selected["leaf_class"]]
            if "lexical" in selected:
                lexical = vocabulary["lexeme_names"][selected["lexical"]]
            elif "lexical_fallback" in selected:
                lexical = vocabulary["lexeme_names"][selected["lexical_fallback"]]
            elif "copy_start" in selected and "copy_length" in selected:
                start, length = selected["copy_start"], selected["copy_length"]
                lexical = prompt[start : min(start + length, MAX_PROMPT)]
            else:
                lexical = ""
            gap_id = selected.get("gap", selected.get("gap_fallback"))
            if gap_id is None:
                raise ValueError("leaf trace has no recorded gap choice")
            action = DeclareLeafAction(
                node_index,
                leaf_class,
                lexical,
                vocabulary["gap_names"][gap_id],
            )
        elif kind == "CloseScopeAction":
            action = CloseScopeAction(decoder.state.open_scopes[-1])
        elif kind == "TerminateAction":
            action = TerminateAction()
        else:
            raise ValueError(f"trace action cannot be replayed: {kind}")
        decoder.apply(action)
    return render_tree_structure(decoder.structure()), decoder.structure()


def _leaf_signature_counts(tree: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter(
        (str(node["kind"]), str(node.get("leaf_class")))
        for node in tree["nodes"]
        if not node["children"]
    )


def enrich_report(report_path: Path, corpus_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_legacy60_eval_manifest.json").read_text())
    rows = _load_rows(corpus_path, list(manifest["ordered_case_ids"]))
    language_summary: dict[str, dict[str, Any]] = {}
    overall_mismatches = Counter()
    for seed_report in report["seeds"]:
        seed = int(seed_report["seed"])
        checkpoint = torch.load(checkpoint_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=False)
        vocabulary = checkpoint["vocabulary"]
        for case in seed_report["cases"]:
            row = rows[case["case_id"]]
            language = str(case["language"])
            key = f"{seed}:{language}"
            summary = language_summary.setdefault(
                key,
                {"seed": seed, "language": language, "cases": 0, "reconstructed": 0,
                 "terminal_errors": Counter(), "parse_errors": Counter(), "node_kind_counts": Counter(),
                 "typed_leaf_signature_mismatch_cases": 0},
            )
            summary["cases"] += 1
            trace = case["live_trace"]
            try:
                grammar = load_ast_grammar(ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json", language)
                generated_source, generated_tree = replay_decisions(
                    prompt=str(row["prompt"]), language=language,
                    decisions=trace["decisions"], grammar=grammar, vocabulary=vocabulary,
                )
                case["reconstructed_source"] = generated_source
                case["reconstructed_source_sha256"] = hashlib.sha256(generated_source.encode()).hexdigest()
                case["trace_reconstruction_status"] = "pass"
                summary["reconstructed"] += 1
                for node in generated_tree["nodes"]:
                    summary["node_kind_counts"][str(node["kind"])] += 1
                gold_tree = decode_actions(
                    grammar,
                    [action_from_record(item) for item in row["actions"]],
                )
                gold_signature = _leaf_signature_counts(gold_tree)
                generated_signature = _leaf_signature_counts(generated_tree)
                missing = gold_signature - generated_signature
                extra = generated_signature - gold_signature
                case["typed_leaf_signature"] = {
                    "gold_counts": {f"{a}|{b}": n for (a, b), n in gold_signature.items()},
                    "generated_counts": {f"{a}|{b}": n for (a, b), n in generated_signature.items()},
                    "missing": {f"{a}|{b}": n for (a, b), n in missing.items()},
                    "extra": {f"{a}|{b}": n for (a, b), n in extra.items()},
                    "exact_match": not missing and not extra,
                }
                if missing or extra:
                    summary["typed_leaf_signature_mismatch_cases"] += 1
                    overall_mismatches.update({f"missing:{a}|{b}": n for (a, b), n in missing.items()})
                    overall_mismatches.update({f"extra:{a}|{b}": n for (a, b), n in extra.items()})
                if case["generated_graph_sha256"] is not None:
                    observed = hashlib.sha256(serialize_ast_graph(parse_ast_graph(generated_source, language)).encode()).hexdigest()
                    if observed != case["generated_graph_sha256"]:
                        raise ValueError("reconstructed tree hash differs from the captured generated graph hash")
            except (KeyError, IndexError, TypeError, ValueError) as error:
                case["trace_reconstruction_status"] = "failed"
                case["trace_reconstruction_error"] = str(error)
                summary["terminal_errors"][str(error)] += 1
            if case.get("generation_error"):
                summary["parse_errors"][str(case["generation_error"])] += 1
    report["trace_reconstruction"] = {
        "status": "pass" if all(row["reconstructed"] == row["cases"] for row in language_summary.values()) else "partial",
        "method": "offline replay of logged selected IDs against frozen prompt rows and checkpoint vocabularies; no model forward pass",
        "test_content_read": False,
        "per_seed_language": [
            {
                **row,
                "terminal_errors": dict(row["terminal_errors"]),
                "parse_errors": dict(row["parse_errors"]),
                "node_kind_counts": dict(row["node_kind_counts"]),
            }
            for row in language_summary.values()
        ],
        "typed_leaf_signature_mismatch_counts": dict(overall_mismatches),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["trace_reconstruction"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/reports/nlp_ast_oracle_decomposition_v12.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/train/nlp_hierarchical_tree_v12")
    args = parser.parse_args()
    result = enrich_report(args.report, args.corpus, args.checkpoint_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
