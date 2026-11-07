#!/usr/bin/env python3
"""Audit train-only canonical AST, source-byte and trivia replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import action_from_record
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import decode_actions, render_tree_structure
from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import TypedLexicalValue
from newera.nlp_flow.components.generation.trees.typed.typed_source_renderer import render_typed_source

ROOT = Path(__file__).resolve().parents[5]
ARM_IDS = ("oracle_lexical", "flat_source", "typed_projection")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed_values(graph: ASTGraph) -> list[TypedLexicalValue]:
    values = []
    next_id = 0

    def visit(node: ASTNode) -> None:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        if node.children:
            for _field, child in node.children:
                visit(child)
            return
        if node.lexical_value is None:
            raise ValueError(f"train AST leaf lacks lexical text: {node_id}")
        values.append(
            TypedLexicalValue(
                node_id=node_id,
                node_kind=node.node_type,
                leaf_class=classify_leaf(node.node_type, node.lexical_value),
                scope_id=node.scope_id,
                value=node.lexical_value,
                origin="generated",
                minimum_length=0,
                maximum_length=max(1, len(node.lexical_value)),
            )
        )

    visit(graph.root)
    return values


def _trivia_signature(node: ASTNode) -> tuple[str, ...]:
    if not node.children:
        return (node.gap_before,)
    return tuple(
        gap
        for _field, child in node.children
        for gap in _trivia_signature(child)
    )


def audit_train_replay(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, object]:
    action_path = root / config["training_action_corpus"]
    grammar_path = root / config["transition_grammar"]
    action_hash = _sha256(action_path)
    grammar_hash = _sha256(grammar_path)
    if action_hash != config["expected_training_action_sha256"]:
        raise ValueError("train-action corpus identity differs from its pinned hash")
    if grammar_hash != config["expected_transition_grammar_sha256"]:
        raise ValueError("transition grammar identity differs from its pinned hash")
    grammars = {
        language: load_ast_grammar(grammar_path, language)
        for language in config["languages"]
    }
    counters = {
        language: Counter(
            cases=0,
            canonical_replays=0,
            source_byte_exact=0,
            trivia_byte_exact=0,
            unsupported=0,
        )
        for language in config["languages"]
    }
    failures = []
    rows = 0
    with action_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"non-train row in the pinned training action corpus: line {line_number}")
            language = str(row["language"])
            if language not in counters:
                raise ValueError(f"unsupported train corpus language: {language}")
            count = counters[language]
            count["cases"] += 1
            rows += 1
            try:
                actions = [action_from_record(value) for value in row["actions"]]
                structure = decode_actions(grammars[language], actions)
                source = render_tree_structure(structure)
                if hashlib.sha256(source.encode("utf-8")).hexdigest() != row["target_source_sha256"]:
                    raise ValueError("teacher action source hash mismatch")
                graph = parse_ast_graph(source, language)
                rendered = render_typed_source(graph, _typed_values(graph))
                replay = parse_ast_graph(rendered.source, language)
                if rendered.canonical_signature[0] != replay.root.node_type:
                    raise ValueError("canonical replay root mismatch")
                count["canonical_replays"] += 1
                if rendered.source.encode("utf-8") == source.encode("utf-8"):
                    count["source_byte_exact"] += 1
                if _trivia_signature(graph.root) == _trivia_signature(replay.root):
                    count["trivia_byte_exact"] += 1
            except (KeyError, TypeError, ValueError, UnicodeError) as error:
                count["unsupported"] += 1
                failures.append(
                    {
                        "case_id": row.get("case_id", f"line-{line_number}"),
                        "language": language,
                        "reason": str(error).splitlines()[0][:160],
                    }
                )
    if rows != int(config["expected_training_rows"]):
        raise ValueError(f"train action row count differs from its pinned value: {rows}")
    language_audit = {}
    for language, count in counters.items():
        cases = count["cases"]
        language_audit[language] = {
            **dict(count),
            "canonical_replay_rate": count["canonical_replays"] / max(1, cases),
            "source_byte_exact_rate": count["source_byte_exact"] / max(1, cases),
            "trivia_byte_exact_rate": count["trivia_byte_exact"] / max(1, cases),
        }
    metric_names = [metric["name"] for metric in config["metric_schema"]]
    return {
        "version": "nlp-typed-source-contract-v11",
        "status": "pass_train_only_replay" if not failures else "partial_unsupported_train_rows",
        "input_identity": {
            "training_actions": {"path": config["training_action_corpus"], "sha256": action_hash},
            "transition_grammar": {"path": config["transition_grammar"], "sha256": grammar_hash},
        },
        "train_rows": rows,
        "language_audit": language_audit,
        "unsupported_rows": failures,
        "matched_dry_run": {
            "arms": [
                {"id": arm["id"], "metrics": {name: None for name in metric_names}}
                for arm in config["arms"]
            ],
            "training_performed": False,
            "gpu_used": False,
            "winner_selected": False,
            "quality_claim": False,
        },
        "validation_rows_read": False,
        "test_rows_read": False,
        "evidence_boundary": config["evidence_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/structured_generation/transition_grammar_ast/lexical_matched_policy/nlp_lexical_matched.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("version") != "nlp-lexical-matched-v11":
        raise ValueError("unsupported lexical replay configuration")
    if tuple(arm.get("id") for arm in config["arms"]) != ARM_IDS:
        raise ValueError(f"expected lexical dry-run arms in order: {ARM_IDS}")
    report = audit_train_replay(config)
    output = args.output or ROOT / config["report"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_rows": report["train_rows"],
                "language_audit": report["language_audit"],
                "test_rows_read": report["test_rows_read"],
            },
            indent=2,
        )
    )
    if report["status"] != "pass_train_only_replay":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
