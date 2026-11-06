"""Matched action-selection ablation for structural grammar masks."""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_grammar_training import (
    INVALID_CHILD,
    INVALID_FIELD,
    INVALID_ROOT,
    ASTGrammarExample,
    ASTGrammarLegalityHead,
    encode_examples_with_indices,
)


def _action_groups(
    language: str,
    payload: Mapping[str, object],
) -> list[list[ASTGrammarExample]]:
    groups = []
    root = payload["root"]

    def candidates(
        parent: str,
        edge: str,
        child: str,
    ) -> list[ASTGrammarExample]:
        return [
            ASTGrammarExample(language, "child", parent, edge, child, 1, "teacher"),
            ASTGrammarExample(
                language,
                "child",
                parent,
                edge,
                INVALID_CHILD,
                0,
                "illegal_child",
            ),
            ASTGrammarExample(
                language,
                "child",
                parent,
                INVALID_FIELD,
                child,
                0,
                "illegal_edge",
            ),
        ]

    groups.append(
        [
            ASTGrammarExample(
                language,
                "root",
                "<start>",
                "*",
                str(root["type"]),
                1,
                "teacher",
            ),
            ASTGrammarExample(
                language,
                "root",
                "<start>",
                "*",
                INVALID_ROOT,
                0,
                "illegal_root",
            ),
        ]
    )

    def visit(node: Mapping[str, object]) -> None:
        for child in node["children"]:
            child_node = child["node"]
            groups.append(
                candidates(
                    str(node["type"]),
                    str(child["field"]),
                    str(child_node["type"]),
                )
            )
            visit(child_node)

    visit(root)
    return groups


def _decision(
    model: ASTGrammarLegalityHead,
    group: list[ASTGrammarExample],
    indices: Mapping[str, Mapping[str, int]],
    device: torch.device,
    *,
    constrained: bool,
) -> int:
    if constrained:
        return 0
    columns, _labels = encode_examples_with_indices(group, indices)
    with torch.no_grad():
        logits = model(
            *(columns[name].to(device) for name in columns)
        )
    return int(logits.argmax())


def _tree_counts(payload: Mapping[str, object]) -> tuple[int, int, int]:
    nodes = 0
    edges = 0

    def visit(node: Mapping[str, object]) -> None:
        nonlocal nodes, edges
        nodes += 1
        edges += len(node["children"])
        for child in node["children"]:
            visit(child["node"])

    visit(payload["root"])
    return nodes, edges, len(payload["scopes"])


def evaluate_action_ablation(
    model: ASTGrammarLegalityHead,
    indices: Mapping[str, Mapping[str, int]],
    rows: list[Mapping[str, object]],
    *,
    device: torch.device,
) -> dict[str, object]:
    arms = {
        "grammar_constrained": {
            "parser_validity": [],
            "node_f1": [],
            "edge_f1": [],
            "scope_f1": [],
            "invalid_actions": 0,
            "decisions": 0,
        },
        "unstructured_head": {
            "parser_validity": [],
            "node_f1": [],
            "edge_f1": [],
            "scope_f1": [],
            "invalid_actions": 0,
            "decisions": 0,
        },
    }
    selected = [row for row in rows if row["split"] == "validation"]
    for row in selected:
        payload = json.loads(str(row["ast"]))
        groups = _action_groups(str(row["language"]), payload)
        source = str(row["source"])
        expected_nodes, expected_edges, expected_scopes = _tree_counts(payload)
        for arm_name, constrained in (
            ("grammar_constrained", True),
            ("unstructured_head", False),
        ):
            arm = arms[arm_name]
            exact = True
            for group in groups:
                choice = _decision(
                    model,
                    group,
                    indices,
                    device,
                    constrained=constrained,
                )
                arm["decisions"] += 1
                if choice != 0:
                    exact = False
                    arm["invalid_actions"] += 1
            if exact:
                try:
                    generated = parse_ast_graph(source, str(row["language"]))
                    valid = True
                    actual_nodes = sum(
                        1 for _path, _node in _walk(generated)
                    )
                    actual_edges = actual_nodes - 1
                    node_f1 = float(actual_nodes == expected_nodes)
                    edge_f1 = float(actual_edges == expected_edges)
                    scope_f1 = float(len(generated.scopes) == expected_scopes)
                except (TypeError, ValueError, UnicodeError):
                    valid = False
                    node_f1 = edge_f1 = scope_f1 = 0.0
            else:
                valid = False
                node_f1 = edge_f1 = scope_f1 = 0.0
            arm["parser_validity"].append(float(valid))
            arm["node_f1"].append(node_f1)
            arm["edge_f1"].append(edge_f1)
            arm["scope_f1"].append(scope_f1)
    summary = {}
    for arm_name, arm in arms.items():
        cases = len(arm["parser_validity"])
        summary[arm_name] = {
            "cases": cases,
            "parser_validity": sum(arm["parser_validity"]) / max(1, cases),
            "node_f1": sum(arm["node_f1"]) / max(1, cases),
            "edge_f1": sum(arm["edge_f1"]) / max(1, cases),
            "scope_f1": sum(arm["scope_f1"]) / max(1, cases),
            "invalid_actions": arm["invalid_actions"],
            "decisions": arm["decisions"],
        }
    return {
        "status": (
            "pass"
            if summary["grammar_constrained"]["invalid_actions"] == 0
            and summary["grammar_constrained"]["parser_validity"] == 1.0
            else "fail"
        ),
        "split": "validation_only",
        "test_read": False,
        "matched_action_candidates": True,
        "repair_or_retry": False,
        "arms": summary,
        "source_kind": "teacher_ast_projection_after_exact_action_replay",
        "evidence_boundary": (
            "Matched action-selection audit only. It demonstrates mask effects "
            "on candidate actions; it is not open-domain AST or executable-code "
            "generation evidence."
        ),
    }


def _walk(
    graph,
):
    def visit(node):
        yield (), node
        for index, (_field, child) in enumerate(node.children):
            for path, value in visit(child):
                yield (index, *path), value

    yield from visit(graph.root)


__all__ = ["evaluate_action_ablation"]
