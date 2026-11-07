"""Teacher-forced and invalid-action audits for the typed AST grammar."""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTGrammarState
from newera.nlp_flow.components.ast.grammar.training.ast_grammar_training import (
    INVALID_CHILD,
    INVALID_FIELD,
    INVALID_ROOT,
)


def _mask_rejects(logits: torch.Tensor) -> bool:
    return bool(logits.isfinite().sum() == 0)


def replay_teacher_ast(
    grammar: ASTGrammarState,
    ast_payload: Mapping[str, object],
) -> dict[str, int]:
    holder = [grammar]
    legal_actions = 0
    invalid_probes = 0
    invalid_accepts = 0

    def probe(masked: torch.Tensor, index: int | None = None) -> None:
        nonlocal invalid_probes, invalid_accepts
        invalid_probes += 1
        rejected = (
            _mask_rejects(masked)
            if index is None
            else bool(masked[index] == -torch.inf)
        )
        invalid_accepts += int(not rejected)

    root = ast_payload["root"]
    root_candidates = (str(root["type"]), INVALID_ROOT)
    probe(
        grammar.mask_root_logits(
            torch.zeros(len(root_candidates)),
            root_candidates,
        ),
        1,
    )
    holder[0] = holder[0].start_root(str(root["type"]))
    legal_actions += 1

    def visit(node: Mapping[str, object], node_index: int, is_root: bool) -> None:
        nonlocal legal_actions
        for child in node["children"]:
            child_node = child["node"]
            current = holder[0]
            probe(
                current.mask_child_logits(
                    torch.zeros(2),
                    (str(child_node["type"]), INVALID_CHILD),
                    parent_index=node_index,
                    edge_field=str(child["field"]),
                ),
                1,
            )
            edge_candidates = (str(child["field"]), INVALID_FIELD)
            probe(
                current.mask_edge_logits(
                    torch.zeros(2),
                    edge_candidates,
                    parent_index=node_index,
                    child_kind=str(child_node["type"]),
                ),
                1,
            )
            probe(current.mask_eos_logits(torch.zeros(1)))
            holder[0] = current.add_child(
                parent_index=node_index,
                edge_field=str(child["field"]),
                child_kind=str(child_node["type"]),
            )
            legal_actions += 1
            child_index = len(holder[0].nodes) - 1
            visit(child_node, child_index, False)
        opened = str(node["type"]) in holder[0].scope_owner_kinds
        if not is_root and opened:
            holder[0] = holder[0].close_scope(holder[0].open_scopes[-1])
            legal_actions += 1
            closed_scope = holder[0].nodes[node_index].scope_id
            probe(
                holder[0].mask_scope_close_logits(
                    torch.zeros(1),
                    (int(closed_scope),),
                )
            )

    visit(root, 0, True)
    parent_indices = [node.parent_index for node in holder[0].nodes]
    if holder[0].topology_is_acyclic(parent_indices):
        cycle_logits = holder[0].mask_cycle_edges(
            torch.zeros(len(parent_indices), len(parent_indices)),
            parent_indices,
        )
        probe(torch.diagonal(cycle_logits))
    holder[0] = holder[0].close_scope(0)
    holder[0] = holder[0].finish()
    legal_actions += 2
    return {
        "nodes": len(holder[0].nodes),
        "legal_actions": legal_actions,
        "invalid_probes": invalid_probes,
        "invalid_accepts": invalid_accepts,
    }


def audit_teacher_replay(
    grammars: Mapping[str, ASTGrammarState],
    rows: list[Mapping[str, object]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    for language, grammar in sorted(grammars.items()):
        selected = [
            row
            for row in rows
            if row["split"] == "validation" and row["language"] == language
        ]
        audits = []
        failures = []
        for row in selected:
            try:
                result = replay_teacher_ast(
                    grammar,
                    json.loads(str(row["ast"])),
                )
                audits.append(result)
            except (KeyError, TypeError, ValueError) as error:
                failures.append(
                    {"case_id": row["case_id"], "error": str(error)}
                )
        results[language] = {
            "cases": len(selected),
            "passed": len(audits),
            "failed": failures,
            "legal_actions": sum(value["legal_actions"] for value in audits),
            "invalid_probes": sum(value["invalid_probes"] for value in audits),
            "invalid_accepts": sum(value["invalid_accepts"] for value in audits),
        }
    all_passed = all(
        value["cases"] == value["passed"]
        and value["invalid_accepts"] == 0
        for value in results.values()
    )
    return {
        "status": "pass" if all_passed else "fail",
        "split": "validation_only",
        "test_read": False,
        "languages": results,
        "total_cases": sum(value["cases"] for value in results.values()),
        "total_passed": sum(value["passed"] for value in results.values()),
        "total_invalid_accepts": sum(
            value["invalid_accepts"] for value in results.values()
        ),
        "repair_or_retry": False,
    }


__all__ = ["audit_teacher_replay", "replay_teacher_ast"]
