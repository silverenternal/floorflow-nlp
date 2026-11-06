"""Deterministic AST grammar negatives derived only from training rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from floorflow.nlp_flow.ast_codec import parse_ast_graph

SEMANTIC_WRONG_CASES: tuple[dict[str, str], ...] = (
    {
        "language": "python",
        "source": "def add(a, b):\n    return a - b\n",
    },
    {
        "language": "rust",
        "source": "fn add(a: i32, b: i32) -> i32 { a - b }\n",
    },
    {
        "language": "cpp",
        "source": "int add(int a, int b) { return a - b; }\n",
    },
)


def _walk_payload(node: Mapping[str, Any]):
    yield node
    for child in node["children"]:
        yield from _walk_payload(child["node"])


def _first_edge(root: Mapping[str, Any]) -> tuple[str, str, str]:
    for parent in _walk_payload(root):
        for child in parent["children"]:
            child_node = child["node"]
            if child_node["type"] != "trivia":
                return (
                    str(parent["type"]),
                    str(child["field"]),
                    str(child_node["type"]),
                )
    raise ValueError("training AST has no non-trivia child edge")


def _training_rows_by_language(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    output: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        language = str(row["language"])
        output.setdefault(language, []).append(row)
    for language_rows in output.values():
        language_rows.sort(
            key=lambda row: (str(row["source_sha256"]), str(row["case_id"]))
        )
    return output


def _grammar_negatives(
    language: str,
    row: Mapping[str, object],
) -> list[dict[str, object]]:
    payload = json.loads(str(row["ast"]))
    root = payload["root"]
    observed_root = str(root["type"])
    parent, field, child = _first_edge(root)
    common = {
        "split": "train",
        "source_split": "train",
        "provenance_hashes": [str(row["source_sha256"])],
        "target_log_probability": "negative_infinity",
        "training_allowed": True,
    }
    return [
        {
            **common,
            "case_id": f"{language}-illegal-root",
            "language": language,
            "kind": "illegal_root",
            "state": {"expected_root": observed_root, "observed_root": "ERROR"},
        },
        {
            **common,
            "case_id": f"{language}-illegal-child",
            "language": language,
            "kind": "illegal_child",
            "state": {
                "parent": parent,
                "field": field,
                "observed_child": "__unknown_child__",
            },
        },
        {
            **common,
            "case_id": f"{language}-illegal-edge",
            "language": language,
            "kind": "illegal_edge",
            "state": {
                "parent": parent,
                "expected_field": field,
                "child": child,
                "observed_field": "__unknown_field__",
            },
        },
        {
            **common,
            "case_id": f"{language}-illegal-scope",
            "language": language,
            "kind": "illegal_scope",
            "state": {
                "scope_ids": [int(scope["scope_id"]) for scope in payload["scopes"]],
                "observed_scope_id": int(payload["scopes"][-1]["scope_id"]) + 2,
            },
        },
        {
            **common,
            "case_id": f"{language}-parent-cycle",
            "language": language,
            "kind": "parent_cycle",
            "state": {
                "scope_id": int(payload["scopes"][-1]["scope_id"]),
                "observed_parent_scope_id": int(payload["scopes"][-1]["scope_id"]),
            },
        },
        {
            **common,
            "case_id": f"{language}-premature-eos",
            "language": language,
            "kind": "premature_eos",
            "state": {
                "expected_root_closed": True,
                "observed_root_closed": False,
                "eos_action": "end_sequence",
            },
        },
    ]


def _semantic_negatives() -> list[dict[str, object]]:
    cases = []
    for fixture in SEMANTIC_WRONG_CASES:
        language = fixture["language"]
        source = fixture["source"]
        graph = parse_ast_graph(source, language)
        cases.append(
            {
                "case_id": f"{language}-valid-but-semantically-wrong",
                "language": language,
                "kind": "valid_but_semantically_wrong",
                "split": "train",
                "source_split": "synthesized_train",
                "source": source,
                "source_sha256": graph.source_sha256,
                "ast_parse_valid": True,
                "expected_semantics": "a + b",
                "observed_semantics": "a - b",
                "target_log_probability": "negative_infinity",
                "training_allowed": True,
                "provenance_hashes": [],
            }
        )
    return cases


def build_ast_negative_suite(
    corpus_rows: Sequence[Mapping[str, object]],
    *,
    grammar_cases_per_language: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_language = _training_rows_by_language(corpus_rows)
    cases: list[dict[str, object]] = []
    for language in sorted(by_language):
        rows = by_language[language]
        if not rows:
            continue
        row = rows[grammar_cases_per_language % len(rows)]
        cases.extend(_grammar_negatives(language, row))
    cases.extend(_semantic_negatives())
    excluded_hashes = {
        str(row["source_sha256"])
        for row in corpus_rows
        if row["split"] in {"validation", "test"}
    }
    provenance_hashes = {
        str(source_hash)
        for case in cases
        for source_hash in case["provenance_hashes"]
    }
    kind_counts: dict[str, int] = {}
    for case in cases:
        kind = str(case["kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    required_kinds = {
        "illegal_root",
        "illegal_child",
        "illegal_edge",
        "illegal_scope",
        "parent_cycle",
        "premature_eos",
        "valid_but_semantically_wrong",
    }
    report = {
        "version": "nlp-ast-negative-suite-v1",
        "cases": len(cases),
        "kind_counts": kind_counts,
        "required_kinds_present": required_kinds == set(kind_counts),
        "validation_test_provenance": sorted(provenance_hashes & excluded_hashes),
        "all_training_only": all(
            case["split"] == "train"
            and case["source_split"] in {"train", "synthesized_train"}
            for case in cases
        ),
        "deterministic": True,
        "evidence_boundary": (
            "Supervised grammar and semantic negatives only; no model mask, "
            "generation-validity, or held-out capability result."
        ),
    }
    report["status"] = (
        "pass"
        if report["required_kinds_present"]
        and not report["validation_test_provenance"]
        and report["all_training_only"]
        else "fail"
    )
    return cases, report


def suite_sha256(cases: Sequence[Mapping[str, object]]) -> str:
    payload = "".join(
        json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n"
        for case in cases
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SEMANTIC_WRONG_CASES",
    "build_ast_negative_suite",
    "suite_sha256",
]
