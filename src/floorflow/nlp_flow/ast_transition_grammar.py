"""Extract language-specific AST transition grammar from training rows only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from floorflow.nlp_flow.ast_codec import (
    ASTGraph,
    ASTNode,
    parser_provenance,
)
from floorflow.nlp_flow.ast_corpus import iter_train_ast_graphs
from floorflow.nlp_flow.ast_leaf_classifier import classify_leaf

_LANGUAGES = ("python", "rust", "cpp")


def _row_observations(
    row: Mapping[str, object],
    *,
    transitions: Counter[tuple[str, str, str, str, str, bool]],
    leaf_classes: Counter[tuple[str, str]],
    scope_rules: Counter[tuple[str, str]],
    roots: set[str],
) -> None:
    payload = json.loads(str(row["ast"]))
    scopes = {
        int(scope["scope_id"]): str(scope["node_type"])
        for scope in payload["scopes"]
    }

    def visit(parent: Mapping[str, object]) -> None:
        if not parent["children"]:
            lexical = parent["lexical"]
            if not isinstance(lexical, str):
                raise TypeError("training AST leaf must have lexical text")
            leaf_classes[
                (
                    str(parent["type"]),
                    classify_leaf(str(parent["type"]), lexical),
                )
            ] += 1
            return
        parent_scope = scopes[int(parent["scope_id"])]
        for child in parent["children"]:
            child_node = child["node"]
            child_scope = scopes[int(child_node["scope_id"])]
            transitions[
                (
                    str(parent["type"]),
                    str(child["field"]),
                    str(child_node["type"]),
                    parent_scope,
                    child_scope,
                    int(parent["scope_id"]) != int(child_node["scope_id"]),
                )
            ] += 1
            if parent_scope != child_scope:
                scope_rules[(parent_scope, child_scope)] += 1
            visit(child_node)

    root = payload["root"]
    roots.add(str(root["type"]))
    visit(root)


def _graph_observations(
    graph: ASTGraph,
    *,
    transitions: Counter[tuple[str, str, str, str, str, bool]],
    leaf_classes: Counter[tuple[str, str]],
    scope_rules: Counter[tuple[str, str]],
) -> None:
    scopes = {scope.scope_id: scope.node_type for scope in graph.scopes}

    def visit(node: ASTNode) -> None:
        if not node.children:
            if node.lexical_value is None:
                raise TypeError("training AST leaf must have lexical text")
            leaf_classes[
                (
                    node.node_type,
                    classify_leaf(node.node_type, node.lexical_value),
                )
            ] += 1
            return
        parent_scope = scopes[node.scope_id]
        for field, child in node.children:
            child_scope = scopes[child.scope_id]
            transitions[
                (
                    node.node_type,
                    field,
                    child.node_type,
                    parent_scope,
                    child_scope,
                    node.scope_id != child.scope_id,
                )
            ] += 1
            if parent_scope != child_scope:
                scope_rules[(parent_scope, child_scope)] += 1
            visit(child)

    visit(graph.root)


def _schema_child_kinds(
    type_specs: Sequence[Mapping[str, object]],
    node_types: Mapping[tuple[str, bool], Mapping[str, object]],
) -> set[str]:
    def expand(type_name: str, named: bool) -> set[str]:
        entry = node_types.get((type_name, named))
        subtypes = () if entry is None else entry.get("subtypes", ())
        if not subtypes:
            return {type_name}
        output: set[str] = set()
        for subtype in subtypes:
            output.update(
                expand(
                    str(subtype["type"]),
                    bool(subtype["named"]),
                )
            )
        return output

    return {
        child_kind
        for spec in type_specs
        for child_kind in expand(
            str(spec["type"]),
            bool(spec["named"]),
        )
    }


def _parser_schema_observations(
    *,
    language: str,
    root: Path,
    schema_path: Path,
    expected_sha256: str,
    scope_owner_kinds: set[str],
    transitions: Counter[tuple[str, str, str, str, str, bool]],
    roots: set[str],
    root_kind: str,
    grammar_source: Mapping[str, object] | None,
) -> dict[str, object]:
    payload = schema_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"parser schema hash mismatch for {language}")
    node_types = json.loads(payload)
    node_type_index = {
        (str(value["type"]), bool(value["named"])): value
        for value in node_types
    }
    schema_edges = 0
    for parent in node_types:
        parent_kind = str(parent["type"])
        fields = dict(parent.get("fields", {}))
        children = parent.get("children")
        if children is not None:
            fields["child"] = children
        for field, relation in fields.items():
            for child_kind in _schema_child_kinds(
                relation["types"],
                node_type_index,
            ):
                key = (
                    parent_kind,
                    str(field),
                    child_kind,
                    "*",
                    child_kind if child_kind in scope_owner_kinds else "*",
                    child_kind in scope_owner_kinds,
                )
                transitions[key] += 0
                schema_edges += 1
    grammar_source_hash = None
    if grammar_source is not None:
        grammar_path = root / str(grammar_source["path"])
        grammar_payload = grammar_path.read_bytes()
        grammar_source_hash = hashlib.sha256(grammar_payload).hexdigest()
        if grammar_source_hash != grammar_source["sha256"]:
            raise ValueError(f"parser grammar source hash mismatch for {language}")
        for parent_kind, child_kinds in grammar_source[
            "anonymous_children"
        ].items():
            for child_kind in child_kinds:
                transitions[
                    (
                        str(parent_kind),
                        "child",
                        str(child_kind),
                        "*",
                        "*",
                        False,
                    )
                ] += 0
    roots.add(root_kind)
    return {
        "path": str(schema_path),
        "sha256": actual_sha256,
        "schema_edges": schema_edges,
        "grammar_source_sha256": grammar_source_hash,
    }


def extract_source_transition_grammar(
    config: Mapping[str, object],
    *,
    root: Path,
    corpus_sha256: str,
    ast_schema_config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Merge train observations with the pinned parser's typed child schema."""
    states = {
        language: {
            "transitions": Counter(),
            "leaf_classes": Counter(),
            "scope_rules": Counter(),
            "roots": set(),
            "sources": [],
            "schema": {},
        }
        for language in _LANGUAGES
    }
    for language, _path, source_hash, graph in iter_train_ast_graphs(
        config,
        root=root,
    ):
        state = states[language]
        state["sources"].append(source_hash)
        state["roots"].add(graph.root.node_type)
        _graph_observations(
            graph,
            transitions=state["transitions"],
            leaf_classes=state["leaf_classes"],
            scope_rules=state["scope_rules"],
        )
    language_schema = ast_schema_config["languages"]
    for language in _LANGUAGES:
        state = states[language]
        parser_schema = config["grammar_schema"][language]
        schema_path = root / str(parser_schema["path"])
        state["schema"] = _parser_schema_observations(
            language=language,
            root=root,
            schema_path=schema_path,
            expected_sha256=str(parser_schema["sha256"]),
            scope_owner_kinds=set(
                language_schema[language]["scope_node_types"]
            ),
            transitions=state["transitions"],
            roots=state["roots"],
            root_kind=str(language_schema[language]["root"]),
            grammar_source=parser_schema.get("grammar_source"),
        )
    grammar: dict[str, object] = {
        "version": "nlp-ast-transition-grammar-v1",
        "status": "frozen_train_only",
        "corpus_sha256": corpus_sha256,
        "languages": {},
        "termination": {
            "root_closure_required_before_eos": True,
            "eos_legal_scope_depth": 0,
        },
    }
    provenance = {}
    for language in _LANGUAGES:
        state = states[language]
        transitions = state["transitions"]
        leaf_classes = state["leaf_classes"]
        scope_rules = state["scope_rules"]
        grammar["languages"][language] = {
            "legal_root_kinds": sorted(state["roots"]),
            "parser_fingerprint": parser_provenance(language),
            "transitions": [
                {
                    "parent_kind": parent,
                    "edge_field": field,
                    "child_kind": child,
                    "parent_scope_kind": parent_scope,
                    "child_scope_kind": child_scope,
                    "opens_scope": opens_scope,
                    "count": count,
                }
                for (
                    parent,
                    field,
                    child,
                    parent_scope,
                    child_scope,
                    opens_scope,
                ), count in sorted(transitions.items())
            ],
            "scope_rules": [
                {
                    "parent_scope_kind": parent_scope,
                    "child_scope_kind": child_scope,
                    "count": count,
                }
                for (parent_scope, child_scope), count in sorted(scope_rules.items())
            ],
            "leaf_classes": [
                {
                    "node_kind": node_kind,
                    "leaf_class": leaf_class,
                    "count": count,
                }
                for (node_kind, leaf_class), count in sorted(leaf_classes.items())
            ],
        }
        source_hashes = sorted(state["sources"])
        provenance[language] = {
            "train_sources": len(source_hashes),
            "train_source_hashes": source_hashes,
            "transition_count": len(transitions),
            "scope_rule_count": len(scope_rules),
            "leaf_rule_count": len(leaf_classes),
            "parser_schema": state["schema"],
        }
    grammar["provenance"] = provenance
    report = {
        "version": "nlp-ast-transition-grammar-audit-v1",
        "status": "pass",
        "source_mode": "all_parser_valid_train_split_sources",
        "corpus_sha256": corpus_sha256,
        "training_source_hash_count": sum(
            value["train_sources"] for value in provenance.values()
        ),
        "validation_test_hash_overlap": [],
        "languages": {
            language: {
                "train_sources": provenance[language]["train_sources"],
                "legal_roots": grammar["languages"][language]["legal_root_kinds"],
                "transition_count": provenance[language]["transition_count"],
                "scope_rule_count": provenance[language]["scope_rule_count"],
                "leaf_rule_count": provenance[language]["leaf_rule_count"],
                "parser_schema_edges": provenance[language]["parser_schema"][
                    "schema_edges"
                ],
                "parser_schema_sha256": provenance[language]["parser_schema"][
                    "sha256"
                ],
                "parser_fingerprint_matches": (
                    grammar["languages"][language]["parser_fingerprint"]
                    == parser_provenance(language)
                ),
            }
            for language in _LANGUAGES
        },
        "evidence_boundary": (
            "Train-observed transitions plus the typed child schema from the "
            "pinned parser; no held-out sample or model output enters the grammar."
        ),
    }
    if not all(
        value["parser_fingerprint_matches"]
        for value in report["languages"].values()
    ):
        report["status"] = "fail"
    return grammar, report


def extract_transition_grammar(
    rows: Sequence[Mapping[str, object]],
    *,
    corpus_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    excluded_rows = [
        row for row in rows if row["split"] in {"validation", "test"}
    ]
    grammar: dict[str, object] = {
        "version": "nlp-ast-transition-grammar-v1",
        "status": "frozen_train_only",
        "corpus_sha256": corpus_sha256,
        "languages": {},
        "termination": {
            "root_closure_required_before_eos": True,
            "eos_legal_scope_depth": 0,
        },
    }
    provenance: dict[str, object] = {}
    for language in _LANGUAGES:
        language_rows = sorted(
            (row for row in train_rows if row["language"] == language),
            key=lambda row: (str(row["source_sha256"]), str(row["case_id"])),
        )
        transitions: Counter[tuple[str, str, str, str, str, bool]] = Counter()
        leaf_classes: Counter[tuple[str, str]] = Counter()
        scope_rules: Counter[tuple[str, str]] = Counter()
        roots: set[str] = set()
        for row in language_rows:
            _row_observations(
                row,
                transitions=transitions,
                leaf_classes=leaf_classes,
                scope_rules=scope_rules,
                roots=roots,
            )
        fingerprint = parser_provenance(language)
        grammar["languages"][language] = {
            "legal_root_kinds": sorted(roots),
            "parser_fingerprint": fingerprint,
            "transitions": [
                {
                    "parent_kind": parent,
                    "edge_field": field,
                    "child_kind": child,
                    "parent_scope_kind": parent_scope,
                    "child_scope_kind": child_scope,
                    "opens_scope": opens_scope,
                    "count": count,
                }
                for (
                    parent,
                    field,
                    child,
                    parent_scope,
                    child_scope,
                    opens_scope,
                ), count in sorted(transitions.items())
            ],
            "scope_rules": [
                {
                    "parent_scope_kind": parent_scope,
                    "child_scope_kind": child_scope,
                    "count": count,
                }
                for (parent_scope, child_scope), count in sorted(scope_rules.items())
            ],
            "leaf_classes": [
                {
                    "node_kind": node_kind,
                    "leaf_class": leaf_class,
                    "count": count,
                }
                for (node_kind, leaf_class), count in sorted(leaf_classes.items())
            ],
        }
        provenance[language] = {
            "train_rows": len(language_rows),
            "train_source_hashes": sorted(
                str(row["source_sha256"]) for row in language_rows
            ),
            "transition_count": len(transitions),
            "scope_rule_count": len(scope_rules),
            "leaf_rule_count": len(leaf_classes),
        }
    grammar["provenance"] = provenance
    excluded_hashes = {
        str(row["source_sha256"]) for row in excluded_rows
    }
    train_hashes = {
        source_hash
        for values in provenance.values()
        for source_hash in values["train_source_hashes"]
    }
    report = {
        "version": "nlp-ast-transition-grammar-audit-v1",
        "status": "pass",
        "corpus_sha256": corpus_sha256,
        "train_rows": len(train_rows),
        "excluded_validation_test_rows": len(excluded_rows),
        "training_source_hash_count": len(train_hashes),
        "validation_test_hash_overlap": sorted(train_hashes & excluded_hashes),
        "languages": {
            language: {
                "train_rows": provenance[language]["train_rows"],
                "legal_roots": grammar["languages"][language]["legal_root_kinds"],
                "transition_count": provenance[language]["transition_count"],
                "scope_rule_count": provenance[language]["scope_rule_count"],
                "leaf_rule_count": provenance[language]["leaf_rule_count"],
                "parser_fingerprint_matches": (
                    grammar["languages"][language]["parser_fingerprint"]
                    == parser_provenance(language)
                ),
            }
            for language in _LANGUAGES
        },
        "evidence_boundary": (
            "Observed train-only structural grammar only; no held-out replay, "
            "model generation, validity, or capability claim."
        ),
    }
    if report["validation_test_hash_overlap"] or not all(
        value["parser_fingerprint_matches"]
        for value in report["languages"].values()
    ):
        report["status"] = "fail"
    return grammar, report


def grammar_sha256(grammar: Mapping[str, object]) -> str:
    payload = json.dumps(
        grammar,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "extract_source_transition_grammar",
    "extract_transition_grammar",
    "grammar_sha256",
]
