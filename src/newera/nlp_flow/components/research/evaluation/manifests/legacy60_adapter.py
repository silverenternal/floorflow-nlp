"""Adapt canonical graph predictions to the frozen 60-case AST evaluator."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ast_metrics, parser_fingerprint
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import deserialize_ast_graph, serialize_ast_graph

_ROOT = Path(__file__).resolve().parents[7]
_REQUIRED_CODE_GROUPS = ("canonicalizer", "metrics", "legacy_adapter")


def _verify_code_bindings(manifest: Mapping[str, Any]) -> None:
    code_groups = manifest.get("implementation_sha256")
    if not isinstance(code_groups, Mapping):
        raise TypeError("legacy evaluator implementation fingerprints are missing")
    for group in _REQUIRED_CODE_GROUPS:
        files = code_groups.get(group)
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"legacy evaluator fingerprint group is missing: {group}")
        for relative, expected in files.items():
            path = (_ROOT / str(relative)).resolve(strict=True)
            if not path.is_relative_to(_ROOT):
                raise ValueError("legacy evaluator fingerprint path escapes the repository")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"legacy evaluator implementation changed: {relative}")

def _case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if manifest.get("version") != "nlp-legacy60-evaluator-v11":
        raise ValueError("unsupported legacy evaluator manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 60:
        raise ValueError("legacy evaluator must freeze exactly 60 cases")
    by_id = {str(case.get("case_id") or ""): case for case in cases if isinstance(case, Mapping)}
    if len(by_id) != 60 or "" in by_id:
        raise ValueError("legacy evaluator case IDs must be unique and nonempty")
    if manifest.get("ordered_case_ids") != [case["case_id"] for case in cases]:
        raise ValueError("legacy evaluator case order disagrees with its manifest")
    _verify_code_bindings(manifest)
    return by_id


def _verify_parser_binding(language: str, manifest: Mapping[str, Any]) -> None:
    bindings = manifest.get("parser_fingerprints")
    if not isinstance(bindings, Mapping) or language not in bindings:
        raise ValueError("legacy evaluator has no parser fingerprint for this language")
    expected = bindings[language]
    current = parser_fingerprint(language)
    observed = {
        "package": current.package.replace("_", "-"),
        "version": current.package_version,
        "abi_version": current.abi_version,
        "semantic_version": None if current.semantic_version is None else list(current.semantic_version),
        "native_sha256": current.native_sha256,
        "grammar_sha256": current.grammar_sha256,
    }
    if dict(expected) != observed:
        raise ValueError("runtime parser fingerprint differs from the frozen evaluator")


def adapt_new_ir_prediction(
    serialized_prediction: str,
    *,
    case_id: str,
    manifest: Mapping[str, Any],
) -> ASTGraph:
    """Decode a prediction into the original evaluator's canonical graph type."""
    cases = _case_map(manifest)
    case = cases.get(case_id)
    if case is None:
        raise ValueError(f"case is not in the frozen legacy suite: {case_id}")
    language = case.get("language")
    if not isinstance(language, str):
        raise TypeError("frozen case language is invalid")
    _verify_parser_binding(language, manifest)
    prediction = deserialize_ast_graph(serialized_prediction)
    if prediction.language != language:
        raise ValueError("prediction language differs from the frozen case")
    return prediction


def evaluate_legacy60_prediction(
    serialized_prediction: str,
    *,
    case_id: str,
    reference_graph: ASTGraph,
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Score against the untouched reference graph and frozen per-case gates."""
    cases = _case_map(manifest)
    case = cases.get(case_id)
    if case is None:
        raise ValueError(f"case is not in the frozen legacy suite: {case_id}")
    if reference_graph.language != case.get("language"):
        raise ValueError("reference graph language differs from the frozen case")
    actual_gold_hash = hashlib.sha256(serialize_ast_graph(reference_graph).encode("utf-8")).hexdigest()
    if actual_gold_hash != case.get("gold_graph_sha256"):
        raise ValueError("reference graph differs from the frozen legacy gold")
    prediction = adapt_new_ir_prediction(serialized_prediction, case_id=case_id, manifest=manifest)
    metrics = ast_metrics(prediction, reference_graph)
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("legacy evaluator thresholds are missing")
    checks = {
        "parser_valid": metrics["parse_validity"] is True,
        "node_type_f1": metrics["node_type_f1"] >= thresholds["minimum_node_type_f1"],
        "typed_edge_f1": metrics["typed_edge_f1"] >= thresholds["minimum_typed_edge_f1"],
        "scope_relation_f1": metrics["scope_relation_f1"] >= thresholds["minimum_scope_relation_f1"],
        "lexical_f1": metrics["lexical_f1"] >= thresholds["minimum_lexical_f1"],
        "normalized_tree_edit_distance": (
            metrics["normalized_tree_edit_distance"]
            <= thresholds["maximum_normalized_tree_edit_distance"]
        ),
    }
    return {
        "case_id": case_id,
        "status": "pass" if all(checks.values()) else "fail",
        "strict_passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "gold_graph_sha256": actual_gold_hash,
        "prediction_graph_sha256": hashlib.sha256(serialized_prediction.encode("utf-8")).hexdigest(),
    }


__all__ = ["adapt_new_ir_prediction", "evaluate_legacy60_prediction"]
