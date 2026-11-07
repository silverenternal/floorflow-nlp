import hashlib
from pathlib import Path

import pytest

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph, parser_fingerprint
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.research.evaluation.manifests.legacy60_adapter import (
    adapt_new_ir_prediction,
    evaluate_legacy60_prediction,
)

ROOT = Path(__file__).resolve().parents[5]


def _manifest(reference) -> dict:
    bindings = {}
    for language in ("python", "rust", "cpp"):
        fingerprint = parser_fingerprint(language)
        bindings[language] = {
            "package": fingerprint.package.replace("_", "-"),
            "version": fingerprint.package_version,
            "abi_version": fingerprint.abi_version,
            "semantic_version": (
                None if fingerprint.semantic_version is None else list(fingerprint.semantic_version)
            ),
            "native_sha256": fingerprint.native_sha256,
            "grammar_sha256": fingerprint.grammar_sha256,
        }
    gold_hash = hashlib.sha256(serialize_ast_graph(reference).encode()).hexdigest()
    cases = [
        {"case_id": f"case-{index:02d}", "language": "python", "gold_graph_sha256": gold_hash}
        for index in range(60)
    ]
    code_paths = {
        "canonicalizer": (
            "src/newera/nlp_flow/components/ast/representation/codec/ast_codec.py",
            "src/newera/nlp_flow/components/ast/representation/codec/ast_codec_io.py",
            "src/newera/nlp_flow/components/ast/representation/codec/ast_codec_parser.py",
        ),
        "metrics": (
            "src/newera/nlp_flow/components/ast/representation/codec/ast_codec_metrics.py",
            "src/newera/nlp_flow/components/ast/diagnostics/validation/ast_quality_gate.py",
        ),
        "legacy_adapter": ("src/newera/nlp_flow/components/research/evaluation/manifests/legacy60_adapter.py",),
    }
    code_hashes = {
        group: {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths
        }
        for group, paths in code_paths.items()
    }
    return {
        "version": "nlp-legacy60-evaluator-v11",
        "ordered_case_ids": [case["case_id"] for case in cases],
        "cases": cases,
        "parser_fingerprints": bindings,
        "implementation_sha256": code_hashes,
        "thresholds": {
            "minimum_node_type_f1": 0.95,
            "minimum_typed_edge_f1": 0.95,
            "minimum_scope_relation_f1": 0.90,
            "minimum_lexical_f1": 0.95,
            "maximum_normalized_tree_edit_distance": 0.05,
        },
    }


def test_new_ir_graph_uses_original_legacy_metrics_and_thresholds() -> None:
    source = "def answer(x):\n    return x + 1\n"
    reference = parse_ast_graph(source, "python")
    manifest = _manifest(reference)
    prediction = serialize_ast_graph(reference)
    adapted = adapt_new_ir_prediction(prediction, case_id="case-00", manifest=manifest)
    result = evaluate_legacy60_prediction(
        prediction,
        case_id="case-00",
        reference_graph=reference,
        manifest=manifest,
    )
    assert adapted == reference
    assert result["status"] == "pass"
    assert result["strict_passed"] is True
    assert all(result["checks"].values())


def test_changed_prediction_is_measured_and_not_repaired() -> None:
    reference = parse_ast_graph("def answer(x):\n    return x + 1\n", "python")
    changed = parse_ast_graph("def answer(x):\n    return x - 1\n", "python")
    result = evaluate_legacy60_prediction(
        serialize_ast_graph(changed),
        case_id="case-00",
        reference_graph=reference,
        manifest=_manifest(reference),
    )
    assert result["status"] == "fail"
    assert result["checks"]["parser_valid"] is True
    assert result["checks"]["lexical_f1"] is False
    assert result["checks"]["normalized_tree_edit_distance"] is False


def test_unknown_case_wrong_language_and_changed_gold_are_rejected() -> None:
    reference = parse_ast_graph("def answer():\n    return 1\n", "python")
    manifest = _manifest(reference)
    serialized = serialize_ast_graph(reference)
    with pytest.raises(ValueError, match="not in the frozen"):
        adapt_new_ir_prediction(serialized, case_id="other", manifest=manifest)
    manifest["cases"][0]["language"] = "rust"
    with pytest.raises(ValueError, match="prediction language"):
        adapt_new_ir_prediction(serialized, case_id="case-00", manifest=manifest)
    manifest = _manifest(reference)
    changed_gold = parse_ast_graph("def answer():\n    return 2\n", "python")
    with pytest.raises(ValueError, match="frozen legacy gold"):
        evaluate_legacy60_prediction(
            serialized,
            case_id="case-00",
            reference_graph=changed_gold,
            manifest=manifest,
        )


def test_manifest_case_order_and_case_count_are_integrity_constraints() -> None:
    reference = parse_ast_graph("pass\n", "python")
    manifest = _manifest(reference)
    manifest["ordered_case_ids"][0], manifest["ordered_case_ids"][1] = (
        manifest["ordered_case_ids"][1],
        manifest["ordered_case_ids"][0],
    )
    with pytest.raises(ValueError, match="order disagrees"):
        adapt_new_ir_prediction(serialize_ast_graph(reference), case_id="case-00", manifest=manifest)
    manifest = _manifest(reference)
    manifest["cases"].pop()
    with pytest.raises(ValueError, match="exactly 60"):
        adapt_new_ir_prediction(serialize_ast_graph(reference), case_id="case-00", manifest=manifest)


def test_changed_metric_implementation_invalidates_legacy_adapter() -> None:
    reference = parse_ast_graph("pass\n", "python")
    manifest = _manifest(reference)
    manifest["implementation_sha256"]["metrics"]["src/newera/nlp_flow/components/ast/diagnostics/validation/ast_quality_gate.py"] = "0" * 64
    with pytest.raises(ValueError, match="implementation changed"):
        adapt_new_ir_prediction(serialize_ast_graph(reference), case_id="case-00", manifest=manifest)
