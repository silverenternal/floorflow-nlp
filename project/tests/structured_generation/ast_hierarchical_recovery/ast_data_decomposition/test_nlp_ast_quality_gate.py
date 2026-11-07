from __future__ import annotations

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast
from newera.nlp_flow.components.ast.diagnostics.validation.ast_quality_gate import (
    ASTHandoffEvidence,
    evaluate_ast_upstream_source,
)


def test_perfect_token_endpoint_passes_strict_ast_gate() -> None:
    source = "def add(a, b):\n    return a + b\n"
    reference = parse_ast(source, "python")

    report = evaluate_ast_upstream_source(
        source,
        reference,
        "python",
        source_kind="token_endpoint",
    )

    assert report["status"] == "pass"
    assert all(report["checks"].values())


def test_grammar_template_is_rejected_even_with_perfect_source() -> None:
    source = "def add(a, b):\n    return a + b\n"
    reference = parse_ast(source, "python")

    report = evaluate_ast_upstream_source(
        source,
        reference,
        "python",
        source_kind="grammar_template",
    )

    assert report["status"] == "fail"
    assert not report["parse_validity"]


def test_structural_mismatch_fails_upstream_gate() -> None:
    reference = parse_ast(
        "def add(a, b):\n    return a + b\n",
        "python",
    )

    report = evaluate_ast_upstream_source(
        "def add(a, b):\n    return a - b\n",
        reference,
        "python",
        source_kind="token_endpoint",
    )

    assert report["status"] == "fail"


def test_runtime_structural_evidence_is_separate_from_reference_quality() -> None:
    runtime = ASTHandoffEvidence.runtime_structural(
        parse_validity=True,
        single_root=True,
        nonempty=True,
        within_node_budget=True,
        confidence=0.8,
        minimum_confidence=0.5,
    )

    runtime.require_stage_b_ready()
    assert not runtime.training_ready
    try:
        runtime.require_stage_b_ready(require_reference_metrics=True)
    except ValueError as error:
        assert "token-endpoint" in str(error)
    else:
        raise AssertionError("runtime evidence unlocked a reference-quality gate")
