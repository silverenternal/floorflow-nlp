from __future__ import annotations

import pytest

from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import (
    LexicalDecision,
    LexicalEvidence,
    LexicalNode,
    TypedLexicalContract,
)


def _contract() -> TypedLexicalContract:
    return TypedLexicalContract(
        nodes=(
            LexicalNode(12, "identifier", 2, ("identifier",), 1, 128),
            LexicalNode(13, "string", 2, ("string_literal",), 0, 256),
        ),
        scope_parents=((0, None), (1, 0), (2, 1), (3, 0)),
        evidence=(
            LexicalEvidence("hidden", "training_only", "training_name", False, None),
            LexicalEvidence("sibling", "user_available", "external_name", True, 3),
            LexicalEvidence("target", "target_derived", "private_gold", False, 1),
            LexicalEvidence("user", "user_available", "name = π🙂", True, 1),
        ),
    )


def test_visible_copy_uses_unicode_codepoint_span_and_variable_bounds() -> None:
    value = _contract().resolve(
        LexicalDecision(
            node_id=12,
            node_kind="identifier",
            leaf_class="identifier",
            mode="copy",
            minimum_length=2,
            maximum_length=4,
            evidence_id="user",
            span_start=7,
            span_end=9,
        )
    )
    assert value.value == "π🙂"
    assert value.origin == "user_available"
    assert value.span_start == 7 and value.span_end == 9
    assert value.minimum_length == 2 and value.maximum_length == 4


def test_generated_lexical_value_has_no_copy_evidence() -> None:
    value = _contract().resolve(
        LexicalDecision(
            node_id=13,
            node_kind="string",
            leaf_class="string_literal",
            mode="generate",
            minimum_length=1,
            maximum_length=8,
            value="λ🙂",
        )
    )
    assert value.origin == "generated"
    assert value.value == "λ🙂"
    assert value.evidence_id is None and value.span_start is None


@pytest.mark.parametrize("evidence_id", ["target", "hidden", "sibling"])
def test_target_only_hidden_or_out_of_scope_copy_is_rejected(evidence_id: str) -> None:
    with pytest.raises(ValueError, match="user-visible|outside.*scope"):
        _contract().resolve(
            LexicalDecision(
                node_id=12,
                node_kind="identifier",
                leaf_class="identifier",
                mode="copy",
                minimum_length=1,
                maximum_length=32,
                evidence_id=evidence_id,
                span_start=0,
                span_end=3,
            )
        )


def test_copy_rejects_wrong_type_length_bounds_and_invalid_spans() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="type or class"):
        contract.resolve(
            LexicalDecision(12, "string", "identifier", "copy", 1, 20, evidence_id="user", span_start=0, span_end=2)
        )
    with pytest.raises(ValueError, match="variable-length bounds"):
        contract.resolve(
            LexicalDecision(12, "identifier", "identifier", "copy", 3, 10, evidence_id="user", span_start=7, span_end=9)
        )
    with pytest.raises(ValueError, match="outside the evidence"):
        contract.resolve(
            LexicalDecision(12, "identifier", "identifier", "copy", 1, 10, evidence_id="user", span_start=8, span_end=20)
        )


def test_generated_values_cannot_claim_source_spans() -> None:
    with pytest.raises(ValueError, match="cannot reference source evidence"):
        _contract().resolve(
            LexicalDecision(
                12,
                "identifier",
                "identifier",
                "generate",
                1,
                20,
                value="value",
                evidence_id="user",
                span_start=0,
                span_end=5,
            )
        )
