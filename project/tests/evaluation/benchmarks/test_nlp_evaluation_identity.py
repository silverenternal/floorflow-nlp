from __future__ import annotations

import pytest

from newera.nlp_flow.components.research.evaluation.evidence.evaluation_identity import (
    EvaluationIdentity,
    strict_join_labels,
)


def _row(**changes):
    row = {
        "model_hash": "a" * 64,
        "config_hash": "b" * 64,
        "corpus_hash": "c" * 64,
        "split": "validation",
        "task_id": "compile_python",
        "case_id": "case-1",
        "seed": 7,
        "parser_hash": "d" * 64,
        "evaluator_hash": "e" * 64,
    }
    return {**row, **changes}


def test_evaluation_identity_and_exact_one_to_one_label_join() -> None:
    prediction = {**_row(), "prediction": 1}
    label = {**_row(), "pass": 1, "provenance": "verifier", "evidence_sha256": "f" * 64}
    joined = strict_join_labels([prediction], [label], label_field="pass")
    assert joined[0]["label"] == 1
    assert len(joined[0]["identity_sha256"]) == 64
    assert EvaluationIdentity.from_mapping(prediction).digest == joined[0]["identity_sha256"]


def test_identity_join_rejects_every_stale_or_ambiguous_link() -> None:
    prediction = {**_row(), "prediction": 1}
    label = {**_row(), "pass": 1, "provenance": "verifier", "evidence_sha256": "f" * 64}
    with pytest.raises(ValueError, match="missing exact identity"):
        strict_join_labels([prediction], [{**label, "model_hash": "9" * 64}], label_field="pass")
    with pytest.raises(ValueError, match="split is not selectable"):
        strict_join_labels([prediction], [{**label, "split": "test"}], label_field="pass")
    with pytest.raises(ValueError, match="duplicate label identity"):
        strict_join_labels([prediction], [label, label], label_field="pass")
    with pytest.raises(ValueError, match="stale or unproven"):
        strict_join_labels([prediction], [{**label, "stale": True}], label_field="pass")
    with pytest.raises(ValueError, match="unmatched identities"):
        strict_join_labels([prediction], [label, {**label, "case_id": "extra"}], label_field="pass")


def test_fixture_and_training_rows_cannot_masquerade_as_validation_labels() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation split"):
        EvaluationIdentity.from_mapping(_row(split="training"))
    prediction = {**_row(), "prediction": 1}
    label = {**_row(split="fixture"), "pass": 1, "provenance": "verifier", "evidence_sha256": "f" * 64}
    with pytest.raises(ValueError, match="split is not selectable"):
        strict_join_labels([prediction], [label], label_field="pass")
