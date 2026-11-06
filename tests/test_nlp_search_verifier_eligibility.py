from __future__ import annotations

from floorflow.nlp_flow.search_eligibility import assess_candidate, select_pareto_candidates


def _candidate(candidate_id: str, **changes):
    row = {
        "candidate_id": candidate_id,
        "checkpoint_id": candidate_id,
        "split": "validation",
        "provenance": "external_verified_validation",
        "verifier_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
        "identity_sha256": "c" * 64,
        "family": "natural_task_executable_success",
        "support_count": 40,
        "case_count": 40,
        "execution_verified": True,
        "metrics": {
            "executable_success": 0.7,
            "ast_validity": 1.0,
            "native_validity": 1.0,
            "coverage": 1.0,
        },
        "token_exposure": 1000,
        "resident_parameters": 1_000_000,
        "total_compute": 2_000_000,
    }
    return {**row, **changes}


def test_loss_only_test_fixture_zero_support_and_missing_verifier_are_ineligible() -> None:
    candidates = [
        _candidate("loss", metrics={"validation_loss": 0.1}),
        _candidate("test", split="test"),
        _candidate("fixture", provenance="fixture"),
        _candidate("zero", support_count=0),
        _candidate("no-verifier", verifier_sha256=None),
    ]
    result = select_pareto_candidates(candidates)
    assert result["status"] == "no_winner_ineligible"
    assert result["winner"] is None
    assert all(not decision.eligible for decision in result["decisions"])


def test_constant_labels_ties_and_pareto_dominance_are_explicit() -> None:
    constant = select_pareto_candidates([_candidate(f"constant-{i}") for i in range(5)])
    assert constant["status"] == "no_winner_constant_labels"
    assert constant["winner"] is None

    rows = [
        _candidate(
            f"dominated-{i}",
            metrics={
                "executable_success": 0.6,
                "ast_validity": 1.0,
                "native_validity": 1.0,
                "coverage": 0.9,
            },
        )
        for i in range(4)
    ]
    rows.append(
        _candidate(
            "winner",
            metrics={
                "executable_success": 0.8,
                "ast_validity": 1.0,
                "native_validity": 1.0,
                "coverage": 1.0,
            },
            token_exposure=900,
            resident_parameters=900_000,
            total_compute=1_900_000,
        )
    )
    selected = select_pareto_candidates(rows)
    assert selected["status"] == "winner_selected"
    assert selected["winner"] == "winner"


def test_missing_execution_verifier_and_oracle_test_labels_are_rejected() -> None:
    missing_execution = _candidate("missing", execution_verified=False)
    assert "missing_execution_verification" in assess_candidate(missing_execution).reasons
    oracle = _candidate("oracle", oracle_only=True)
    assert "forbidden_evidence_origin" in assess_candidate(oracle).reasons
    test_label = _candidate("test-label", test_split_label=True)
    assert "forbidden_evidence_origin" in assess_candidate(test_label).reasons
