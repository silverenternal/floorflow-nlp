from __future__ import annotations

import pytest

from floorflow.nlp_flow.proxy_alignment import align_proxy_metrics


def _rows(count: int = 30):
    return [
        {
            "split": "validation",
            "family": "compile",
            "checkpoint_id": f"checkpoint-{checkpoint}",
            "case_id": f"case-{case}",
            "proxy_value": checkpoint * 100 + case,
            "external_value": checkpoint * 100 + case,
            "evidence_sha256": "a" * 64,
        }
        for checkpoint in range(5)
        for case in range(count)
    ]


def test_proxy_alignment_requires_checkpoint_and_case_coverage() -> None:
    result = align_proxy_metrics(_rows(3), bootstrap_resamples=20)
    assert result["families"]["compile"]["status"] == "unqualified_insufficient_support"
    assert not result["families"]["compile"]["qualified"]


def test_proxy_alignment_reports_rho_sign_agreement_and_bootstrap_intervals() -> None:
    result = align_proxy_metrics(_rows(), bootstrap_resamples=40, seed=9)
    family = result["families"]["compile"]
    assert family["qualified"]
    assert family["rho"] == 1.0
    assert family["sign_agreement"] == 1.0
    assert len(family["rho_interval"]) == 2
    assert family["label_provenance"] == "joined_validation_verifier_evidence"


@pytest.mark.parametrize("change", [{"split": "test"}, {"oracle_only": True}, {"fixture_execution": True}])
def test_proxy_alignment_rejects_test_or_oracle_fixture_rows(change) -> None:
    row = {**_rows(1)[0], **change}
    with pytest.raises(ValueError):
        align_proxy_metrics([row])


def test_proxy_alignment_rejects_duplicate_case_checkpoint_pairs() -> None:
    rows = _rows(1)
    with pytest.raises(ValueError, match="duplicate"):
        align_proxy_metrics([rows[0], rows[0]])
