from __future__ import annotations

import pytest

from floorflow.nlp_flow.capability_evidence import (
    CapabilityEvidence,
    ClaimScope,
    EvidenceOrigin,
)


def _evidence(origin: EvidenceOrigin, scope: ClaimScope, **kwargs):
    return CapabilityEvidence(
        origin,
        scope,
        kwargs.pop("split", "validation"),
        kwargs.pop("case_count", 30),
        kwargs.pop("evidence_sha256", "a" * 64),
        kwargs.pop("verifier_sha256", None),
        kwargs.pop("independent_audit_sha256", None),
        kwargs.pop("baseline_audit_sha256", None),
    )


@pytest.mark.parametrize(
    "origin",
    [EvidenceOrigin.IMPLEMENTATION, EvidenceOrigin.FIXTURE, EvidenceOrigin.TEACHER_REPLAY,
     EvidenceOrigin.ORACLE, EvidenceOrigin.INTERNAL_RECOVERY],
)
def test_non_external_evidence_cannot_fill_user_task_or_sota_claims(origin) -> None:
    user = _evidence(origin, ClaimScope.EXTERNAL_TASK_SUCCESS, verifier_sha256="b" * 64,
                     split="test", independent_audit_sha256="c" * 64)
    sota = _evidence(origin, ClaimScope.SOTA, verifier_sha256="b" * 64,
                     split="test", independent_audit_sha256="c" * 64,
                     baseline_audit_sha256="d" * 64)
    assert not user.claimable
    assert not sota.claimable
    with pytest.raises(ValueError):
        user.require_claimable()


def test_internal_recovery_claim_is_separate_from_user_task_success() -> None:
    recovery = _evidence(EvidenceOrigin.INTERNAL_RECOVERY, ClaimScope.INTERNAL_RECONSTRUCTION)
    assert recovery.claimable
    assert not _evidence(EvidenceOrigin.INTERNAL_RECOVERY, ClaimScope.EXTERNAL_TASK_SUCCESS).claimable


def test_external_claim_requires_independent_audit_and_sota_baseline_audit() -> None:
    user = _evidence(EvidenceOrigin.USER_TASK_VERIFIED, ClaimScope.EXTERNAL_TASK_SUCCESS,
                     split="test", verifier_sha256="b" * 64,
                     independent_audit_sha256="c" * 64)
    assert user.claimable
    sota_without_baseline = _evidence(EvidenceOrigin.USER_TASK_VERIFIED, ClaimScope.SOTA,
                                      split="test", verifier_sha256="b" * 64,
                                      independent_audit_sha256="c" * 64)
    assert not sota_without_baseline.claimable
    sota = _evidence(EvidenceOrigin.USER_TASK_VERIFIED, ClaimScope.SOTA, split="test",
                     verifier_sha256="b" * 64, independent_audit_sha256="c" * 64,
                     baseline_audit_sha256="d" * 64)
    assert sota.claimable
