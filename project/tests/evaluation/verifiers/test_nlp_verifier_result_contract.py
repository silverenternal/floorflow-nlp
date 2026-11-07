from __future__ import annotations

import pytest

from newera.nlp_flow.components.research.evaluation.evidence.verifier_result_contract import (
    PatchApplicationReceipt,
    VerificationEvidence,
    VerificationIdentity,
    classify_command_result,
)
from newera.nlp_flow.components.research.evaluation.evidence.verifier_result_contract import (
    VerificationStatus as Status,
)


def _identity() -> VerificationIdentity:
    digest = "a" * 64
    return VerificationIdentity("case-1", digest, "b" * 64, "c" * 64, "d" * 64)


def test_command_outcomes_keep_failure_modes_distinct_and_hash_bound() -> None:
    identity = _identity()
    timeout = classify_command_result(
        identity, stage="tests", returncode=-1, timed_out=True,
        environment_available=True,
    )
    missing = classify_command_result(
        identity, stage="compile", returncode=-1, timed_out=False,
        environment_available=False,
    )
    crash = classify_command_result(
        identity, stage="tests", returncode=-11, timed_out=False,
        environment_available=True,
    )
    failed = classify_command_result(
        identity, stage="tests", returncode=1, timed_out=False,
        environment_available=True,
    )
    assert {timeout.status, missing.status, crash.status, failed.status} == {
        Status.TIMED_OUT, Status.ENVIRONMENT_MISSING, Status.CRASHED, Status.FAILED,
    }
    assert not any(item.passed for item in (timeout, missing, crash, failed))
    assert timeout.require_current(identity) is timeout
    with pytest.raises(ValueError, match="stale"):
        timeout.require_current(VerificationIdentity("other", "a" * 64, "b" * 64, "c" * 64, "d" * 64))


def test_wrong_output_and_numerical_success_require_digest_evidence() -> None:
    identity = _identity()
    with pytest.raises(ValueError, match="different output digests"):
        VerificationEvidence(
            identity, "numerical", Status.WRONG_OUTPUT,
            expected_output_sha256="a" * 64, observed_output_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="exact expected output digest parity"):
        VerificationEvidence(
            identity, "numerical", Status.NUMERICAL_PASSED,
            expected_output_sha256="a" * 64, observed_output_sha256="b" * 64,
        )
    passed = VerificationEvidence(
        identity, "numerical", Status.NUMERICAL_PASSED,
        expected_output_sha256="a" * 64, observed_output_sha256="a" * 64,
    )
    assert passed.passed and len(passed.evidence_sha256) == 64


def test_patch_receipt_binds_base_and_application_tree() -> None:
    receipt = PatchApplicationReceipt(
        _identity(), "revision-1", Status.PATCH_APPLIED, "e" * 64, "f" * 64,
    )
    assert receipt.passed
    with pytest.raises(ValueError, match="resulting tree digest"):
        PatchApplicationReceipt(_identity(), "revision-1", Status.PATCH_APPLIED, "e" * 64, None)


