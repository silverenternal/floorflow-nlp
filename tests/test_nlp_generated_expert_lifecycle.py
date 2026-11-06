from __future__ import annotations

import pytest
import torch

from floorflow.nlp_flow.generated_expert_contract import (
    GeneratedExpertEnvelope,
    GeneratedExpertLifecycle,
    validate_generated_expert,
)
from floorflow.nlp_flow.model.expert_factory import ContextConditionedExpertFactory
from floorflow.nlp_flow.model.runtime_competence import (
    RuntimeCompetenceConfig,
    RuntimeCompetenceManager,
)
from floorflow.nlp_flow.verifier_result_contract import (
    VerificationEvidence,
    VerificationIdentity,
    VerificationStatus,
)


def _lifecycle(capacity: int = 2):
    torch.manual_seed(2)
    factory = ContextConditionedExpertFactory(8, max_rank=16, basis_count=2, context_dim=8)
    manager = RuntimeCompetenceManager(
        factory,
        RuntimeCompetenceConfig(
            hidden_size=8,
            capacity=capacity,
            trigger_threshold=0,
            promotion_trials=2,
            promotion_success_rate=0.5,
        ),
    )
    envelope = GeneratedExpertEnvelope(
        max_rank=16,
        max_width=8,
        max_basis_parameters=512,
        max_materialized_parameters=512,
        max_expert_norm=100,
    )
    return GeneratedExpertLifecycle(factory, manager, envelope)


def _evidence(record, *, valid: bool) -> VerificationEvidence:
    digest = "a" * 64
    identity = VerificationIdentity(
        case_id=record.gap_key,
        case_sha256=digest,
        candidate_sha256=record.signature,
        verifier_sha256="b" * 64,
        environment_sha256="c" * 64,
    )
    return VerificationEvidence(
        identity,
        "tests",
        VerificationStatus.TESTS_PASSED if valid else VerificationStatus.FAILED,
        exit_code=0 if valid else 1,
    )


def test_generated_e_blueprint_is_bounded_zero_gated_then_verifier_promoted() -> None:
    lifecycle = _lifecycle()
    pooled = torch.randn(1, 8)
    record = lifecycle.create_candidate(pooled, torch.ones(1, 5), gap_key="gap-1")
    assert record is not None and record.status == "candidate"
    hidden = torch.randn(1, 3, 8)
    assert torch.count_nonzero(lifecycle.apply_promoted(hidden, record)) == 0

    lifecycle.observe(record, _evidence(record, valid=True), utility=1.0)
    assert record.status == "candidate"
    lifecycle.observe(record, _evidence(record, valid=True), utility=1.0)
    assert record.status == "promoted"
    assert torch.isfinite(lifecycle.apply_promoted(hidden, record)).all()
    lifecycle.retire(record)
    assert record.status == "retired"
    assert torch.count_nonzero(lifecycle.apply_promoted(hidden, record)) == 0


def test_generated_e_rejects_invalid_blueprint_evidence_and_unbounded_basis() -> None:
    lifecycle = _lifecycle()
    materialized = lifecycle.factory.materialize(torch.randn(1, 8))
    with pytest.raises(ValueError, match="basis exceeds"):
        validate_generated_expert(
            lifecycle.factory,
            materialized,
            GeneratedExpertEnvelope(16, 8, 1, 512, 100),
        )
    small_factory = ContextConditionedExpertFactory(8, max_rank=4, basis_count=2, context_dim=8)
    with torch.no_grad():
        small_factory.blueprint.weight.zero_()
        small_factory.blueprint.bias.zero_()
        small_factory.blueprint.bias[3] = 100
    with pytest.raises(ValueError, match="rank is outside"):
        validate_generated_expert(
            small_factory,
            small_factory.materialize(torch.randn(1, 8)),
            GeneratedExpertEnvelope(16, 8, 512, 512, 100),
        )
    record = lifecycle.create_candidate(torch.randn(1, 8), torch.ones(1, 5), gap_key="gap-1")
    assert record is not None
    wrong = VerificationEvidence(
        VerificationIdentity("elsewhere", "a" * 64, record.signature, "b" * 64, "c" * 64),
        "tests",
        VerificationStatus.TESTS_PASSED,
        exit_code=0,
    )
    with pytest.raises(ValueError, match="another runtime gap"):
        lifecycle.observe(record, wrong, utility=1)


def test_runtime_store_save_restore_preserves_lifecycle_accounting(tmp_path) -> None:
    first = _lifecycle()
    record = first.create_candidate(torch.randn(1, 8), torch.ones(1, 5), gap_key="gap-1")
    assert record is not None
    first.observe(record, _evidence(record, valid=True), utility=1.0)
    checkpoint = tmp_path / "runtime.pt"
    first.save(str(checkpoint))

    restored = _lifecycle()
    restored.load(checkpoint)
    counts = restored.runtime.metrics()
    assert counts == {"candidate": 1, "promoted": 0, "retired": 0}
    assert restored.runtime.records[record.signature].successes == 1


def test_failed_verifier_cannot_receive_utility_or_promote() -> None:
    lifecycle = _lifecycle()
    record = lifecycle.create_candidate(torch.randn(1, 8), torch.ones(1, 5), gap_key="gap-1")
    assert record is not None
    with pytest.raises(ValueError, match="cannot receive positive utility"):
        lifecycle.observe(record, _evidence(record, valid=False), utility=1)


def test_runtime_store_capacity_evicts_without_exceeding_its_hard_limit() -> None:
    lifecycle = _lifecycle(capacity=2)
    for index in range(4):
        lifecycle.create_candidate(
            torch.randn(1, 8), torch.ones(1, 5), gap_key=f"gap-{index}"
        )
    assert len(lifecycle.runtime.records) <= 2
