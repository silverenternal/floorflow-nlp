from __future__ import annotations

import torch

from newera.nlp_flow.model.components.experts.banks.expert_factory import (
    ContextConditionedExpertFactory,
)
from newera.nlp_flow.model.components.experts.competence.runtime_competence import (
    RuntimeCompetenceConfig,
    RuntimeCompetenceManager,
)


def _manager(capacity: int = 4) -> RuntimeCompetenceManager:
    factory = ContextConditionedExpertFactory(32, max_rank=8)
    with torch.no_grad():
        factory.birth_gate.fill_(1.0)
    return RuntimeCompetenceManager(
        factory,
        RuntimeCompetenceConfig(
            hidden_size=32,
            capacity=capacity,
            trigger_threshold=0.0,
            promotion_trials=2,
            promotion_success_rate=0.75,
        ),
    )


def _evidence(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[0.8, 0.5, 0.2, 0.9, 0.1]]).expand(batch, -1)


def test_runtime_expert_lifecycle_promotes_only_after_verified_reuse() -> None:
    manager = _manager()
    context = torch.randn(1, 32)
    record = manager.synthesize(
        context,
        context,
        _evidence(),
        gap_key="python-execution",
    )
    assert record is not None
    assert record.status == "candidate"
    assert manager.reusable("python-execution") is None
    output = manager.apply(torch.randn(1, 5, 32), record)
    assert torch.isfinite(output).all()

    assert manager.observe(record, valid=True, utility=0.4) == "candidate"
    assert manager.observe(record, valid=True, utility=0.6) == "promoted"
    assert manager.reusable("python-execution") is record
    assert manager.metrics()["promoted"] == 1


def test_runtime_expert_is_retired_after_repeated_invalid_results() -> None:
    manager = _manager()
    context = torch.randn(1, 32)
    record = manager.synthesize(
        context,
        context,
        _evidence(),
        gap_key="rust-borrowing",
    )
    assert record is not None

    assert manager.observe(record, valid=False, utility=0.0) == "candidate"
    assert manager.observe(record, valid=False, utility=0.0) == "retired"
    assert manager.reusable("rust-borrowing") is None


def test_generated_experts_are_context_dependent_and_store_bounded() -> None:
    manager = _manager(capacity=2)
    signatures = set()
    for index in range(5):
        context = torch.randn(1, 32)
        record = manager.synthesize(
            context,
            context,
            _evidence(),
            gap_key=f"gap-{index}",
        )
        assert record is not None
        signatures.add(record.signature)

    assert len(signatures) > 1
    assert len(manager.records) <= 2
    assert record.novelty_distance >= 0.0


def test_promoted_runtime_expert_store_round_trips(tmp_path) -> None:
    manager = _manager()
    context = torch.randn(1, 32)
    record = manager.synthesize(
        context,
        context,
        _evidence(),
        gap_key="cpp-template",
    )
    assert record is not None
    manager.observe(record, valid=True, utility=0.4)
    manager.observe(record, valid=True, utility=0.6)
    path = tmp_path / "runtime-store.pt"
    manager.save(path)

    restored = _manager()
    restored.load(path)

    assert restored.metrics()["promoted"] == 1
    reusable = restored.reusable("cpp-template")
    assert reusable is not None
    assert reusable.signature == record.signature
