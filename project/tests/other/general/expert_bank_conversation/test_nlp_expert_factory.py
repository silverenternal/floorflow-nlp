from __future__ import annotations

import torch

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model import ContextConditionedExpertFactory


def test_factory_is_zero_gated_and_context_dependent() -> None:
    factory = ContextConditionedExpertFactory(32, max_rank=8)
    parent = torch.randn(2, 5, 32)
    first = factory(parent, torch.randn(2, 32))
    second = factory(parent, torch.randn(2, 32))

    assert torch.equal(first.output, torch.zeros_like(first.output))
    assert not torch.allclose(first.left, second.left)
    assert not torch.allclose(first.right, second.right)


def test_factory_outputs_receive_gradients_after_birth() -> None:
    factory = ContextConditionedExpertFactory(32, max_rank=8)
    with torch.no_grad():
        factory.birth_gate.fill_(1.0)
    parent = torch.randn(2, 5, 32, requires_grad=True)
    context = torch.randn(2, 32, requires_grad=True)
    generated = factory(parent, context)

    generated.output.square().mean().backward()

    assert parent.grad is not None
    assert context.grad is not None
    assert factory.left_basis.grad is not None
    assert factory.right_basis.grad is not None


def test_factory_is_compact() -> None:
    factory = ContextConditionedExpertFactory(160, max_rank=16)

    assert count_unique_trainable_parameters(factory) < 50_000


def test_expert_weights_can_be_materialized_and_applied_without_factory_forward() -> None:
    factory = ContextConditionedExpertFactory(32, max_rank=8)
    with torch.no_grad():
        factory.birth_gate.fill_(1.0)
    parent = torch.randn(2, 5, 32)
    context = torch.randn(2, 32)

    dynamic = factory(parent, context)
    materialized = factory.materialize(context)
    static = factory.apply_materialized(parent, materialized)

    assert torch.equal(dynamic.output, static)
    assert materialized.left.requires_grad
    materialized_detached = type(materialized)(
        left=materialized.left.detach(),
        right=materialized.right.detach(),
        rank_mask=materialized.rank_mask,
        width_mask=materialized.width_mask,
        input_scale=materialized.input_scale.detach(),
        output_scale=materialized.output_scale.detach(),
        birth_gate=torch.ones_like(materialized.birth_gate),
        rank=materialized.rank,
        width=materialized.width,
    )
    assert torch.equal(
        factory.apply_materialized(parent, materialized_detached),
        static,
    )
