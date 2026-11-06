from __future__ import annotations

import torch

from floorflow.nlp_flow.model.expert_bank import (
    ExpertBankConfig,
    ExpertResidualBank,
)


def test_generated_and_fixed_experts_start_zero_gated() -> None:
    hidden = torch.randn(2, 5, 32)
    for mode in ("generated", "fixed_pool"):
        bank = ExpertResidualBank(
            ExpertBankConfig(
                hidden_size=32,
                mode=mode,
                expert_count=4,
                top_k=2,
                max_rank=8,
                basis_count=3,
                context_dim=16,
            )
        ).eval()
        with torch.no_grad():
            output = bank(hidden)
        assert torch.equal(output.residual, torch.zeros_like(hidden))


def test_generated_and_fixed_bank_gradients_are_reachable() -> None:
    hidden = torch.randn(2, 5, 32)
    generated = ExpertResidualBank(
        ExpertBankConfig(hidden_size=32, mode="generated", max_rank=8)
    )
    with torch.no_grad():
        generated.factory.birth_gate.fill_(0.5)
    generated(hidden).residual.square().mean().backward()
    assert generated.factory.birth_gate.grad is not None

    fixed = ExpertResidualBank(
        ExpertBankConfig(
            hidden_size=32,
            mode="fixed_pool",
            expert_count=4,
            top_k=2,
            max_rank=8,
        )
    )
    with torch.no_grad():
        fixed.fixed_pool.gate.fill_(0.5)
    fixed(hidden).residual.square().mean().backward()
    assert fixed.fixed_pool.left.grad is not None
    assert fixed.fixed_pool.right.grad is not None
    assert fixed.fixed_pool.router.weight.grad is not None


def test_fixed_pool_routing_and_freeze_diagnostics() -> None:
    bank = ExpertResidualBank(
        ExpertBankConfig(
            hidden_size=32,
            mode="fixed_pool",
            expert_count=5,
            top_k=2,
            max_rank=8,
        )
    )
    output = bank(torch.randn(2, 7, 32), torch.ones(2, 7, dtype=torch.bool))

    assert output.router_weights is not None
    assert torch.allclose(
        output.router_weights.sum(dim=-1),
        torch.ones(2, 7),
    )
    assert "expert_balance_loss" in output.diagnostics
    bank.freeze_growth()
    assert all(not parameter.requires_grad for parameter in bank.parameters())


def test_random_frozen_birth_control_is_nonzero_and_frozen() -> None:
    bank = ExpertResidualBank(
        ExpertBankConfig(
            hidden_size=32,
            mode="random_frozen",
            max_rank=8,
        )
    ).eval()
    output = bank(torch.randn(2, 5, 32))

    assert output.residual.abs().sum() > 0.0
    assert output.diagnostics["expert_mode"] == "random_frozen"
    assert all(not parameter.requires_grad for parameter in bank.factory.parameters())
