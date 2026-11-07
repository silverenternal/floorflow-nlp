from __future__ import annotations

import torch

from newera.nlp_flow.components.research.architecture.experts.fixed_moe import FixedExpertPoolMoE
from newera.nlp_flow.model import ContextConditionedExpertFactory


def test_fixed_pool_and_generated_factory_are_context_sensitive() -> None:
    torch.manual_seed(31)
    fixed = FixedExpertPoolMoE(32, expert_count=4, rank=8)
    generated = ContextConditionedExpertFactory(32, max_rank=8, basis_count=4)
    with torch.no_grad():
        fixed.birth_gate.fill_(0.2)
        generated.birth_gate.fill_(0.2)
    parent = torch.randn(2, 5, 32)
    context = torch.randn(2, 32)

    fixed_output = fixed(parent, context)
    generated_output = generated(parent, context)

    assert not torch.allclose(fixed_output.expert_weights[0], fixed_output.expert_weights[1])
    generated_norm = generated_output.right.flatten(1).norm(dim=1)
    assert not torch.allclose(generated_norm[0], generated_norm[1])
    assert fixed_output.hidden_states.shape == generated_output.output.shape


def test_fixed_pool_has_no_factory_or_growth_interface() -> None:
    fixed = FixedExpertPoolMoE(16)

    assert not hasattr(fixed, "synthesize")
    assert not hasattr(fixed, "birth")
    assert fixed.left_basis.shape[0] == fixed.expert_count
    assert fixed.right_basis.shape[0] == fixed.expert_count
