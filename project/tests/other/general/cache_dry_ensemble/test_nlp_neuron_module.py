from __future__ import annotations

import torch

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model import NeuronModule


def test_neuron_module_is_identity_at_birth() -> None:
    module = NeuronModule(32, 8)
    parent = torch.randn(2, 5, 32)

    output = module(parent)

    assert torch.equal(output.hidden_states, parent)
    assert torch.equal(output.residual, torch.zeros_like(parent))
    assert output.state.fast.shape == (2, 8)
    assert output.state.slow.shape == (2, 8)


def test_neuron_module_state_is_history_sensitive_and_trainable() -> None:
    module = NeuronModule(32, 8)
    with torch.no_grad():
        module.birth_gate.fill_(1.0)
    parent = torch.randn(2, 5, 32, requires_grad=True)
    first = module(parent)
    second = module(parent, first.state)

    loss = second.hidden_states.square().mean() + second.local_output.square().mean()
    loss.backward()

    assert not torch.allclose(first.residual, second.residual)
    assert parent.grad is not None
    assert module.fast.weight_ih.grad is not None
    assert module.slow.weight_ih.grad is not None


def test_neuron_module_is_compact() -> None:
    module = NeuronModule(160, 32)

    assert count_unique_trainable_parameters(module) < 100_000
