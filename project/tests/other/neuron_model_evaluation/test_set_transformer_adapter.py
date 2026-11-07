import pytest
import torch

from newera.neuron.cells.aggregation import (
    ContextualSumResidualDeepSetsAggregator,
    DeepSetsMessageAggregator,
    SetTransformerSetAggregator,
)
from newera.neuron.composition import ModelNeuron

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device_name", DEVICES)
def test_adapter_matches_upstream_and_backpropagates(device_name: str):
    torch.manual_seed(41)
    device = torch.device(device_name)
    adapter = SetTransformerSetAggregator(8, heads=2).to(device)
    messages = torch.randn(7, 8, device=device, requires_grad=True)
    actual = adapter(messages)
    expected = adapter.upstream(
        messages, index=torch.zeros(7, dtype=torch.long, device=device), dim_size=1
    )
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_adapter_is_permutation_invariant(device_name: str):
    device = torch.device(device_name)
    adapter = SetTransformerSetAggregator(8, heads=2).to(device).eval()
    messages = torch.randn(9, 8, device=device)
    order = torch.randperm(9, device=device)
    torch.testing.assert_close(adapter(messages), adapter(messages[order]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device_name", DEVICES)
def test_adapter_is_repeatable_in_evaluation_mode(device_name: str):
    device = torch.device(device_name)
    adapter = SetTransformerSetAggregator(8, heads=2).to(device).eval()
    messages = torch.randn(5, 8, device=device)
    torch.testing.assert_close(adapter(messages), adapter(messages), atol=0, rtol=0)


@pytest.mark.parametrize("device_name", DEVICES)
def test_empty_input_has_explicit_zero_identity(device_name: str):
    device = torch.device(device_name)
    adapter = SetTransformerSetAggregator(8, heads=2).to(device)
    output = adapter(torch.empty(0, 8, device=device))
    assert output.shape == (1, 8)
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("device_name", DEVICES)
def test_masked_padding_is_ignored_and_duplicate_contacts_are_preserved(device_name: str):
    device = torch.device(device_name)
    adapter = SetTransformerSetAggregator(8, heads=2).to(device).eval()
    messages = torch.randn(3, 8, device=device)
    padded = torch.cat((messages, torch.full((2, 8), 1e6, device=device)))
    mask = torch.tensor([True, True, True, False, False], device=device)
    torch.testing.assert_close(adapter(messages), adapter(padded, mask=mask))
    duplicate_contact = torch.cat((messages, messages[:1]))
    assert not torch.allclose(adapter(messages), adapter(duplicate_contact))


def test_rejects_bad_shape_mask_and_head_configuration():
    with pytest.raises(ValueError, match="divisible"):
        SetTransformerSetAggregator(7, heads=2)
    adapter = SetTransformerSetAggregator(8, heads=2)
    with pytest.raises(ValueError, match="mask"):
        adapter(torch.randn(3, 8), mask=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="shape"):
        adapter(torch.randn(3, 7))


@pytest.mark.parametrize("device_name", DEVICES)
def test_deep_sets_adapter_preserves_set_semantics_and_gradients(device_name: str):
    device = torch.device(device_name)
    adapter = DeepSetsMessageAggregator(8).to(device).eval()
    messages = torch.randn(3, 5, 8, device=device, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False], [False] * 5], device=device)
    output = adapter.forward_batched(messages, mask=mask)
    assert output.shape == (3, 8)
    assert torch.equal(output[2], torch.zeros_like(output[2]))
    torch.testing.assert_close(adapter(messages[0, :3]), adapter(messages[0, :3].flip(0)), atol=1e-6, rtol=1e-6)
    output.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in adapter.parameters())






@pytest.mark.parametrize("device_name", DEVICES)
def test_deep_sets_is_owned_by_model_neuron(device_name: str):
    device = torch.device(device_name)
    aggregator = DeepSetsMessageAggregator(8)
    neuron = ModelNeuron(input_size=3, hidden_size=8, message_size=8, message_aggregator=aggregator).to(device)
    output = neuron.step(
        torch.randn(2, 3, device=device),
        incoming_messages=torch.randn(2, 4, 8, device=device),
        incoming_strengths=torch.ones(2, 4, device=device),
        incoming_edge_types=torch.zeros(2, 4, dtype=torch.long, device=device),
    )
    output.value.sum().backward()
    assert output.value.shape == (2, 8)
    assert any(name.startswith("message_aggregator.") for name, _ in neuron.named_parameters())
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_contextual_deep_sets_starts_exactly_at_sum_and_backpropagates(device_name: str):
    device = torch.device(device_name)
    aggregator = ContextualSumResidualDeepSetsAggregator(8, context_size=3, expansion=8).to(device)
    messages = torch.randn(2, 5, 8, device=device, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, False, False, False, False]], device=device)
    context = torch.randn(2, 3, device=device)
    actual = aggregator.forward_batched_with_context(messages, mask=mask, context=context)
    expected = (messages * mask.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())






@pytest.mark.parametrize("device_name", DEVICES)
def test_aggregator_parameters_are_registered_on_one_model_neuron(device_name: str):
    device = torch.device(device_name)
    aggregator = SetTransformerSetAggregator(8, heads=2)
    neuron = ModelNeuron(
        input_size=3,
        hidden_size=8,
        message_size=8,
        message_aggregator=aggregator,
    ).to(device)
    messages = torch.randn(2, 4, 8, device=device, requires_grad=True)
    output = neuron.step(
        torch.randn(2, 3, device=device),
        incoming_messages=messages,
        incoming_strengths=torch.ones(2, 4, device=device),
        incoming_edge_types=torch.zeros(2, 4, dtype=torch.long, device=device),
    )
    output.value.square().sum().backward()
    assert output.value.shape == (2, 8)
    assert any(name.startswith("message_aggregator.") for name, _ in neuron.named_parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in neuron.named_parameters()
        if name.startswith("message_aggregator.")
    )
