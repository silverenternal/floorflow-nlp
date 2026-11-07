import pytest
import torch

from newera.neuron.cells.aggregation import GroupRoleDeepSetsAggregator
from newera.neuron.composition import ModelNeuron

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _cell(device: str, aggregator: torch.nn.Module | None) -> ModelNeuron:
    torch.manual_seed(19)
    return ModelNeuron(
        input_size=1,
        hidden_size=32,
        message_size=9,
        num_edge_types=2,
        message_aggregator=aggregator,
    ).to(device).eval()


def _pairing_output(
    cell: ModelNeuron,
    group_ids: torch.Tensor,
    order: torch.Tensor | None = None,
) -> torch.Tensor:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=group_ids.device)
    roles = torch.tensor([[0, 1, 0, 1]], device=group_ids.device)
    if order is not None:
        values, roles, group_ids = values[:, order], roles[:, order], group_ids[:, order]
    features = torch.zeros((*values.shape, 8), device=values.device)
    return cell.step(
        torch.ones((1, 1), device=group_ids.device),
        incoming_messages=torch.cat((values.unsqueeze(-1), features), dim=-1),
        incoming_strengths=torch.ones_like(values),
        incoming_edge_types=roles,
        incoming_group_ids=group_ids.to(torch.long),
    ).value


@pytest.mark.parametrize("device_name", DEVICES)
def test_global_project_then_sum_cannot_distinguish_group_pairing(device_name: str):
    device = torch.device(device_name)
    cell = _cell(device_name, None)
    aligned = _pairing_output(cell, torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device))
    crossed = _pairing_output(cell, torch.tensor([[0.0, 1.0, 1.0, 0.0]], device=device))
    torch.testing.assert_close(aligned, crossed, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device_name", DEVICES)
def test_set_transformer_preserves_group_pairing_information(device_name: str):
    device = torch.device(device_name)
    aggregator = GroupRoleDeepSetsAggregator(32, num_edge_types=2)
    cell = _cell(device_name, aggregator)
    aligned = _pairing_output(cell, torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device))
    crossed = _pairing_output(cell, torch.tensor([[0.0, 1.0, 1.0, 0.0]], device=device))
    assert not torch.allclose(aligned, crossed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device_name", DEVICES)
def test_grouped_message_set_is_permutation_invariant(device_name: str):
    device = torch.device(device_name)
    cell = _cell(device_name, GroupRoleDeepSetsAggregator(32, num_edge_types=2))
    ids = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
    order = torch.tensor([2, 0, 3, 1], device=device)
    original = _pairing_output(cell, ids)
    permuted = _pairing_output(cell, ids, order)
    torch.testing.assert_close(original, permuted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device_name", DEVICES)
def test_sum_residual_candidate_starts_from_sum_and_registers_attention(device_name: str):
    from newera.neuron.cells.aggregation import SumResidualSetTransformerAggregator

    device = torch.device(device_name)
    torch.manual_seed(41)
    candidate = SumResidualSetTransformerAggregator(8, heads=2).to(device)
    messages = torch.randn(2, 5, 8, device=device)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]], device=device)
    torch.testing.assert_close(candidate.forward_batched(messages, mask=mask), (messages * mask.unsqueeze(-1)).sum(1))
    candidate(messages[0], mask=mask[0]).square().sum().backward()
    assert candidate.residual_scale.grad is not None
    assert any(name.startswith("set_transformer.") for name, _ in candidate.named_parameters())
    assert all(parameter.grad is not None for parameter in candidate.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_contextual_set_residual_is_owned_by_model_neuron(device_name: str):
    from newera.neuron.cells.aggregation import ContextualSumResidualSetTransformerAggregator

    device = torch.device(device_name)
    torch.manual_seed(71)
    aggregator = ContextualSumResidualSetTransformerAggregator(8, context_size=2, heads=2)
    cell = ModelNeuron(
        input_size=2,
        hidden_size=8,
        message_size=8,
        message_aggregator=aggregator,
    ).to(device)
    output = cell.step(
        torch.randn(3, 2, device=device),
        incoming_messages=torch.randn(3, 5, 8, device=device, requires_grad=True),
        incoming_strengths=torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float32,
            device=device,
        ),
        incoming_edge_types=torch.zeros((3, 5), dtype=torch.long, device=device),
    )
    output.value.square().sum().backward()
    assert output.value.shape == (3, 8)
    assert any(name.startswith("message_aggregator.context_gate") for name, _ in cell.named_parameters())
    assert all(parameter.grad is not None for parameter in aggregator.parameters())
