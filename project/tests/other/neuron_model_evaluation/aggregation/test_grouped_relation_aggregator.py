import pytest
import torch

from newera.neuron.cells.aggregation import GroupRoleDeepSetsAggregator
from newera.neuron.composition import ModelNeuron

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _dense_group_reference(aggregator, messages, roles, labels, mask):
    items = messages.shape[1]
    positions = torch.arange(items, device=messages.device)
    same_group = labels.unsqueeze(2) == labels.unsqueeze(1)
    earlier_valid = mask.unsqueeze(1) & (
        positions.view(1, 1, items) < positions.view(1, items, 1)
    )
    first = mask & ~(same_group & earlier_valid).any(dim=2)
    members = same_group & first.unsqueeze(2) & mask.unsqueeze(1)
    summaries = [
        torch.bmm(members.to(messages.dtype), messages * ((roles == role) & mask).unsqueeze(-1))
        for role in range(aggregator.num_edge_types)
    ]
    features = torch.cat(summaries, dim=-1)
    if aggregator.pairwise_interactions:
        products = []
        index = 0
        for left in range(aggregator.num_edge_types):
            for right in range(left + 1, aggregator.num_edge_types):
                if aggregator.bilinear_interactions:
                    left_projection, right_projection = aggregator.pair_projections[index]
                    products.append(left_projection(summaries[left]) * right_projection(summaries[right]))
                    index += 1
                else:
                    products.append(summaries[left] * summaries[right])
        features = torch.cat((features, *products), dim=-1)
    return (aggregator.group_network(features) * first.unsqueeze(-1)).sum(dim=1)


@pytest.mark.parametrize("device_name", DEVICES)
def test_group_relation_is_permutation_and_relabel_invariant(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(181)
    aggregator = GroupRoleDeepSetsAggregator(8, 2).to(device).eval()
    messages = torch.randn(1, 6, 8, device=device, requires_grad=True)
    roles = torch.tensor([[0, 1, 0, 1, 0, 1]], device=device)
    labels = torch.tensor([[2**60, 2**60, -2**60, -2**60, 77, 77]], device=device)
    mask = torch.ones((1, 6), dtype=torch.bool, device=device)
    expected = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask
    )
    order = torch.tensor([5, 2, 0, 4, 1, 3], device=device)
    actual = aggregator.forward_batched_with_relations(
        messages[:, order], edge_types=roles[:, order],
        group_ids=labels[:, order] * -3 + 123456789, mask=mask[:, order]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_empty_groups_and_padding_have_zero_identity(device_name: str):
    device = torch.device(device_name)
    aggregator = GroupRoleDeepSetsAggregator(8, 2).to(device)
    messages = torch.randn(2, 4, 8, device=device)
    roles = torch.zeros((2, 4), dtype=torch.long, device=device)
    labels = torch.tensor([[9, 9, 5, 5], [2**60, 2**60, 0, 0]], device=device)
    mask = torch.tensor([[True, True, False, False], [False] * 4], device=device)
    result = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask
    )
    expected = aggregator.forward_batched_with_relations(
        messages[:, :2], edge_types=roles[:, :2], group_ids=labels[:, :2],
        mask=torch.tensor([[True, True], [False, False]], device=device)
    )
    torch.testing.assert_close(result, expected)
    assert torch.equal(result[1], torch.zeros_like(result[1]))


@pytest.mark.parametrize("device_name", DEVICES)
def test_group_role_encoder_can_condition_on_neuron_context(device_name: str):
    device = torch.device(device_name)
    aggregator = GroupRoleDeepSetsAggregator(8, 2, context_size=3).to(device)
    messages = torch.randn(2, 4, 8, device=device, requires_grad=True)
    context = torch.randn(2, 3, device=device, requires_grad=True)
    result = aggregator.forward_batched_with_relations(
        messages,
        edge_types=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]], device=device),
        group_ids=torch.tensor([[9, 9, 3, 3], [9, 9, 3, 3]], device=device),
        mask=torch.ones((2, 4), dtype=torch.bool, device=device),
        context=context,
    )
    result.square().sum().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_pairwise_role_interaction_features_are_deterministic_and_differentiable(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(188)
    aggregator = GroupRoleDeepSetsAggregator(
        8, 2, pairwise_interactions=True
    ).to(device)
    messages = torch.randn(1, 4, 8, device=device, requires_grad=True)
    result = aggregator.forward_batched_with_relations(
        messages,
        edge_types=torch.tensor([[0, 1, 0, 1]], device=device),
        group_ids=torch.tensor([[5, 5, 2**60, 2**60]], device=device),
        mask=torch.ones((1, 4), dtype=torch.bool, device=device),
    )
    result.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_contextual_group_relation_residual_starts_at_exact_sum(device_name: str):
    device = torch.device(device_name)
    aggregator = GroupRoleDeepSetsAggregator(
        8, 2, context_size=3, pairwise_interactions=True, sum_residual=True
    ).to(device)
    messages = torch.randn(2, 5, 8, device=device, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, False, False, False, False]],
        device=device,
    )
    actual = aggregator.forward_batched_with_relations(
        messages,
        edge_types=torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], device=device),
        group_ids=torch.tensor([[4, 4, 9, 0, 0], [2, 0, 0, 0, 0]], device=device),
        mask=mask,
        context=torch.randn(2, 3, device=device),
    )
    expected = (messages * mask.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert aggregator.residual_gate.weight.grad is not None


@pytest.mark.parametrize("device_name", DEVICES)
def test_low_rank_bilinear_group_interactions_are_permutation_invariant(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(191)
    aggregator = GroupRoleDeepSetsAggregator(
        8,
        2,
        context_size=3,
        bilinear_interactions=True,
        sum_residual=True,
    ).to(device).eval()
    messages = torch.randn(1, 6, 8, device=device, requires_grad=True)
    roles = torch.tensor([[0, 1, 0, 1, 0, 1]], device=device)
    labels = torch.tensor([[2**60, 2**60, -7, -7, 91, 91]], device=device)
    mask = torch.ones((1, 6), dtype=torch.bool, device=device)
    context = torch.randn(1, 3, device=device)
    actual = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask, context=context
    )
    order = torch.tensor([5, 2, 0, 4, 1, 3], device=device)
    permuted = aggregator.forward_batched_with_relations(
        messages[:, order], edge_types=roles[:, order], group_ids=labels[:, order],
        mask=mask[:, order], context=context,
    )
    torch.testing.assert_close(actual, permuted, atol=1e-6, rtol=1e-6)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_exact_bilinear_readout_is_permutation_invariant(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(194)
    aggregator = GroupRoleDeepSetsAggregator(
        8,
        2,
        context_size=3,
        bilinear_interactions=True,
        exact_bilinear=True,
        sum_residual=True,
    ).to(device).eval()
    messages = torch.randn(1, 17, 8, device=device, requires_grad=True)
    roles = torch.arange(17, device=device).remainder(2).reshape(1, -1)
    labels = torch.arange(17, device=device).remainder(6).reshape(1, -1) * 31 - 2**45
    mask = torch.rand(1, 17, device=device) > 0.1
    context = torch.randn(1, 3, device=device)
    actual = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask, context=context
    )
    order = torch.randperm(17, device=device)
    permuted = aggregator.forward_batched_with_relations(
        messages[:, order],
        edge_types=roles[:, order],
        group_ids=labels[:, order] + 9001,
        mask=mask[:, order],
        context=context,
    )
    torch.testing.assert_close(actual, permuted, atol=2e-6, rtol=2e-6)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


@pytest.mark.parametrize("device_name", DEVICES)
def test_full_bilinear_readout_is_permutation_invariant(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(195)
    aggregator = GroupRoleDeepSetsAggregator(
        8,
        2,
        context_size=3,
        bilinear_interactions=True,
        full_bilinear=True,
        sum_residual=True,
    ).to(device).eval()
    messages = torch.randn(1, 17, 8, device=device, requires_grad=True)
    roles = torch.arange(17, device=device).remainder(2).reshape(1, -1)
    labels = torch.arange(17, device=device).remainder(6).reshape(1, -1) * 31 - 2**45
    mask = torch.rand(1, 17, device=device) > 0.1
    context = torch.randn(1, 3, device=device)
    actual = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask, context=context
    )
    order = torch.randperm(17, device=device)
    permuted = aggregator.forward_batched_with_relations(
        messages[:, order],
        edge_types=roles[:, order],
        group_ids=labels[:, order] + 9001,
        mask=mask[:, order],
        context=context,
    )
    torch.testing.assert_close(actual, permuted, atol=2e-6, rtol=2e-6)
    actual.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()
    assert all(parameter.grad is not None for parameter in aggregator.parameters())


def test_relation_aggregation_handles_many_unique_groups_with_linear_workspace():
    device = torch.device("cpu")
    groups_count = 16_384
    aggregator = GroupRoleDeepSetsAggregator(8, 2).to(device)
    messages = torch.randn(1, groups_count, 8, device=device, requires_grad=True)
    roles = torch.arange(groups_count, device=device).remainder(2).reshape(1, -1)
    labels = torch.arange(groups_count, device=device).reshape(1, -1) * 17 - 2**40
    mask = torch.ones((1, groups_count), dtype=torch.bool, device=device)

    output = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask
    )
    assert output.shape == (1, 8)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert messages.grad is not None and torch.isfinite(messages.grad).all()


def test_sorted_large_degree_path_matches_dense_output_and_gradients():
    torch.manual_seed(192)
    aggregator = GroupRoleDeepSetsAggregator(8, 2, bilinear_interactions=True)
    messages = torch.randn(2, 257, 8, requires_grad=True)
    roles = torch.randint(0, 2, (2, 257))
    labels = torch.randint(-20, 20, (2, 257), dtype=torch.long)
    mask = torch.rand(2, 257) > 0.15

    actual = aggregator.forward_batched_with_relations(
        messages, edge_types=roles, group_ids=labels, mask=mask
    )
    expected = _dense_group_reference(aggregator, messages, roles, labels, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    actual.square().sum().backward()
    actual_message_grad = messages.grad.detach().clone()
    actual_parameter_grads = [parameter.grad.detach().clone() for parameter in aggregator.parameters()]
    aggregator.zero_grad(set_to_none=True)
    messages.grad = None
    expected.square().sum().backward()
    torch.testing.assert_close(messages.grad, actual_message_grad, atol=3e-6, rtol=3e-6)
    for expected_grad, parameter in zip(actual_parameter_grads, aggregator.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected_grad, atol=5e-5, rtol=1e-5)


def test_relation_aggregation_supports_deterministic_mps_backward():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device = torch.device("mps")
        torch.manual_seed(193)
        aggregator = GroupRoleDeepSetsAggregator(
            8, 2, context_size=3, bilinear_interactions=True, sum_residual=True
        ).to(device)
        messages = torch.randn(1, 4_097, 8, device=device, requires_grad=True)
        roles = torch.arange(4_097, device=device).remainder(2).reshape(1, -1)
        labels = torch.arange(4_097, device=device).reshape(1, -1) * 13 - 2**40
        mask = torch.rand(1, 4_097, device=device) > 0.2
        context = torch.randn(1, 3, device=device)

        output = aggregator.forward_batched_with_relations(
            messages,
            edge_types=roles,
            group_ids=labels,
            mask=mask,
            context=context,
        )
        output.square().sum().backward()
        assert torch.isfinite(output).all()
        assert messages.grad is not None and torch.isfinite(messages.grad).all()
        assert all(parameter.grad is not None for parameter in aggregator.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)


def test_model_neuron_requires_group_identity_for_relation_aggregator():
    neuron = ModelNeuron(2, 8, 8, num_edge_types=2, message_aggregator=GroupRoleDeepSetsAggregator(8, 2))
    kwargs = {
        "inputs": torch.randn(1, 2),
        "incoming_messages": torch.randn(1, 3, 8),
        "incoming_strengths": torch.ones(1, 3),
        "incoming_edge_types": torch.tensor([[0, 1, 0]]),
    }
    with pytest.raises(ValueError, match="requires incoming_group_ids"):
        neuron.step(**kwargs)
    output = neuron.step(**kwargs, incoming_group_ids=torch.tensor([[2**60, 2**60, -2**60]]))
    assert output.value.shape == (1, 8)


@pytest.mark.parametrize("device_name", DEVICES)
def test_structured_readout_channels_are_isolated_from_trainable_state(device_name: str):
    device = torch.device(device_name)
    torch.manual_seed(731)
    aggregator = GroupRoleDeepSetsAggregator(
        8,
        2,
        context_size=2,
        bilinear_interactions=True,
        full_bilinear=True,
        full_bilinear_bias=False,
        sum_residual=True,
    )
    neuron = ModelNeuron(
        2,
        8,
        9,
        num_edge_types=2,
        message_aggregator=aggregator,
        exact_input_sum_output_channel=0,
        aggregate_only_output_channels=(1,),
    ).to(device).eval()
    messages = torch.randn(2, 5, 9, device=device, requires_grad=True)
    strengths = torch.tensor([[1, 0.5, 1, 0, 2], [1, 1, 0, 0, 0]], device=device)
    roles = torch.tensor([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]], device=device)
    groups = torch.tensor([[4, 4, -2**50, -2**50, 4], [2**55, 2**55, 0, 0, 0]], device=device)
    inputs = torch.randn(2, 2, device=device)

    initial = neuron.step(
        inputs,
        incoming_messages=messages,
        incoming_strengths=strengths,
        incoming_edge_types=roles,
        incoming_group_ids=groups,
    )
    expected_sum = (messages * strengths.unsqueeze(-1)).sum(dim=1)[:, 0]
    assert torch.equal(initial.value[:, 0], expected_sum)
    direct_gradient = torch.autograd.grad(initial.value[:, 0].sum(), messages)[0]
    torch.testing.assert_close(direct_gradient[:, :, 0], strengths)
    assert torch.count_nonzero(direct_gradient[:, :, 1:]) == 0

    with torch.no_grad():
        neuron.output_projection.weight.fill_(1000)
        neuron.output_projection.bias.fill_(1000)
    after_state_change = neuron.step(
        inputs,
        incoming_messages=messages,
        incoming_strengths=strengths,
        incoming_edge_types=roles,
        incoming_group_ids=groups,
    )
    assert torch.equal(after_state_change.value[:, 0], expected_sum)
    torch.testing.assert_close(after_state_change.value[:, 1], initial.value[:, 1])

    empty = neuron.step(
        inputs,
        incoming_messages=messages[:, :0],
        incoming_strengths=strengths[:, :0],
        incoming_edge_types=roles[:, :0],
        incoming_group_ids=groups[:, :0],
    )
    assert torch.equal(empty.value[:, 0], torch.zeros(2, device=device))


def test_full_bilinear_can_remove_per_group_bias():
    aggregator = GroupRoleDeepSetsAggregator(
        8,
        2,
        bilinear_interactions=True,
        full_bilinear=True,
        full_bilinear_bias=False,
    )
    assert aggregator.full_bilinear_layer is not None
    assert aggregator.full_bilinear_layer.bias is None
    biased = GroupRoleDeepSetsAggregator(
        8, 2, bilinear_interactions=True, full_bilinear=True, full_bilinear_bias=True
    )
    exact_signature = ModelNeuron(2, 8, 9, message_aggregator=aggregator).architecture_signature
    biased_signature = ModelNeuron(2, 8, 9, message_aggregator=biased).architecture_signature
    assert exact_signature != biased_signature


@pytest.mark.parametrize("device_name", DEVICES)
def test_model_neuron_linear_aggregate_readout_scales_beyond_bounded_state(device_name: str):
    device = torch.device(device_name)
    neuron = ModelNeuron(1, 1, 1).to(device)
    with torch.no_grad():
        neuron.message_projection.weight.fill_(1.0)
        neuron.edge_type_embedding.weight.zero_()
        neuron.input_projection.weight.zero_()
        neuron.input_projection.bias.zero_()
        neuron.output_projection.weight.zero_()
        neuron.output_projection.bias.zero_()
        neuron.aggregate_output_projection.weight.fill_(1.0)
        neuron.aggregate_output_projection.bias.zero_()
    for degree in (3, 128):
        output = neuron.step(
            torch.zeros(1, 1, device=device),
            incoming_messages=torch.ones(1, degree, 1, device=device),
            incoming_strengths=torch.ones(1, degree, device=device),
            incoming_edge_types=torch.zeros(1, degree, dtype=torch.long, device=device),
        )
        torch.testing.assert_close(output.value, torch.tensor([[float(degree)]], device=device))
        assert output.state.abs().max() <= 1
