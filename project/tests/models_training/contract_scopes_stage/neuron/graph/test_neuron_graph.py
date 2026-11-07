from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from newera.neuron import ModelNeuron, ModelNeuronGraph
from newera.neuron.graph import (
    DynamicNeuronGraphLayer,
    LinkMutation,
    LinkProposalPolicy,
    NeuronGraphTopology,
    TopologyController,
)
from newera.nlp_flow.components.research.training.objectives.training_losses import batch_loss
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def test_graph_accepts_unbounded_per_neuron_degree_and_backpropagates() -> None:
    nodes = torch.randn(1, 24, 8, requires_grad=True)
    topology = NeuronGraphTopology(
        batch_size=1,
        nodes_per_graph=24,
        edge_index=torch.tensor([[*range(20)], [*[23] * 20]], dtype=torch.long),
        edge_type=torch.zeros(20, dtype=torch.long),
        strength=torch.ones(20),
    )
    layer = DynamicNeuronGraphLayer(8, 2)
    with torch.no_grad():
        layer.residual_gate.fill_(0.5)

    result = layer(nodes, topology)
    result.hidden_states.square().mean().backward()

    assert result.hidden_states.shape == nodes.shape
    assert result.diagnostics["edge_count"] == 20
    assert result.diagnostics["max_weighted_indegree"] == 20
    assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
    assert layer.message_projection[0].weight.grad is not None


def test_model_neuron_graph_shares_one_weight_set_with_variable_fanin_and_per_node_state() -> None:
    torch.manual_seed(113)
    neuron_count = 21
    neuron = ModelNeuron(4, 8, 4, num_edge_types=2)
    graph = ModelNeuronGraph(neuron)
    topology = NeuronGraphTopology(
        batch_size=1,
        nodes_per_graph=neuron_count,
        edge_index=torch.tensor([list(range(20)), [20] * 20], dtype=torch.long),
        edge_type=torch.arange(20, dtype=torch.long) % 2,
        strength=torch.ones(20),
        node_ids=torch.arange(100, 100 + neuron_count).unsqueeze(0),
    )
    inputs = torch.randn(3, neuron_count, 4)

    first = graph(inputs, topology)
    second = graph(inputs, topology, state=first.state)
    second.values[:, 20].sum().backward()

    assert second.values.shape == (3, neuron_count, 4)
    assert len(list(graph.parameters())) == len(list(neuron.parameters()))
    assert neuron.output_projection.weight.grad is not None
    assert neuron.output_projection.weight.grad.abs().sum() > 0
    assert neuron.message_projection.weight.grad is not None
    assert neuron.message_projection.weight.grad.abs().sum() > 0
    assert len(graph.state_dict()) == len(neuron.state_dict())
    assert first.state.neuron_states.shape == (3, neuron_count, neuron.hidden_size)
    assert not torch.equal(first.state.neuron_states[:, 0], first.state.neuron_states[:, 1])


def test_model_neuron_graph_node_count_does_not_duplicate_weights() -> None:
    neuron = ModelNeuron(3, 6, 3)
    graph = ModelNeuronGraph(neuron)
    parameter_count = sum(parameter.numel() for parameter in graph.parameters())
    for node_count in (2, 13):
        topology = NeuronGraphTopology.empty(
            1,
            node_count,
            device="cpu",
            node_ids=torch.arange(100, 100 + node_count).unsqueeze(0),
        )
        output = graph(torch.randn(2, node_count, 3), topology)
        assert output.values.shape == (2, node_count, 3)
        assert sum(parameter.numel() for parameter in graph.parameters()) == parameter_count


def test_model_neuron_graph_requires_nonnegative_stable_identities() -> None:
    graph = ModelNeuronGraph(ModelNeuron(3, 6, 3))
    inputs = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="stable node_ids"):
        graph(inputs, NeuronGraphTopology.empty(1, 2, device="cpu"))
    padded = NeuronGraphTopology.empty(
        1, 2, device="cpu", node_ids=torch.tensor([[10, -1]])
    )
    with pytest.raises(ValueError, match="registered non-negative"):
        graph(inputs, padded)


def test_model_neuron_graph_candidate_edge_weights_receive_task_gradients() -> None:
    torch.manual_seed(37)
    neuron = ModelNeuron(3, 6, 3, num_edge_types=2)
    graph = ModelNeuronGraph(neuron)
    topology = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0], [2]]),
        torch.tensor([0]),
        torch.tensor([0.5]),
        torch.tensor([[10, 20, 30]]),
    )
    inputs = torch.randn(2, 3, 3)
    candidates = torch.tensor([[0, 1], [2, 2]])
    weights = torch.full((2, 2, 2), 0.5, requires_grad=True)

    first = graph(inputs, topology)
    second = graph(
        inputs,
        topology,
        state=first.state,
        candidate_edges=candidates,
        candidate_type_weights=weights,
    )
    second.values[:, 2].square().mean().backward()

    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert weights.grad.abs().sum() > 0
    assert neuron.message_projection.weight.grad is not None


def test_model_neuron_graph_state_reorders_and_extends_by_stable_identity() -> None:
    graph = ModelNeuronGraph(ModelNeuron(3, 6, 3))
    old_ids = torch.tensor([[10, 20, 30]])
    old_topology = NeuronGraphTopology.empty(1, 3, device="cpu", node_ids=old_ids)
    inputs = torch.randn(2, 3, 3)
    first = graph(inputs, old_topology)
    new_ids = torch.tensor([[30, 10, 40, 20]])
    new_topology = NeuronGraphTopology.empty(1, 4, device="cpu", node_ids=new_ids)

    with pytest.raises(ValueError, match="remap state by node ID"):
        graph(torch.randn(2, 4, 3), new_topology, state=first.state)
    remapped = graph.remap_state(first.state, new_ids)
    next_output = graph(torch.randn(2, 4, 3), new_topology, state=remapped)

    assert remapped.node_ids.tolist() == [30, 10, 40, 20]
    assert torch.equal(remapped.previous_values[:, 0], first.state.previous_values[:, 2])
    assert torch.equal(remapped.previous_values[:, 1], first.state.previous_values[:, 0])
    assert torch.count_nonzero(remapped.previous_values[:, 2]) == 0
    assert torch.count_nonzero(remapped.neuron_states[:, 2]) == 0
    assert torch.equal(remapped.neuron_states[:, 0], first.state.neuron_states[:, 2])
    assert next_output.values.shape == (2, 4, 3)


def test_topology_controller_applies_many_add_retype_remove_without_degree_cap() -> None:
    topology = NeuronGraphTopology.empty(1, 12, device="cpu")
    mutations = tuple(
        LinkMutation("add", 0, source, 11, edge_type=source % 3)
        for source in range(11)
    )
    controller = TopologyController()
    connected = controller.apply(topology, mutations, num_edge_types=3)
    changed = controller.apply(
        connected,
        (
            LinkMutation("retype", 0, 0, 11, edge_type=2, strength=0.25),
            LinkMutation("remove", 0, 1, 11),
        ),
        num_edge_types=3,
    )

    assert connected.edge_count == 11
    assert changed.edge_count == 10
    assert changed.edge_type[0].item() == 2
    assert changed.strength[0].item() == pytest.approx(0.25)
    assert changed.edge_index[1].tolist() == [11] * 10


def test_topology_controller_supports_optional_global_budget() -> None:
    topology = NeuronGraphTopology.empty(1, 5, device="cpu")
    controller = TopologyController(max_edges=2)
    with pytest.raises(ValueError, match="global edge budget"):
        controller.apply(
            topology,
            tuple(LinkMutation("add", 0, source, 4, edge_type=0) for source in range(3)),
            num_edge_types=1,
        )


def test_topology_rejects_future_cross_batch_and_duplicate_links() -> None:
    topology = NeuronGraphTopology.empty(1, 3, device="cpu")
    controller = TopologyController()
    with pytest.raises(ValueError, match="future"):
        controller.apply(topology, (LinkMutation("add", 0, 2, 0, edge_type=0),), num_edge_types=1)
    with pytest.raises(ValueError, match="already exists"):
        controller.apply(
            topology,
            (
                LinkMutation("add", 0, 0, 2, edge_type=0),
                LinkMutation("add", 0, 0, 2, edge_type=0),
            ),
            num_edge_types=1,
        )
    with pytest.raises(ValueError, match="batch graph boundaries"):
        invalid = NeuronGraphTopology(
            2,
            3,
            torch.tensor([[0], [3]]),
            torch.tensor([0]),
            torch.ones(1),
        )
        invalid.validate(batch_size=2, nodes_per_graph=3, num_edge_types=1, device=torch.device("cpu"), causal=True)


def test_link_proposal_policy_has_trainable_actions_and_no_top_k() -> None:
    nodes = torch.randn(1, 7, 6, requires_grad=True)
    topology = NeuronGraphTopology.causal_chain(1, 7, device="cpu")
    candidates = torch.tensor(
        [[0, 0, 0, 1, 2, 3], [2, 3, 4, 4, 5, 6]], dtype=torch.long
    )
    policy = LinkProposalPolicy(6, 2)
    logits = policy(nodes, candidates, topology)
    loss = policy.supervised_loss(logits, torch.tensor([1, 1, 1, 2, 0, 3]))
    loss.backward()

    assert logits.shape == (6, 4)
    assert nodes.grad is not None
    assert policy.scorer[0].weight.grad is not None
    add_logits = torch.full((candidates.shape[1], 4), -10.0)
    add_logits[:, 1] = 10.0
    proposals = policy.propose(add_logits, candidates, topology)
    assert len(proposals) == candidates.shape[1]
    assert all(mutation.operation == "add" for mutation in proposals)


def test_compact_decoder_uses_graph_before_decoder_and_returns_mutable_topology() -> None:
    torch.manual_seed(5)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_max_edges=8,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    node_ids = torch.tensor([[101, 102, 103, 104]])
    topology = NeuronGraphTopology.causal_chain(1, 4, device="cpu", node_ids=node_ids)
    output = model(
        input_ids,
        neuron_topology=topology,
        neuron_node_ids=node_ids,
        neuron_mutations=(LinkMutation("add", 0, 0, 3, edge_type=1),),
    )
    loss = output.token_logits.square().mean()
    loss.backward()

    assert output.hidden_states.shape == (1, 4, 16)
    assert output.neuron_topology is not None and output.neuron_topology.edge_count == 4
    assert output.neuron_graph_diagnostics is not None
    assert output.neuron_graph_diagnostics["edge_count"] == 4
    assert model.neuron_graph is not None
    assert model.neuron_graph.message_projection[0].weight.grad is not None
    assert model.neuron_graph.message_projection[0].weight.grad.abs().sum() > 0
    assert model.neuron_link_policy is not None


def test_graph_filters_padding_and_packed_sequence_edges() -> None:
    topology = NeuronGraphTopology(
        1,
        5,
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.ones(4),
    )
    mask = torch.tensor([[False, True, True, True, True]])
    segments = torch.tensor([[0, 0, 0, 1, 1]])
    filtered = topology.filter_context(attention_mask=mask, segment_ids=segments)

    assert filtered.edge_index.tolist() == [[1, 3], [2, 4]]


@pytest.mark.parametrize("enable_dynamic_stack", [False, True])
def test_model_padding_and_packed_segments_do_not_leak_between_valid_tokens(enable_dynamic_stack: bool) -> None:
    torch.manual_seed(9)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_candidate_budget=None,
            enable_dynamic_stack=enable_dynamic_stack,
        )
    ).eval()
    mask = torch.tensor([[False, True, True, True]])
    left_a = model(torch.tensor([[1, 2, 3, 4]]), attention_mask=mask).hidden_states
    left_b = model(torch.tensor([[9, 2, 3, 4]]), attention_mask=mask).hidden_states
    assert torch.allclose(left_a[:, 1:], left_b[:, 1:], atol=1e-6)

    segments = torch.tensor([[0, 0, 1, 1]])
    packed_a = model(torch.tensor([[1, 2, 3, 4]]), segment_ids=segments).hidden_states
    packed_b = model(torch.tensor([[8, 9, 3, 4]]), segment_ids=segments).hidden_states
    assert torch.allclose(packed_a[:, 2:], packed_b[:, 2:], atol=1e-6)


def test_candidate_search_budget_does_not_cap_stored_node_degree() -> None:
    candidates = NeuronGraphTopology.causal_candidates(
        1,
        12,
        device="cpu",
        max_candidates_per_graph=3,
        randomize=False,
    )
    all_candidates = NeuronGraphTopology.causal_candidates(
        1,
        12,
        device="cpu",
        max_candidates_per_graph=None,
    )
    assert candidates.shape == (2, 3)
    assert all_candidates.shape == (2, 66)

    topology = NeuronGraphTopology.empty(1, 12, device="cpu")
    fully_connected_target = TopologyController().apply(
        topology,
        tuple(LinkMutation("add", 0, source, 11, edge_type=0) for source in range(11)),
        num_edge_types=1,
    )
    assert fully_connected_target.edge_count == 11


def test_candidate_search_fills_budget_inside_short_packed_segments() -> None:
    segments = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    candidates = NeuronGraphTopology.causal_candidates(
        1,
        8,
        device="cpu",
        segment_ids=segments,
        max_candidates_per_graph=4,
        randomize=False,
    )
    assert candidates.shape == (2, 4)
    assert torch.equal(
        segments[0, candidates[0] % 8],
        segments[0, candidates[1] % 8],
    )


def test_candidate_offset_scans_without_wrapping_or_repeating() -> None:
    first = NeuronGraphTopology.causal_candidates(
        1, 4, device="cpu", max_candidates_per_graph=3, randomize=False
    )
    second = NeuronGraphTopology.causal_candidates(
        1,
        4,
        device="cpu",
        max_candidates_per_graph=3,
        randomize=False,
        candidate_offset=3,
    )
    exhausted = NeuronGraphTopology.causal_candidates(
        1,
        4,
        device="cpu",
        max_candidates_per_graph=3,
        randomize=False,
        candidate_offset=6,
    )
    assert first.shape == second.shape == (2, 3)
    assert not set(map(tuple, first.t().tolist())) & set(map(tuple, second.t().tolist()))
    assert exhausted.shape == (2, 0)


def test_negative_segment_ids_are_invalid_for_existing_and_custom_edges() -> None:
    segments = torch.tensor([[-1, -1, 2, 2]])
    topology = NeuronGraphTopology(
        1,
        4,
        torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2),
    )
    assert topology.filter_context(segment_ids=segments).edge_index.tolist() == [[2], [3]]
    candidates = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    assert NeuronGraphTopology.filter_candidates(
        candidates,
        batch_size=1,
        nodes_per_graph=4,
        segment_ids=segments,
    ).tolist() == [[2], [3]]


def test_decoder_rejects_duplicate_edges_in_supplied_topology_and_candidates() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
        )
    )
    duplicate_topology = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2),
        torch.tensor([[10, 11, 12]]),
    )
    with pytest.raises(ValueError, match="duplicate directed edges"):
        model(
            torch.tensor([[1, 2, 3]]),
            neuron_topology=duplicate_topology,
            neuron_node_ids=duplicate_topology.node_ids,
        )
    with pytest.raises(ValueError, match="cannot contain duplicate"):
        model(
            torch.tensor([[1, 2, 3]]),
            neuron_candidate_edges=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        )


def test_existing_links_are_scored_and_hard_updates_require_budget() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_candidate_budget=1,
        )
    ).eval()
    output = model(torch.tensor([[1, 2, 3, 4]]))
    assert output.neuron_candidate_edges is not None
    assert {tuple(edge) for edge in output.neuron_candidate_edges.t().tolist()} >= {
        (0, 1), (1, 2), (2, 3)
    }
    with pytest.raises(ValueError, match="finite neuron_graph_max_edges"):
        model(torch.tensor([[1, 2, 3, 4]]), update_neuron_topology=True)
    model.train()
    with pytest.raises(ValueError, match="hard topology updates require eval mode"):
        model(
            torch.tensor([[1, 2, 3, 4]]),
            neuron_mutations=(LinkMutation("add", 0, 0, 3, edge_type=0),),
        )

    bounded = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_max_edges=8,
        )
    ).eval()
    with pytest.raises(ValueError, match="cannot connect padding or different segments"):
        bounded(
            torch.tensor([[1, 2, 3, 4]]),
            segment_ids=torch.tensor([[0, 1, 1, 1]]),
            neuron_mutations=(LinkMutation("add", 0, 0, 1, edge_type=0),),
        )


def test_reused_topology_requires_exact_stable_node_identity() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    node_ids = torch.tensor([[501, 502, 503]])
    topology = NeuronGraphTopology.causal_chain(1, 3, device="cpu", node_ids=node_ids)
    output = model(input_ids, neuron_topology=topology, neuron_node_ids=node_ids)
    assert output.neuron_topology is not None
    with pytest.raises(ValueError, match="node identities do not match"):
        model(
            input_ids,
            neuron_topology=output.neuron_topology,
            neuron_node_ids=torch.tensor([[501, 999, 503]]),
        )
    with pytest.raises(ValueError, match="requires stored and supplied"):
        model(input_ids, neuron_topology=output.neuron_topology)

    changed_order = torch.tensor([[503, 502, 501]])
    with pytest.raises(ValueError, match="node identities do not match"):
        model(
            input_ids,
            neuron_topology=output.neuron_topology,
            neuron_node_ids=changed_order,
        )


def test_global_edge_budget_is_checked_without_any_mutations() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_max_edges=1,
        )
    )
    with pytest.raises(ValueError, match="global edge budget"):
        model(torch.tensor([[1, 2, 3]]))


def test_zero_strength_edges_do_not_change_weighted_aggregation() -> None:
    torch.manual_seed(13)
    nodes = torch.randn(1, 3, 8)
    layer = DynamicNeuronGraphLayer(8, 1)
    with torch.no_grad():
        layer.residual_gate.fill_(1.0)
    empty = NeuronGraphTopology.empty(1, 3, device="cpu")
    zero_weight = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0], [2]], dtype=torch.long),
        torch.tensor([0]),
        torch.tensor([0.0]),
    )

    assert torch.allclose(layer(nodes, empty).hidden_states, layer(nodes, zero_weight).hidden_states)


def test_topology_append_preserves_batch_edges_and_adds_incremental_chain() -> None:
    topology = NeuronGraphTopology.causal_chain(2, 3, device="cpu")
    extended = topology.append_causal_nodes(2)
    extended.validate(
        batch_size=2,
        nodes_per_graph=5,
        num_edge_types=1,
        device=torch.device("cpu"),
        causal=True,
    )

    assert extended.nodes_per_graph == 5
    assert extended.edge_count == 8
    assert extended.edge_index[0].tolist() == [0, 1, 5, 6, 2, 7, 3, 8]
    assert extended.edge_index[1].tolist() == [1, 2, 6, 7, 3, 8, 4, 9]

    identified = NeuronGraphTopology.causal_chain(
        1, 3, device="cpu", node_ids=torch.tensor([[40, 41, 42]])
    )
    with pytest.raises(ValueError, match="requires matching new node_ids"):
        identified.append_causal_nodes(1)
    identified_extension = identified.append_causal_nodes(
        2, node_ids=torch.tensor([[43, 44]])
    )
    assert identified_extension.node_ids is not None
    assert identified_extension.node_ids.tolist() == [[40, 41, 42, 43, 44]]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS is unavailable")
def test_mps_graph_forward_backward_supports_deterministic_mode() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device = torch.device("mps")
        nodes = torch.randn(1, 5, 8, device=device, requires_grad=True)
        segments = torch.tensor([[0, 0, 1, 1, 1]], device=device)
        packed_candidates = NeuronGraphTopology.causal_candidates(
            1,
            5,
            device=device,
            segment_ids=segments,
            max_candidates_per_graph=2,
            randomize=False,
        )
        assert packed_candidates.device.type == "mps"
        assert torch.equal(
            segments[0, packed_candidates[0] % 5],
            segments[0, packed_candidates[1] % 5],
        )
        topology = NeuronGraphTopology.causal_chain(1, 5, device=device)
        layer = DynamicNeuronGraphLayer(8, 2).to(device)
        output = layer(nodes, topology).hidden_states
        output.square().mean().backward()
        assert nodes.grad is not None and torch.isfinite(nodes.grad).all()
        assert layer.message_projection[0].weight.grad is not None
        repeated = layer(nodes.detach(), topology).hidden_states
        assert torch.allclose(output.detach(), repeated, atol=1e-6)

        model_neuron = ModelNeuron(4, 8, 4, num_edge_types=2)
        model_graph = ModelNeuronGraph(model_neuron).to(device)
        model_topology = NeuronGraphTopology(
            1,
            5,
            torch.tensor([[0, 1, 2, 3], [4, 4, 4, 4]], device=device),
            torch.tensor([0, 1, 0, 1], device=device),
            torch.ones(4, device=device),
            torch.arange(50, 55, device=device).unsqueeze(0),
        )
        model_inputs = torch.randn(2, 5, 4, device=device)
        model_first = model_graph(model_inputs, model_topology)
        model_second = model_graph(model_inputs, model_topology, state=model_first.state)
        model_second.values[:, 4].square().mean().backward()
        assert torch.isfinite(model_second.values).all()
        assert model_neuron.output_projection.weight.grad is not None

        candidate_weights = torch.full((2, 2, 2), 0.25, device=device, requires_grad=True)
        candidate_edges = torch.tensor([[0, 1], [4, 4]], device=device)
        candidate_first = model_graph(model_inputs, model_topology)
        candidate_second = model_graph(
            model_inputs,
            model_topology,
            state=candidate_first.state,
            candidate_edges=candidate_edges,
            candidate_type_weights=candidate_weights,
        )
        candidate_second.values[:, 4].sum().backward()
        assert candidate_weights.grad is not None and torch.isfinite(candidate_weights.grad).all()

        model = CompactNLPDecoder(
            NLPModelConfig(
                vocab_size=16,
                hidden_size=8,
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
                intermediate_size=16,
                max_seq_len=8,
                enable_neuron_graph=True,
                neuron_graph_candidate_budget=4,
            )
        ).to(device)
        model_node_ids = torch.tensor([[100, 101, 102, 103]], device=device)
        model_output = model(
            torch.tensor([[1, 2, 3, 4]], device=device),
            neuron_node_ids=model_node_ids,
        )
        model_output.token_logits.square().mean().backward()
        assert model_output.neuron_topology is not None
        assert model_output.neuron_topology.node_ids is not None
        assert torch.equal(model_output.neuron_topology.node_ids, model_node_ids)
        assert model.neuron_link_policy is not None
        assert model.neuron_link_policy.scorer[-1].weight.grad is not None
    finally:
        torch.use_deterministic_algorithms(previous)

    device = torch.device("mps")
    regular_model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_candidate_budget=4,
        )
    ).to(device)
    regular_output = regular_model(torch.tensor([[1, 2, 3, 4]], device=device))
    regular_output.token_logits.square().mean().backward()
    assert regular_model.neuron_graph is not None
    assert regular_model.neuron_graph.message_projection[0].weight.grad is not None


def test_joint_training_loss_can_supervise_link_actions() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_candidate_budget=None,
        )
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "neuron_candidate_edges": torch.tensor(
            [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long
        ),
        "neuron_link_targets": torch.tensor([0, 1, 1, 0, 2, 0]),
    }
    loss, _, _, _ = batch_loss(
        model,
        batch,
        device=torch.device("cpu"),
        autocast=nullcontext(),
        neuron_link_weight=0.25,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.neuron_link_policy is not None
    gradient = model.neuron_link_policy.scorer[-1].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0


def test_stable_neuron_ids_may_repeat_across_independent_batch_graphs() -> None:
    node_ids = torch.tensor([[10, 11, 12], [10, 11, 12]])
    topology = NeuronGraphTopology.causal_chain(2, 3, device="cpu", node_ids=node_ids)
    topology.validate(
        batch_size=2,
        nodes_per_graph=3,
        num_edge_types=1,
        device=torch.device("cpu"),
        causal=True,
    )


def test_stable_neuron_ids_must_be_unique_within_each_graph() -> None:
    topology = NeuronGraphTopology.causal_chain(
        1, 3, device="cpu", node_ids=torch.tensor([[10, 10, 12]])
    )
    with pytest.raises(ValueError, match="unique within each graph"):
        topology.validate(
            batch_size=1,
            nodes_per_graph=3,
            num_edge_types=1,
            device=torch.device("cpu"),
            causal=True,
        )


def test_link_policy_keep_bias_is_configurable_for_exploration() -> None:
    from newera.neuron.graph import LinkProposalPolicy

    policy = LinkProposalPolicy(8, 2, initial_keep_bias=0.0)
    assert policy.scorer[-1].bias[0].item() == 0.0

    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_initial_keep_bias=0.25,
        )
    )
    assert model.neuron_link_policy is not None
    assert model.neuron_link_policy.scorer[-1].bias[0].item() == 0.25


def test_topology_identity_remap_preserves_survivors_and_disconnects_new_nodes() -> None:
    ids = torch.tensor([[10, 20, 30], [10, 20, 30]])
    topology = NeuronGraphTopology(
        2,
        3,
        torch.tensor([[0, 1, 3, 4], [2, 2, 5, 5]]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0.5, 1.0, 0.75, 1.25]),
        ids,
    )
    remapped = topology.remap_node_ids(torch.tensor([[30, 10, 40], [30, 10, 40]]))
    assert remapped.nodes_per_graph == 3
    assert remapped.node_ids is not None
    assert remapped.node_ids.tolist() == [[30, 10, 40], [30, 10, 40]]
    assert remapped.edge_index.tolist() == [[1, 4], [0, 3]]
    assert remapped.edge_type.tolist() == [0, 0]
    assert torch.allclose(remapped.strength, torch.tensor([0.5, 0.75]))


def test_topology_identity_remap_rejects_causal_reversal() -> None:
    topology = NeuronGraphTopology.causal_chain(
        1, 3, device="cpu", node_ids=torch.tensor([[10, 20, 30]])
    )
    with pytest.raises(ValueError, match="violate causal"):
        topology.remap_node_ids(torch.tensor([[20, 10, 30]]), causal=True)


def test_link_policy_rejects_nonfinite_keep_bias() -> None:
    from newera.neuron.graph import LinkProposalPolicy

    with pytest.raises(ValueError, match="initial_keep_bias must be finite"):
        LinkProposalPolicy(8, 2, initial_keep_bias=float("nan"))


def test_decoder_predicts_and_applies_candidate_strengths_when_enabled() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            max_seq_len=8,
            enable_neuron_graph=True,
            neuron_graph_predict_strengths=True,
            neuron_graph_max_edges=8,
            neuron_graph_initial_keep_bias=0.0,
        )
    ).eval()
    assert model.neuron_link_policy is not None
    with torch.no_grad():
        model.neuron_link_policy.scorer[-1].weight.zero_()
        model.neuron_link_policy.scorer[-1].bias.copy_(torch.tensor([5.0, 0.0, -1.0, -1.0, -1.0, -1.0]))
        model.neuron_link_policy.strength_scorer[-1].weight.zero_()
        model.neuron_link_policy.strength_scorer[-1].bias.fill_(2.0)
    output = model(
        torch.tensor([[1, 2, 3]]),
        neuron_candidate_edges=torch.tensor([[0], [1]]),
        neuron_node_ids=torch.tensor([[10, 11, 12]]),
        update_neuron_topology=True,
        neuron_link_threshold=0.5,
    )
    assert output.neuron_topology is not None
    assert output.neuron_link_strengths is not None
    assert output.neuron_link_strengths.shape == (2,)
    edge = output.neuron_topology.edge_index.t().tolist().index([0, 1])
    assert output.neuron_topology.strength[edge].item() > 0.8
