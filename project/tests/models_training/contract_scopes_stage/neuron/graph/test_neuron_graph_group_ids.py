import pytest
import torch

from newera.neuron import ModelNeuron, ModelNeuronGraph
from newera.neuron.cells.aggregation import GroupRoleDeepSetsAggregator
from newera.neuron.graph import LinkMutation, NeuronGraphTopology, TopologyController

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

def _grouped_model() -> ModelNeuron:
    torch.manual_seed(991)
    return ModelNeuron(3, 8, 5, num_edge_types=2, message_aggregator=GroupRoleDeepSetsAggregator(8, 2))


@pytest.mark.parametrize("device_name", DEVICES)
def test_graph_passes_edge_groups_to_same_model_neuron_step(device_name: str):
    device = torch.device(device_name)
    neuron = _grouped_model().to(device)
    graph = ModelNeuronGraph(neuron)
    group_id = 2**60 + 41
    topology = NeuronGraphTopology(
        batch_size=1,
        nodes_per_graph=3,
        edge_index=torch.tensor([[0, 1], [2, 2]], device=device),
        edge_type=torch.tensor([0, 1], device=device),
        strength=torch.ones(2, device=device),
        node_ids=torch.tensor([[10, 20, 30]], device=device),
        edge_group_ids=torch.tensor([group_id, group_id], device=device),
    )
    inputs = torch.randn(2, 3, 3, device=device)
    first = graph(inputs, topology)
    actual = graph(inputs, topology, state=first.state).values[:, 2]
    expected = neuron.step(
        inputs[:, 2],
        incoming_messages=first.values[:, [0, 1]],
        incoming_strengths=torch.ones(2, 2, device=device),
        incoming_edge_types=torch.tensor([[0, 1], [0, 1]], device=device),
        incoming_group_ids=torch.full((2, 2), group_id, dtype=torch.long, device=device),
        state=first.state.neuron_states[:, 2],
    ).value
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device_name", DEVICES)
def test_graph_candidate_edges_accept_group_ids(device_name: str):
    device = torch.device(device_name)
    graph = ModelNeuronGraph(_grouped_model().to(device))
    topology = NeuronGraphTopology.empty(
        1, 3, device=device, node_ids=torch.tensor([[10, 20, 30]], device=device)
    )
    inputs = torch.randn(1, 3, 3, device=device)
    first = graph(inputs, topology)
    result = graph(
        inputs,
        topology,
        state=first.state,
        candidate_edges=torch.tensor([[0, 1], [2, 2]], device=device),
        candidate_type_weights=torch.ones(2, 2, device=device),
        candidate_group_ids=torch.tensor([2**60, 2**60], device=device),
    )
    assert result.values.shape == (1, 3, 5)
    with pytest.raises(ValueError, match="require candidate_group_ids"):
        graph(
            inputs,
            topology,
            state=first.state,
            candidate_edges=torch.tensor([[0], [2]], device=device),
            candidate_type_weights=torch.ones(1, 2, device=device),
        )


def test_group_ids_survive_node_remapping_and_link_mutations():
    topology = NeuronGraphTopology(
        batch_size=1,
        nodes_per_graph=3,
        edge_index=torch.tensor([[0, 1], [2, 2]]),
        edge_type=torch.tensor([0, 1]),
        strength=torch.ones(2),
        node_ids=torch.tensor([[10, 20, 30]]),
        edge_group_ids=torch.tensor([100, 200]),
    )
    remapped = topology.remap_node_ids(torch.tensor([[30, 10, 20]]))
    assert remapped.edge_group_ids.tolist() == [100, 200]
    assert remapped.edge_index.tolist() == [[1, 2], [0, 0]]
    filtered = topology.filter_context(attention_mask=torch.tensor([[True, False, True]]))
    assert filtered.edge_group_ids.tolist() == [100]
    extended = topology.append_causal_nodes(count=1, node_ids=torch.tensor([[40]]))
    assert extended.edge_group_ids.tolist() == [100, 200, 0]
    changed = TopologyController(causal=False).apply(
        topology,
        [
            LinkMutation("add", 0, 0, 1, edge_type=1, group_id=2**60),
            LinkMutation("retype", 0, 0, 2, edge_type=1),
        ],
        num_edge_types=2,
    )
    assert changed.edge_group_ids.tolist() == [100, 200, 2**60]
