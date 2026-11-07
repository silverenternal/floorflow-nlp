from __future__ import annotations

import torch

from newera.neuron.graph import (
    LinkMutation,
    LinkProposalPolicy,
    NeuronGraphTopology,
    NeuronIdentityRegistry,
    TopologyController,
    drop_topology_edges,
    rewire_topology_matched,
)


def test_matched_rewire_preserves_target_degree_edge_attributes_and_segments() -> None:
    segments = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    node_ids = torch.arange(100, 112).unsqueeze(0)
    topology = NeuronGraphTopology(
        1,
        12,
        torch.tensor([[0, 1, 6, 7, 8], [1, 3, 9, 9, 10]]),
        torch.tensor([0, 1, 1, 0, 2]),
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        node_ids,
    )

    first = rewire_topology_matched(topology, segment_ids=segments, seed=7)
    second = rewire_topology_matched(topology, segment_ids=segments, seed=7)

    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_type, second.edge_type)
    assert torch.equal(first.strength, second.strength)
    assert first.edge_count == topology.edge_count
    assert torch.equal(first.node_ids, node_ids)
    assert torch.equal(first.edge_index[1].bincount(minlength=12), topology.edge_index[1].bincount(minlength=12))
    source_batch = first.edge_index[0] // 12
    source_pos = first.edge_index[0] % 12
    target_pos = first.edge_index[1] % 12
    assert torch.all(source_pos < target_pos)
    assert torch.equal(segments[source_batch, source_pos], segments[source_batch, target_pos])
    for target in (9, 10):
        selected = first.edge_index[1] == target
        assert sorted(first.edge_type[selected].tolist()) == sorted(
            topology.edge_type[topology.edge_index[1] == target].tolist()
        )
        assert sorted(first.strength[selected].tolist()) == sorted(
            topology.strength[topology.edge_index[1] == target].tolist()
        )


def test_matched_rewire_rejects_cross_segment_topology() -> None:
    topology = NeuronGraphTopology(
        1,
        4,
        torch.tensor([[0], [3]]),
        torch.zeros(1, dtype=torch.long),
        torch.ones(1),
    )
    try:
        rewire_topology_matched(
            topology, segment_ids=torch.tensor([[0, 0, 1, 1]]), seed=1
        )
    except ValueError as error:
        assert "cross-segment" in str(error)
    else:
        raise AssertionError("cross-segment topology must be rejected")


def test_edge_deletion_is_deterministic_and_preserves_surviving_attributes() -> None:
    topology = NeuronGraphTopology.causal_chain(1, 8, device="cpu")
    topology = NeuronGraphTopology(
        topology.batch_size,
        topology.nodes_per_graph,
        topology.edge_index,
        torch.arange(topology.edge_count) % 2,
        torch.linspace(0.1, 0.8, topology.edge_count),
        torch.arange(8).unsqueeze(0),
    )

    first = drop_topology_edges(topology, fraction=0.5, seed=13)
    second = drop_topology_edges(topology, fraction=0.5, seed=13)
    all_dropped = drop_topology_edges(topology, fraction=1.0, seed=13)

    assert first.edge_count == topology.edge_count - round(topology.edge_count * 0.5)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_type, second.edge_type)
    assert torch.equal(first.strength, second.strength)
    assert all_dropped.edge_count == 0



def test_topology_controller_updates_strength_without_changing_edge_type() -> None:
    topology = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0], [2]]),
        torch.tensor([1]),
        torch.tensor([0.25]),
        torch.tensor([[10, 20, 30]]),
    )
    updated = TopologyController(causal=False).apply(
        topology,
        [LinkMutation("set_strength", 0, 0, 2, strength=0.8)],
        num_edge_types=2,
    )
    assert updated.edge_type.tolist() == [1]
    assert torch.allclose(updated.strength, torch.tensor([0.8]))


def test_policy_predicts_and_proposes_strength_updates() -> None:
    policy = LinkProposalPolicy(4, 1, initial_keep_bias=0.0, predict_strengths=True)
    topology = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0], [2]]),
        torch.tensor([0]),
        torch.tensor([0.2]),
        torch.tensor([[10, 20, 30]]),
    )
    nodes = torch.randn(1, 3, 4)
    candidates = torch.tensor([[0], [2]])
    with torch.no_grad():
        policy.scorer[-1].weight.zero_()
        policy.scorer[-1].bias.copy_(torch.tensor([5.0, 0.0, -1.0]))
        policy.strength_scorer[-1].weight.zero_()
        policy.strength_scorer[-1].bias.fill_(2.0)
    logits = policy(nodes, candidates, topology, causal=False)
    strengths = policy.predict_strengths(nodes, candidates, topology, causal=False)
    mutations = policy.propose(
        logits, candidates, topology, threshold=0.5, candidate_strengths=strengths
    )
    assert len(mutations) == 1
    assert mutations[0].operation == "set_strength"
    updated = TopologyController(causal=False).apply(
        topology, mutations, num_edge_types=1
    )
    assert updated.edge_type.tolist() == [0]
    assert updated.strength[0].item() > topology.strength[0].item()


def test_policy_straight_through_weights_have_hard_forward_and_soft_gradient() -> None:
    policy = LinkProposalPolicy(2, 2)
    topology = NeuronGraphTopology.empty(1, 2, device='cpu')
    logits = torch.tensor([[0.0, 3.0, 1.0, -1.0]], requires_grad=True)
    candidates = torch.tensor([[0], [1]])
    strengths = torch.tensor([0.7])

    weights = policy.expected_type_weights(
        logits,
        candidates,
        topology,
        candidate_strengths=strengths,
        straight_through=True,
    )
    assert torch.allclose(weights, torch.tensor([[0.7, 0.0]]))
    weights.sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


def test_identity_registry_atomic_round_trip_preserves_and_extends_ids(tmp_path) -> None:
    registry = NeuronIdentityRegistry(namespace=23)
    first_id = registry.register("conversation/4/token/12")
    assert registry.register("conversation/4/token/12") == first_id
    registry.save(tmp_path / "identity" / "registry.json")

    restored = NeuronIdentityRegistry.load(tmp_path / "identity" / "registry.json")

    assert restored.resolve("conversation/4/token/12") == first_id
    second_id = restored.register("conversation/4/token/13")
    assert second_id > first_id
    assert second_id >> 48 == 23


def test_identity_registry_rejects_wrong_namespace_and_reused_counter() -> None:
    registry = NeuronIdentityRegistry(namespace=3)
    registry.register("a")
    state = registry.state_dict()
    state["entries"] = {"a": 1}
    try:
        NeuronIdentityRegistry.from_state_dict(state)
    except ValueError as error:
        assert "invalid or duplicated" in str(error)
    else:
        raise AssertionError("a registry must reject IDs from another namespace")
