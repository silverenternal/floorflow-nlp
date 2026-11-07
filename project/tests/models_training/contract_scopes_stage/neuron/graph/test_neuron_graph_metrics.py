from __future__ import annotations

import pytest
import torch

from newera.neuron.graph import (
    DynamicNeuronGraphLayer,
    NeuronGraphTopology,
    action_distribution,
    edge_set_jaccard,
    normalized_effective_rank,
    relative_dirichlet_energy,
    sampled_causal_gradient_transport,
    topology_metrics,
    weighted_dirichlet_energy,
)


def test_topology_metrics_count_only_valid_within_segment_causal_pairs() -> None:
    topology = NeuronGraphTopology(
        1,
        6,
        torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        torch.zeros(5, dtype=torch.long),
        torch.ones(5),
    )
    metrics = topology_metrics(
        topology,
        attention_mask=torch.tensor([[False, True, True, True, True, True]]),
        segment_ids=torch.tensor([[-1, 0, 0, 1, 1, 1]]),
        max_edges=8,
    )

    assert metrics["valid_node_count"] == 5
    assert metrics["edge_count"] == 3
    assert metrics["edge_density"] == pytest.approx(3 / 4)
    assert metrics["edge_budget_utilization"] == pytest.approx(3 / 8)
    assert metrics["max_weighted_indegree"] == pytest.approx(1.0)


def test_layer_degree_diagnostics_exclude_padding_and_invalid_segments() -> None:
    topology = NeuronGraphTopology(
        1,
        4,
        torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2),
    )
    result = DynamicNeuronGraphLayer(4, 1)(
        torch.randn(1, 4, 4),
        topology,
        attention_mask=torch.tensor([[False, True, True, True]]),
        segment_ids=torch.tensor([[-1, 0, 0, 0]]),
    )

    assert result.diagnostics["valid_node_count"] == 3
    assert result.diagnostics["edge_count"] == 1
    assert result.diagnostics["mean_weighted_indegree"] == pytest.approx(1 / 3)


def test_effective_rank_is_normalized_and_detects_collapsed_features() -> None:
    collapsed = torch.ones(1, 5, 4)
    diverse = torch.eye(4).unsqueeze(0)

    assert normalized_effective_rank(collapsed) == pytest.approx(0.25, abs=1e-5)
    assert normalized_effective_rank(diverse) == pytest.approx(1.0, abs=1e-5)
    assert normalized_effective_rank(diverse, valid_mask=torch.tensor([[True, True, False, False]])) > 0


def test_weighted_dirichlet_energy_uses_directed_edge_list_weights() -> None:
    hidden = torch.tensor([[[0.0], [2.0], [6.0]]])
    edges = torch.tensor([[0, 1], [1, 2]])
    energy = weighted_dirichlet_energy(hidden, edges, torch.tensor([1.0, 3.0]))

    assert energy == pytest.approx((1.0 * 4.0 + 3.0 * 16.0) / 4.0)
    relative_energy = relative_dirichlet_energy(hidden, edges, torch.tensor([1.0, 3.0]))
    scaled_relative_energy = relative_dirichlet_energy(hidden * 5, edges, torch.tensor([1.0, 3.0]))
    assert relative_energy == pytest.approx(scaled_relative_energy)
    assert weighted_dirichlet_energy(hidden, torch.empty((2, 0), dtype=torch.long), torch.empty(0)) == 0


def test_action_distribution_reports_uncertainty_and_each_action_fraction() -> None:
    logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    metrics = action_distribution(logits)

    assert metrics["action_count"] == 2
    assert metrics["action_0_fraction"] == pytest.approx(0.5)
    assert metrics["action_1_fraction"] == pytest.approx(0.5)
    assert metrics["action_confidence"] > 0.99


def test_edge_set_jaccard_requires_stable_node_identity() -> None:
    ids = torch.tensor([[10, 11, 12]])
    first = NeuronGraphTopology.causal_chain(1, 3, device="cpu", node_ids=ids)
    second = NeuronGraphTopology(
        1,
        3,
        torch.tensor([[0, 0], [1, 2]]),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2),
        ids,
    )
    assert edge_set_jaccard(first, second) == pytest.approx(1 / 3)

    changed_ids = NeuronGraphTopology.causal_chain(
        1, 3, device="cpu", node_ids=torch.tensor([[10, 99, 12]])
    )
    with pytest.raises(ValueError, match="identical node identities"):
        edge_set_jaccard(first, changed_ids)
    with pytest.raises(ValueError, match="requires stable node identities"):
        edge_set_jaccard(
            NeuronGraphTopology.causal_chain(1, 3, device="cpu"),
            NeuronGraphTopology.causal_chain(1, 3, device="cpu"),
        )


def test_sampled_gradient_transport_measures_causal_source_influence() -> None:
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    weights = torch.tensor([[0.0], [0.0], [2.0], [0.0]])
    loss = (hidden * weights).sum()
    metrics = sampled_causal_gradient_transport(
        hidden,
        loss,
        source_indices=torch.tensor([0, 2]),
        target_indices=torch.tensor([3, 3]),
        nodes_per_graph=4,
    )

    assert metrics["sample_count"] == 2
    assert metrics["mean_causal_distance"] == pytest.approx(2.0)
    assert metrics["mean_gradient_norm"] == pytest.approx(3**0.5)
    assert metrics["gradient_norm_by_distance"]["1-3"]["sample_count"] == 2
    assert metrics["gradient_norm_by_distance"]["4-7"]["sample_count"] == 0
