from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
RUNNER_DIR = ROOT / "project/scripts/experiments/training/neurons"
sys.path.insert(0, str(RUNNER_DIR))
import run_modelized_neuron_graph_validation as validation

from newera.neuron import ModelNeuron, ModelNeuronGraph
from newera.neuron.graph import NeuronGraphTopology


def test_empty_graph_jaccard_is_permutation_identity_and_separately_detectable() -> None:
    assert validation._jaccard(set(), set()) == 1.0
    assert validation._jaccard({(1, 2)}, set()) == 0.0
    assert validation._relative_error(0.0, 0.1, 1e-12) is None
    assert validation._relative_error(2.0, 3.0, 1e-12) == 0.5


def test_pairwise_edge_f1_is_perfect_for_two_empty_graphs() -> None:
    data = {
        "node_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "oracle_edges": [[]],
    }
    topology = validation.NeuronGraphTopology(
        1,
        2,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.float32),
        data["node_ids"],
    )
    assert validation._topology_edge_f1(data, topology) == 1.0


def test_zero_inputs_and_empty_incoming_edges_keep_model_neuron_finite() -> None:
    model = ModelNeuronGraph(ModelNeuron(input_size=3, hidden_size=4, message_size=4, num_edge_types=2))
    features = torch.zeros((1, 2, 3))
    node_ids = torch.tensor([[10, 11]], dtype=torch.long)
    topology = NeuronGraphTopology(
        1,
        2,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.float32),
        node_ids,
    )
    output = model(features, topology)
    assert torch.isfinite(output.values).all()
    assert output.state.neuron_states.shape == (1, 2, 4)
    output.values.square().mean().backward()
    assert model.neuron.input_projection.weight.grad is not None


def test_graph_runner_resume_matches_uninterrupted_training(tmp_path) -> None:
    key_space = 8
    identities = {key: key + 1 for key in range(key_space)}
    raw = validation.graph_proxy._data(
        811,
        8,
        memory_count=4,
        task="single_link_retrieval",
        key_space=key_space,
        device=torch.device("cpu"),
        identity_ids=identities,
        readout_identity=key_space + 1,
    )
    raw["key_space"] = key_space
    torch.testing.assert_close(
        validation._oracle_targets(raw, key_space), raw["labels"], atol=1e-6, rtol=0
    )
    input_size = (key_space + 1) * raw["features"].shape[-1]
    metadata = {"task": "single_link_retrieval", "seed": 811, "model": "flat-control"}

    torch.manual_seed(17)
    uninterrupted = validation.FlatControl(input_size, 4)
    validation._train(
        uninterrupted,
        raw,
        steps=2,
        batch_size=2,
        lr=0.01,
        seed=199,
        device=torch.device("cpu"),
        hard_action_fraction=0.0,
    )

    torch.manual_seed(17)
    resumed = validation.FlatControl(input_size, 4)
    checkpoint = tmp_path / "neuron-run.pt"
    validation._train(
        resumed,
        raw,
        steps=1,
        batch_size=2,
        lr=0.01,
        seed=199,
        device=torch.device("cpu"),
        hard_action_fraction=0.0,
        checkpoint_path=checkpoint,
        checkpoint_interval_steps=1,
        checkpoint_metadata=metadata,
        overfit_probe_steps=5,
    )
    resumed_summary = validation._train(
        resumed,
        raw,
        steps=2,
        batch_size=2,
        lr=0.01,
        seed=199,
        device=torch.device("cpu"),
        hard_action_fraction=0.0,
        checkpoint_path=checkpoint,
        checkpoint_interval_steps=1,
        checkpoint_metadata=metadata,
        resume=True,
    )[1]
    for key, expected in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], expected)
    assert resumed_summary["samples_seen"] == 4
    assert resumed_summary["effective_batch_size"] == 2
    assert resumed_summary["gradient_coverage"]["gradient_connected_count"] == 4
    assert resumed_summary["single_sample_overfit_probe"]["optimizer_steps"] == 5
    assert resumed_summary["single_sample_overfit_probe"]["final_mse"] < resumed_summary["single_sample_overfit_probe"]["initial_mse"]
