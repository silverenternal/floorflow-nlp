"""Compare gradient coverage across fixed, soft, and hard graph routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[6]
NEURON_SCRIPTS = ROOT / "project/scripts/experiments/training/neurons"
sys.path.insert(0, str(NEURON_SCRIPTS))

import run_modelized_neuron_graph_validation as graph_validation

from newera.neuron import ModelNeuron, ModelNeuronGraph
from newera.neuron.graph import NeuronGraphTopology, NeuronIdentityRegistry


def _gradient_summary(model: nn.Module) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("link_policy."):
            group = "policy"
        elif name.startswith("neuron_graph.neuron."):
            group = "shared_neuron"
        elif name.startswith("readout."):
            group = "readout"
        else:
            group = "model"
        stats = groups.setdefault(group, {"parameters": 0, "connected": 0, "finite": 0, "nonzero": 0, "l2_norm": 0.0})
        stats["parameters"] += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        stats["connected"] += 1
        if torch.isfinite(gradient).all():
            stats["finite"] += 1
        if torch.count_nonzero(gradient):
            stats["nonzero"] += 1
        stats["l2_norm"] += float(gradient.detach().float().square().sum().sqrt().item())
    return groups


def _fixed_oracle_topology(sample: dict[str, Any]) -> tuple[NeuronGraphTopology, int]:
    ids = sample["node_ids"][0]
    identity_to_slot = {int(identity): slot for slot, identity in enumerate(ids.tolist())}
    readout_identity = max(identity_to_slot)
    readout_slot = identity_to_slot[readout_identity]
    source_slots = [identity_to_slot[source] for source, _, _ in sample["oracle_edges"][0]]
    edge_types = [edge_type for _, edge_type, _ in sample["oracle_edges"][0]]
    strengths = [strength for _, _, strength in sample["oracle_edges"][0]]
    topology = NeuronGraphTopology(
        batch_size=1,
        nodes_per_graph=ids.numel(),
        edge_index=torch.tensor([source_slots, [readout_slot] * len(source_slots)], dtype=torch.long, device=ids.device),
        edge_type=torch.tensor(edge_types, dtype=torch.long, device=ids.device),
        strength=torch.tensor(strengths, dtype=sample["initial_strength"].dtype, device=ids.device),
        node_ids=sample["node_ids"],
    )
    return topology, readout_slot


def _single_neuron_gradient(sample: dict[str, Any], hidden_size: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    readout_slot = sample["node_ids"].argmax(dim=1, keepdim=True)
    feature = sample["features"].gather(
        1,
        readout_slot.unsqueeze(-1).expand(-1, -1, sample["features"].shape[-1]),
    ).squeeze(1)
    neuron = ModelNeuron(feature.shape[-1], hidden_size, hidden_size).to(feature.device)
    readout = nn.Linear(hidden_size, 1).to(feature.device)
    result = neuron.step(
        feature,
        incoming_messages=feature.new_zeros((1, 0, hidden_size)),
        incoming_strengths=feature.new_zeros((1, 0)),
        incoming_edge_types=torch.empty((1, 0), dtype=torch.long, device=feature.device),
    )
    prediction = readout(result.value).reshape_as(sample["labels"])
    loss = F.mse_loss(prediction, sample["labels"])
    loss.backward()
    return {
        "loss": float(loss.detach().item()),
        "prediction": float(prediction.detach().item()),
        "gradient_coverage": _gradient_summary(nn.ModuleList([neuron, readout])),
    }


def _fixed_oracle_gradient(sample: dict[str, Any], feature_size: int, hidden_size: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    graph = ModelNeuronGraph(ModelNeuron(feature_size, hidden_size, hidden_size, num_edge_types=2)).to(sample["features"].device)
    readout = nn.Linear(hidden_size, 1).to(sample["features"].device)
    topology, slot = _fixed_oracle_topology(sample)
    output = graph(sample["features"], topology)
    prediction = readout(output.values[:, slot]).reshape_as(sample["labels"])
    loss = F.mse_loss(prediction, sample["labels"])
    loss.backward()
    return {
        "loss": float(loss.detach().item()),
        "prediction": float(prediction.detach().item()),
        "oracle_edge_count": topology.edge_count,
        "gradient_coverage": _gradient_summary(nn.ModuleList([graph, readout])),
    }


def _policy_condition(
    sample: dict[str, Any], *, hidden_size: int, edge_types: int, seed: int, mode: str
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = graph_validation.SharedWeightNeuronTask(sample["features"].shape[-1], hidden_size, edge_types).to(sample["features"].device)
    if mode == "straight_through_hard":
        model.train()
        model.hard_action_active = True
        output = model(sample, hard=False)
    elif mode == "discrete_hard":
        model.eval()
        output = model(sample, hard=True, threshold=0.4)
    else:
        model.train()
        model.hard_action_active = False
        output = model(sample, hard=False)
    prediction = output["prediction"].reshape_as(sample["labels"])
    loss = F.mse_loss(prediction, sample["labels"])
    loss.backward()
    actions = output["logits"].detach().argmax(dim=-1)
    action_counts = torch.bincount(actions, minlength=edge_types + 2).cpu().tolist()
    result: dict[str, Any] = {
        "loss": float(loss.detach().item()),
        "prediction": float(prediction.detach().item()),
        "action_counts_keep_connect_type0_connect_type1_remove": action_counts,
        "gradient_coverage": _gradient_summary(model),
    }
    if mode == "discrete_hard":
        result["hard_mutation_count"] = len(output["mutations"])
        result["hard_mutation_types"] = [mutation.operation for mutation in output["mutations"]]
    return result


def run(output_path: Path, config_path: Path, *, device_name: str = "mps") -> dict[str, Any]:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required by this diagnostic; fallback is disabled")
    config = json.loads(config_path.read_text())
    device = torch.device(device_name)
    registry = NeuronIdentityRegistry(config["task"]["identity_namespace"])
    identities = {key: registry.register(f"memory/{key}") for key in range(config["task"]["key_space"])}
    readout_identity = registry.register("readout")
    data = graph_validation._data(
        91_001,
        4,
        memory_count=4,
        task="single_link_retrieval",
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    )
    sample = graph_validation.graph_proxy._subset(data, torch.tensor([0], dtype=torch.long))
    hidden = config["model"]["hidden_size"]
    edge_types = config["model"]["edge_types"]
    seed = 7_301
    comparisons = {
        "single_neuron": _single_neuron_gradient(sample, hidden, seed),
        "fixed_oracle_route": _fixed_oracle_gradient(sample, sample["features"].shape[-1], hidden, seed),
        "soft_candidate_route": _policy_condition(sample, hidden_size=hidden, edge_types=edge_types, seed=seed, mode="soft"),
        "straight_through_hard_forward": _policy_condition(sample, hidden_size=hidden, edge_types=edge_types, seed=seed, mode="straight_through_hard"),
        "discrete_hard_inference": _policy_condition(sample, hidden_size=hidden, edge_types=edge_types, seed=seed, mode="discrete_hard"),
    }
    predictions = [row["prediction"] for row in comparisons.values()]
    result = {
        "device": device_name,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                config_path,
                ROOT / "src/newera/neuron/composition.py",
                ROOT / "src/newera/neuron/graph/policy.py",
                ROOT / "src/newera/neuron/graph/layer.py",
            )
        },
        "data": {"task": "single_link_retrieval", "seed": 91_001, "samples": 4, "diagnostic_sample_index": 0},
        "target": float(sample["labels"].item()),
        "comparisons": comparisons,
        "cross_path_prediction_variance": float(torch.tensor(predictions).var(unbiased=False).item()),
        "scope": "single-step diagnostic only; no model selection or sealed-test use",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=graph_validation.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.config, device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
