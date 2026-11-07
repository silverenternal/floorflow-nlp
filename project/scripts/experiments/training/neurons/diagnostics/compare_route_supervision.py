"""Compare task-only learning, structural action labels, and hard calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[6]
NEURON_SCRIPTS = ROOT / "project/scripts/experiments/training/neurons"
sys.path.insert(0, str(NEURON_SCRIPTS))

import run_modelized_neuron_graph_validation as graph_validation

from newera.neuron.graph import NeuronIdentityRegistry


def _action_targets(sample: dict[str, Any], output: dict[str, Any], edge_types: int) -> Tensor:
    ids = sample["node_ids"][0]
    readout_id = max(int(identity) for identity in ids.tolist())
    oracle_by_source = {
        source: edge_type for source, edge_type, _strength in sample["oracle_edges"][0]
    }
    existing = {
        (int(source), int(target)): int(edge_type)
        for (source, target), edge_type in zip(
            output["topology"].edge_index.t().detach().cpu().tolist(),
            output["topology"].edge_type.detach().cpu().tolist(),
            strict=True,
        )
    }
    candidates = output["candidate_edges"].detach().cpu().t().tolist()
    targets = []
    for source_slot, target_slot in candidates:
        source_id = int(ids[source_slot].item())
        target_id = int(ids[target_slot].item())
        if target_id != readout_id:
            raise ValueError("candidate set unexpectedly contains a non-readout target")
        if source_id in oracle_by_source:
            desired_type = oracle_by_source[source_id]
            current_type = existing.get((source_slot, target_slot))
            targets.append(0 if current_type == desired_type else desired_type + 1)
        elif (source_slot, target_slot) in existing:
            targets.append(edge_types + 1)
        else:
            targets.append(0)
    return torch.tensor(targets, dtype=torch.long, device=output["logits"].device)


def _build_data(config: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, int], int]:
    registry = NeuronIdentityRegistry(config["task"]["identity_namespace"])
    identities = {key: registry.register(f"memory/{key}") for key in range(config["task"]["key_space"])}
    readout_identity = registry.register("readout")
    return graph_validation._data(
        92_001,
        32,
        memory_count=4,
        task="single_link_retrieval",
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    ), identities, readout_identity


def _train_condition(
    name: str,
    train: dict[str, Any],
    validation: dict[str, Any],
    config: dict[str, Any],
    *,
    steps: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], nn.Module]:
    feature_size = train["features"].shape[-1]
    hidden = config["model"]["hidden_size"]
    edge_types = config["model"]["edge_types"]
    torch.manual_seed(seed)
    model = graph_validation.SharedWeightNeuronTask(feature_size, hidden, edge_types).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    generator = torch.Generator().manual_seed(seed + 1)
    action_counts = torch.zeros(edge_types + 2, dtype=torch.float32, device=device)
    with torch.no_grad():
        model.eval()
        for index in range(train["features"].shape[0]):
            sample = graph_validation.graph_proxy._subset(train, torch.tensor([index], dtype=torch.long))
            output = model(sample, hard=False)
            action_counts += torch.bincount(
                _action_targets(sample, output, edge_types), minlength=edge_types + 2
            ).to(device=device, dtype=torch.float32)
    class_weights = torch.zeros_like(action_counts)
    active_classes = action_counts > 0
    class_weights[active_classes] = action_counts.sum() / (
        int(active_classes.sum().item()) * action_counts[active_classes]
    )
    for step in range(steps):
        sample_index = int(torch.randint(train["features"].shape[0], (1,), generator=generator).item())
        sample = graph_validation.graph_proxy._subset(train, torch.tensor([sample_index], dtype=torch.long))
        model.train()
        model.hard_action_active = (
            name == "hard_calibrated" and step >= int(steps * 0.7)
        ) or (name == "production_control" and step >= int(steps * 0.85))
        output = model(sample, hard=False)
        task_loss = F.mse_loss(output["prediction"], sample["labels"])
        loss = task_loss
        if name in {"structure_supervised"}:
            action_targets = _action_targets(sample, output, edge_types)
            structural_loss = F.cross_entropy(
                output["logits"], action_targets, weight=class_weights
            )
            loss = task_loss + structural_loss
        elif name in {"sparsity_only", "production_control"}:
            loss = task_loss + 0.001 * output["candidate_weights"].abs().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    threshold, threshold_trials = graph_validation._select_threshold(
        model, validation, config, seed
    )
    metrics = graph_validation._metrics(model, validation, threshold=threshold, seed=seed)
    metrics.pop("_records")
    mean_prediction = float(train["labels"].mean().item())
    mean_mse = float((validation["labels"] - mean_prediction).square().mean().item())
    deltas: list[float] = []
    deletion_delta_mses: list[float] = []
    mutations = 0
    actions = torch.zeros(edge_types + 2, dtype=torch.long)
    with torch.no_grad():
        for index in range(validation["features"].shape[0]):
            sample = graph_validation.graph_proxy._subset(validation, torch.tensor([index], dtype=torch.long))
            soft = model(sample, hard=False)
            actions += torch.bincount(soft["logits"].argmax(-1).cpu(), minlength=edge_types + 2)
            hard = model(sample, hard=True, threshold=threshold)
            mutations += len(hard["mutations"])
            topology = hard["topology"]
            if topology.edge_count:
                initial = model.neuron_graph(sample["features"], graph_validation._topology(sample))
                readout_slot = int(sample["node_ids"][0].argmax().item())
                reference = hard["prediction"].reshape(-1)[0]
                for edge_index in range(topology.edge_count):
                    keep = torch.ones(topology.edge_count, dtype=torch.bool, device=device)
                    keep[edge_index] = False
                    reduced = type(topology)(
                        topology.batch_size,
                        topology.nodes_per_graph,
                        topology.edge_index[:, keep],
                        topology.edge_type[keep],
                        topology.strength[keep],
                        topology.node_ids,
                    )
                    changed = model.neuron_graph(sample["features"], reduced, state=initial.state)
                    changed_prediction = model.readout(changed.values[:, readout_slot]).reshape(())
                    deltas.append(float((reference - changed_prediction).abs().item()))
                    target = sample["labels"].reshape(-1)[0]
                    deletion_delta_mses.append(
                        float((changed_prediction - target).square().item() - (reference - target).square().item())
                    )
    params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "validation": metrics,
        "validation_train_mean_mse": mean_mse,
        "selected_threshold": threshold,
        "threshold_trials": threshold_trials,
        "candidate_action_counts_keep_type0_type1_remove": actions.tolist(),
        "training_action_target_counts_keep_type0_type1_remove": action_counts.cpu().tolist(),
        "class_balancing_weights": class_weights.cpu().tolist(),
        "hard_mutation_count": mutations,
        "mean_single_edge_deletion_abs_prediction_change": sum(deltas) / len(deltas) if deltas else None,
        "mean_single_edge_deletion_delta_mse": (
            sum(deletion_delta_mses) / len(deletion_delta_mses) if deletion_delta_mses else None
        ),
        "fraction_edge_deletions_that_worsen_mse": (
            sum(delta > 0 for delta in deletion_delta_mses) / len(deletion_delta_mses)
            if deletion_delta_mses
            else None
        ),
        "model_parameters": params,
        "training_updates": steps,
    }, model


def run(output: Path, config_path: Path, *, steps: int = 160, device_name: str = "mps") -> dict[str, Any]:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required by this experiment; fallback is disabled")
    config = json.loads(config_path.read_text())
    device = torch.device(device_name)
    train, identities, readout_identity = _build_data(config, device)
    validation = graph_validation._data(
        92_002,
        64,
        memory_count=4,
        task="single_link_retrieval",
        config=config,
        device=device,
        identities=identities,
        readout_identity=readout_identity,
    )
    conditions = {}
    models = {}
    for name in (
        "production_control",
        "task_only",
        "sparsity_only",
        "structure_supervised",
        "hard_calibrated",
    ):
        conditions[name], models[name] = _train_condition(
            name,
            train,
            validation,
            config,
            steps=steps,
            device=device,
            seed=7_401,
        )
    input_size = (config["task"]["key_space"] + 1) * train["features"].shape[-1]
    largest_graph = models["task_only"]
    target_parameters = sum(parameter.numel() for parameter in largest_graph.parameters())
    width = graph_validation.graph_proxy._closest_width(target_parameters, input_size, parameters=True)
    control = graph_validation.FlatControl(input_size, width).to(device)
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.004)
    generator = torch.Generator().manual_seed(7_402)
    for _ in range(steps):
        index = int(torch.randint(train["features"].shape[0], (1,), generator=generator).item())
        sample = graph_validation.graph_proxy._subset(train, torch.tensor([index], dtype=torch.long))
        inputs = graph_validation.graph_proxy._canonical_features(
            sample["features"], sample["node_ids"], config["task"]["key_space"]
        )
        loss = F.mse_loss(control(inputs), sample["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    control.eval()
    with torch.no_grad():
        val_inputs = graph_validation.graph_proxy._canonical_features(
            validation["features"], validation["node_ids"], config["task"]["key_space"]
        )
        control_mse = float(F.mse_loss(control(val_inputs), validation["labels"]).item())
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
        "task": "single_link_retrieval",
        "train_seed": 92_001,
        "validation_seed": 92_002,
        "training_sample_count": train["features"].shape[0],
        "validation_sample_count": validation["features"].shape[0],
        "steps_per_condition": steps,
        "sealed_test_used": False,
        "parameter_matched_flat_mlp": {
            "parameters": sum(parameter.numel() for parameter in control.parameters()),
            "target_parameters": target_parameters,
            "validation_mse": control_mse,
        },
        "conditions": conditions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=graph_validation.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.config, steps=args.steps, device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
