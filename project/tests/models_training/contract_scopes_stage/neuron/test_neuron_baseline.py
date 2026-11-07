from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from newera.neuron import MLPNeuron, Neuron

ROOT = Path(__file__).resolve().parents[5]


def test_mlp_neuron_is_a_stateless_trainable_contract_implementation() -> None:
    neuron = MLPNeuron(2, 8, 1)
    inputs = torch.randn(4, 2)

    assert isinstance(neuron, Neuron)
    result = neuron(inputs)
    assert result.value.shape == (4, 1)
    assert result.state is None

    result.value.square().mean().backward()
    assert all(parameter.grad is not None for parameter in neuron.parameters())


def test_mlp_neuron_learns_a_small_synthetic_mapping_on_cpu() -> None:
    torch.manual_seed(31)
    inputs = torch.randn(64, 2, device="cpu")
    targets = (0.5 * inputs[:, :1]) - (1.2 * inputs[:, 1:2]) + 0.4
    neuron = MLPNeuron(2, 16, 1)
    optimizer = torch.optim.Adam(neuron.parameters(), lr=0.03)
    initial_loss = torch.nn.functional.mse_loss(neuron(inputs).value, targets)

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(neuron(inputs).value, targets)
        loss.backward()
        optimizer.step()

    final_loss = torch.nn.functional.mse_loss(neuron(inputs).value, targets)
    assert initial_loss.device.type == "cpu"
    assert torch.isfinite(final_loss)
    assert final_loss < initial_loss * 0.1
    assert sum(parameter.numel() for parameter in neuron.parameters()) == 65


def test_mlp_neuron_rejects_implicit_shape_or_state_mismatches() -> None:
    neuron = MLPNeuron(2, 4, 1)

    with pytest.raises(ValueError, match="final dimension"):
        neuron(torch.ones(3))
    with pytest.raises(ValueError, match="stateless"):
        neuron.step(torch.ones(1, 2), state=torch.zeros(1, 2))


def test_synthetic_training_checkpoint_recovers_continuously(tmp_path: Path) -> None:
    script = ROOT / "project/scripts/experiments/training/neurons/capabilities/run_neuron_smoke.py"

    def run(steps: int, checkpoint: Path, *, resume: bool = False) -> dict[str, object]:
        command = [
            sys.executable,
            str(script),
            "--steps",
            str(steps),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            "37",
        ]
        if resume:
            command.append("--resume")
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    continuous = run(80, tmp_path / "continuous.pt")
    checkpoint = tmp_path / "resumed.pt"
    run(35, checkpoint)
    resumed = run(45, checkpoint, resume=True)

    assert resumed["completed_steps"] == continuous["completed_steps"] == 80
    assert resumed["model_sha256"] == continuous["model_sha256"]
    assert resumed["train_mse"] == pytest.approx(continuous["train_mse"], abs=1e-8)
    assert resumed["validation_mse"] == pytest.approx(continuous["validation_mse"], abs=1e-8)


def test_regression_capability_cli_trains_and_evaluates_cpu() -> None:
    script = ROOT / "project/scripts/experiments/training/neurons/capabilities/run_regression_capability.py"
    result = subprocess.run(
        [sys.executable, str(script), "--seed", "31", "--steps", "80"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(result.stdout)

    assert metrics["device"] == "cpu"
    assert metrics["training_examples"] == 64
    assert metrics["validation_examples"] == 32
    assert metrics["train_mse"] < metrics["initial_train_mse"]
    assert metrics["validation_mse"] < 0.1
    assert metrics["validation_mae"] < 0.2


def test_stateful_sequence_capability_cli_trains_and_evaluates_cpu() -> None:
    script = ROOT / "project/scripts/experiments/training/neurons/capabilities/run_sequence_capability.py"
    result = subprocess.run(
        [sys.executable, str(script), "--seed", "41", "--steps", "250"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = json.loads(result.stdout)

    assert metrics["device"] == "cpu"
    assert metrics["training_examples"] == 128
    assert metrics["validation_examples"] == 64
    assert metrics["train_mse"] < metrics["initial_train_mse"]
    assert metrics["validation_mse"] < 0.01
