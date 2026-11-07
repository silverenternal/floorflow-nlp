"""Train and evaluate the first task-facing neuron capability on synthetic data."""

from __future__ import annotations

import argparse
import json

import torch

from newera.capabilities import RegressionCapability, RegressionExample, RegressionTaskAdapter
from newera.neuron import MLPNeuron


def _examples(seed: int, count: int) -> tuple[RegressionExample, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(count, 2, generator=generator)
    targets = 0.5 * features[:, 0] - 1.2 * features[:, 1] + 0.4
    return tuple(
        RegressionExample(
            features=(float(row[0]), float(row[1])),
            target=float(target),
        )
        for row, target in zip(features, targets, strict=True)
    )


def run(seed: int, steps: int) -> dict[str, float | int | str]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    adapter = RegressionTaskAdapter()
    training = _examples(seed + 1, 64)
    validation = _examples(seed + 2, 32)
    train_inputs = adapter.encode(training)
    train_targets = torch.tensor([[row.target] for row in training])
    validation_inputs = adapter.encode(validation)
    validation_targets = torch.tensor([[row.target] for row in validation])

    capability = RegressionCapability(MLPNeuron(2, 16, 1))
    optimizer = torch.optim.Adam(capability.parameters(), lr=0.03)
    initial_mse = torch.nn.functional.mse_loss(
        capability.execute(train_inputs), train_targets
    ).item()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = capability.execute(train_inputs)
        loss = torch.nn.functional.mse_loss(prediction, train_targets)
        loss.backward()
        optimizer.step()

    capability.eval()
    with torch.no_grad():
        train_predictions = capability.execute(train_inputs)
        validation_predictions = capability.execute(validation_inputs)
        train_mse = torch.nn.functional.mse_loss(train_predictions, train_targets).item()
        validation_mse = torch.nn.functional.mse_loss(
            validation_predictions, validation_targets
        ).item()
        validation_mae = torch.nn.functional.l1_loss(
            validation_predictions, validation_targets
        ).item()
    decoded = adapter.decode(validation_predictions)
    return {
        "device": "cpu",
        "training_examples": len(training),
        "validation_examples": len(validation),
        "steps": steps,
        "parameters": sum(parameter.numel() for parameter in capability.parameters()),
        "forward_linear_macs_per_example": 48,
        "initial_train_mse": initial_mse,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "validation_mae": validation_mae,
        "decoded_predictions": len(decoded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.steps), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
