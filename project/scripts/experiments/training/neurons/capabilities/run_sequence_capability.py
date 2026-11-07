"""Train/evaluate a stateful sequence-sum capability on synthetic CPU data."""

from __future__ import annotations

import argparse
import json

import torch

from newera.capabilities import SequenceExample, SequenceSumCapability, SequenceTaskAdapter
from newera.neuron import StatefulMLPNeuron


def _examples(seed: int, count: int) -> tuple[SequenceExample, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(count, 5, generator=generator) * 2.0 - 1.0
    targets = values.sum(dim=1)
    return tuple(
        SequenceExample(
            values=tuple(float(value) for value in sequence),
            target=float(target),
        )
        for sequence, target in zip(values, targets, strict=True)
    )


def run(seed: int, steps: int) -> dict[str, float | int | str]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    adapter = SequenceTaskAdapter()
    training = _examples(seed + 1, 128)
    validation = _examples(seed + 2, 64)
    train_inputs = adapter.encode(training)
    train_targets = torch.tensor([[row.target] for row in training])
    validation_inputs = adapter.encode(validation)
    validation_targets = torch.tensor([[row.target] for row in validation])

    capability = SequenceSumCapability(StatefulMLPNeuron(1, 8, 1))
    optimizer = torch.optim.Adam(capability.parameters(), lr=0.02)
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
        "sequence_length": 5,
        "steps": steps,
        "parameters": sum(parameter.numel() for parameter in capability.parameters()),
        "forward_linear_macs_per_timestep": 400,
        "forward_linear_macs_per_sequence": 2000,
        "initial_train_mse": initial_mse,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "validation_mae": validation_mae,
        "decoded_predictions": len(decoded),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.steps), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
