"""Small regression capability and task-data adapter example."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from newera.neuron import MLPNeuron


@dataclass(frozen=True, slots=True)
class RegressionExample:
    features: tuple[float, float]
    target: float


class RegressionTaskAdapter:
    """Map records into the capability's tensor input and predictions back."""

    def encode(self, task_input: Sequence[RegressionExample]) -> torch.Tensor:
        if not task_input:
            raise ValueError("regression batch must not be empty")
        return torch.tensor(
            [example.features for example in task_input],
            dtype=torch.float32,
        )

    def decode(self, capability_output: torch.Tensor) -> tuple[float, ...]:
        if capability_output.ndim != 2 or capability_output.shape[-1] != 1:
            raise ValueError("regression output must have shape [batch, 1]")
        return tuple(float(value) for value in capability_output.detach().squeeze(-1))


class RegressionCapability(nn.Module):
    """Expose an MLP neuron implementation through a task capability boundary."""

    def __init__(self, neuron: MLPNeuron) -> None:
        super().__init__()
        self.neuron = neuron

    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.neuron(inputs).value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.execute(inputs)


__all__ = ["RegressionCapability", "RegressionExample", "RegressionTaskAdapter"]
