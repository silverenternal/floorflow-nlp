"""A stateful sequence-summing capability example."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from newera.neuron import StatefulMLPNeuron


@dataclass(frozen=True, slots=True)
class SequenceExample:
    values: tuple[float, ...]
    target: float


class SequenceTaskAdapter:
    """Encode fixed-length numeric sequences and decode scalar predictions."""

    def encode(self, task_input: Sequence[SequenceExample]) -> torch.Tensor:
        if not task_input:
            raise ValueError("sequence batch must not be empty")
        length = len(task_input[0].values)
        if length == 0 or any(len(example.values) != length for example in task_input):
            raise ValueError("all examples must have the same non-empty sequence length")
        return torch.tensor(
            [example.values for example in task_input],
            dtype=torch.float32,
        ).unsqueeze(-1)

    def decode(self, capability_output: torch.Tensor) -> tuple[float, ...]:
        if capability_output.ndim != 2 or capability_output.shape[-1] != 1:
            raise ValueError("sequence output must have shape [batch, 1]")
        return tuple(float(value) for value in capability_output.detach().squeeze(-1))


class SequenceSumCapability(nn.Module):
    """Apply a stateful neuron step-by-step and emit the final sequence value."""

    def __init__(self, neuron: StatefulMLPNeuron) -> None:
        super().__init__()
        self.neuron = neuron

    def execute(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError("inputs must be [batch, non-empty sequence, features]")
        state = None
        for index in range(inputs.shape[1]):
            result = self.neuron.step(inputs[:, index], state=state)
            state = result.state
        return result.value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.execute(inputs)


__all__ = ["SequenceExample", "SequenceSumCapability", "SequenceTaskAdapter"]
