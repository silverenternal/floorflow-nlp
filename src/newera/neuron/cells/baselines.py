"""Small trainable controls for experiments against neuron candidates."""

from __future__ import annotations

import torch
from torch import nn

from newera.neuron.behavior import NeuronResult


class MLPNeuron(nn.Module):
    """Stateless MLP control implementing the shared neuron behavior contract."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("output_size", output_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def step(
        self,
        inputs: torch.Tensor,
        *,
        state: None = None,
    ) -> NeuronResult[torch.Tensor, None]:
        if inputs.ndim == 0 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"inputs must have a final dimension of {self.input_size}"
            )
        if state is not None:
            raise ValueError("MLPNeuron is stateless and accepts only state=None")
        return NeuronResult(self.network(inputs))

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        state: None = None,
    ) -> NeuronResult[torch.Tensor, None]:
        return self.step(inputs, state=state)


__all__ = ["MLPNeuron"]
