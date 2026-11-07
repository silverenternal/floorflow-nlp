"""Small stateful neuron-as-model candidates for controlled experiments."""

from __future__ import annotations

import torch
from torch import nn

from newera.neuron.behavior import NeuronResult


class StatefulMLPNeuron(nn.Module):
    """Encode input, update caller-owned memory, and decode a task-facing value."""

    def __init__(self, input_size: int, state_size: int, output_size: int) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("state_size", state_size),
            ("output_size", output_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_size = input_size
        self.state_size = state_size
        self.encoder = nn.Sequential(nn.Linear(input_size, state_size), nn.Tanh())
        self.update = nn.GRUCell(state_size, state_size)
        self.readout = nn.Linear(state_size, output_size)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    def step(
        self,
        inputs: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> NeuronResult[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"inputs must be [batch, {self.input_size}]"
            )
        if state is None:
            state = self.initial_state(
                inputs.shape[0],
                device=inputs.device,
                dtype=inputs.dtype,
            )
        expected_shape = (inputs.shape[0], self.state_size)
        if state.shape != expected_shape:
            raise ValueError(f"state must have shape {expected_shape}")
        if state.device != inputs.device or state.dtype != inputs.dtype:
            raise ValueError("state device and dtype must match inputs")
        encoded = self.encoder(inputs)
        next_state = self.update(encoded, state)
        return NeuronResult(self.readout(next_state), next_state)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> NeuronResult[torch.Tensor, torch.Tensor]:
        return self.step(inputs, state=state)


__all__ = ["StatefulMLPNeuron"]
