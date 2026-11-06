"""Stateful local neuron module with fast and slow temporal state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NeuronState:
    fast: torch.Tensor
    slow: torch.Tensor


@dataclass(frozen=True)
class NeuronModuleOutput:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    local_output: torch.Tensor
    state: NeuronState
    diagnostics: dict[str, torch.Tensor]


class NeuronModule(nn.Module):
    """A compact local model, not a scalar activation unit."""

    def __init__(self, hidden_size: int, local_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.local_size = int(local_size)
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, local_size, bias=False)
        self.fast = nn.GRUCell(local_size, local_size)
        self.slow = nn.GRUCell(local_size, local_size)
        self.fast_gate = nn.Linear(local_size, local_size)
        self.slow_gate = nn.Linear(local_size, local_size)
        self.local_output = nn.Linear(local_size, hidden_size, bias=False)
        self.up = nn.Linear(local_size, hidden_size, bias=False)
        self.birth_gate = nn.Parameter(torch.zeros(1))

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NeuronState:
        shape = (batch, self.local_size)
        return NeuronState(
            fast=torch.zeros(shape, device=device, dtype=dtype),
            slow=torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        parent_hidden: torch.Tensor,
        state: NeuronState | None = None,
    ) -> NeuronModuleOutput:
        if parent_hidden.ndim != 3:
            raise ValueError("parent_hidden must be [batch, sequence, channels]")
        batch, sequence, _ = parent_hidden.shape
        state = (
            self.initial_state(batch, parent_hidden.device, parent_hidden.dtype)
            if state is None
            else state
        )
        fast_values = []
        slow_values = []
        fast_state = state.fast
        slow_state = state.slow
        for index in range(sequence):
            local = F.silu(self.down(self.norm(parent_hidden[:, index])))
            fast_state = self.fast(local, fast_state)
            slow_state = self.slow(local, slow_state)
            fast_gate = torch.sigmoid(self.fast_gate(fast_state))
            slow_gate = torch.sigmoid(self.slow_gate(slow_state))
            fast_values.append(fast_state * fast_gate)
            slow_values.append(slow_state * slow_gate)
        fast_trace = torch.stack(fast_values, dim=1)
        slow_trace = torch.stack(slow_values, dim=1)
        local_state = torch.tanh(fast_trace + 0.5 * slow_trace)
        local_output = torch.tanh(self.local_output(local_state))
        residual = self.birth_gate.to(parent_hidden.dtype) * self.up(local_state)
        return NeuronModuleOutput(
            hidden_states=parent_hidden + residual,
            residual=residual,
            local_output=local_output,
            state=NeuronState(fast_state, slow_state),
            diagnostics={
                "fast_gate_mean": fast_gate.mean(),
                "slow_gate_mean": slow_gate.mean(),
                "fast_state_norm": fast_state.square().mean(),
                "slow_state_norm": slow_state.square().mean(),
            },
        )


__all__ = ["NeuronModule", "NeuronModuleOutput", "NeuronState"]
