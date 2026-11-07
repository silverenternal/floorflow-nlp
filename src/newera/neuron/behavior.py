"""Minimal structural contract for a neuron behavior implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class NeuronResult(Generic[OutputT, StateT]):
    """The value produced by one call and optional caller-owned next state.

    Passing ``state=None`` starts a fresh stateful computation. Stateless
    implementations return ``state=None``. State is opaque to the protocol;
    callers own its lifetime and can reset by discarding it.
    """

    value: OutputT
    state: StateT | None = None


@runtime_checkable
class Neuron(Protocol[InputT, OutputT, StateT]):
    """A structural interface; implementations need not inherit from a base.

    Inputs, outputs, state representation, tensor shapes, and learning rules
    are deliberately left to implementations and their task adapters.
    """

    def step(
        self,
        inputs: InputT,
        *,
        state: StateT | None = None,
    ) -> NeuronResult[OutputT, StateT]: ...


__all__ = ["Neuron", "NeuronResult"]
