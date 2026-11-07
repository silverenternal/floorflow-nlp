"""Small structural contracts for task capabilities and boundary adapters."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

TaskInputT = TypeVar("TaskInputT")
CapabilityInputT = TypeVar("CapabilityInputT")
CapabilityOutputT = TypeVar("CapabilityOutputT")
TaskOutputT = TypeVar("TaskOutputT")


@runtime_checkable
class Capability(Protocol[CapabilityInputT, CapabilityOutputT]):
    """A replaceable task capability with no imposed model or state base class."""

    def execute(self, inputs: CapabilityInputT) -> CapabilityOutputT: ...


@runtime_checkable
class TaskAdapter(
    Protocol[TaskInputT, CapabilityInputT, CapabilityOutputT, TaskOutputT]
):
    """Translate task data to a capability's inputs and back to task output."""

    def encode(self, task_input: TaskInputT) -> CapabilityInputT: ...

    def decode(self, capability_output: CapabilityOutputT) -> TaskOutputT: ...


__all__ = ["Capability", "TaskAdapter"]
