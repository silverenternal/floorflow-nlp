"""Composable, task-facing capability contracts."""

from newera.capabilities.contracts import Capability, TaskAdapter
from newera.capabilities.regression import (
    RegressionCapability,
    RegressionExample,
    RegressionTaskAdapter,
)
from newera.capabilities.sequence import (
    SequenceExample,
    SequenceSumCapability,
    SequenceTaskAdapter,
)

__all__ = [
    "Capability",
    "RegressionCapability",
    "RegressionExample",
    "RegressionTaskAdapter",
    "SequenceExample",
    "SequenceSumCapability",
    "SequenceTaskAdapter",
    "TaskAdapter",
]
