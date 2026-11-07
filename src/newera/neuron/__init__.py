"""Shared behavior contracts for NewEra neuron experiments."""

from newera.neuron.behavior import Neuron, NeuronResult
from newera.neuron.cells import MLPNeuron, StatefulMLPNeuron
from newera.neuron.composition import (
    ModelNeuron,
    ModelNeuronGraph,
    ModelNeuronGraphOutput,
    ModelNeuronGraphState,
    NeuronStateBundle,
    SequentialNeuron,
)
from newera.neuron.graph import (
    DynamicNeuronGraphLayer,
    LinkMutation,
    LinkProposalPolicy,
    NeuronGraphOutput,
    NeuronGraphTopology,
    TopologyController,
    gather_rows,
)

__all__ = [
    "DynamicNeuronGraphLayer",
    "LinkMutation",
    "LinkProposalPolicy",
    "MLPNeuron",
    "ModelNeuron",
    "ModelNeuronGraph",
    "ModelNeuronGraphOutput",
    "ModelNeuronGraphState",
    "Neuron",
    "NeuronGraphOutput",
    "NeuronGraphTopology",
    "NeuronResult",
    "NeuronStateBundle",
    "SequentialNeuron",
    "StatefulMLPNeuron",
    "TopologyController",
    "gather_rows",
]
