"""Dynamic sparse neuron graph contracts and computation."""

from newera.neuron.graph.diagnostics import (
    action_distribution,
    drop_topology_edges,
    edge_set_jaccard,
    normalized_effective_rank,
    relative_dirichlet_energy,
    rewire_topology_matched,
    sampled_causal_gradient_transport,
    topology_metrics,
    weighted_dirichlet_energy,
)
from newera.neuron.graph.layer import DynamicNeuronGraphLayer, NeuronGraphOutput, gather_rows
from newera.neuron.graph.policy import LinkProposalPolicy
from newera.neuron.graph.topology import (
    LinkMutation,
    NeuronGraphTopology,
    NeuronIdentityRegistry,
    TopologyController,
)

__all__ = [
    "DynamicNeuronGraphLayer",
    "LinkMutation",
    "LinkProposalPolicy",
    "NeuronGraphOutput",
    "NeuronGraphTopology",
    "NeuronIdentityRegistry",
    "TopologyController",
    "action_distribution",
    "drop_topology_edges",
    "edge_set_jaccard",
    "gather_rows",
    "normalized_effective_rank",
    "relative_dirichlet_energy",
    "rewire_topology_matched",
    "sampled_causal_gradient_transport",
    "topology_metrics",
    "weighted_dirichlet_energy",
]
