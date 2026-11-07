"""Reusable message aggregation components for model neurons."""

from newera.neuron.cells.aggregation.deep_sets import (
    ContextualSumResidualDeepSetsAggregator,
    DeepSetsMessageAggregator,
)
from newera.neuron.cells.aggregation.grouped_relation import GroupRoleDeepSetsAggregator
from newera.neuron.cells.aggregation.set_transformer import (
    ContextualSumResidualSetTransformerAggregator,
    SetTransformerSetAggregator,
    SumResidualSetTransformerAggregator,
)

__all__ = [
    "ContextualSumResidualDeepSetsAggregator",
    "ContextualSumResidualSetTransformerAggregator",
    "DeepSetsMessageAggregator",
    "GroupRoleDeepSetsAggregator",
    "SetTransformerSetAggregator",
    "SumResidualSetTransformerAggregator",
]
