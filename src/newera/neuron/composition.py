"""PyTorch composition helpers for protocol-compatible neuron modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from newera.neuron.behavior import NeuronResult
from newera.neuron.graph.layer import gather_rows
from newera.neuron.graph.topology import NeuronGraphTopology

NeuronStateBundle = tuple[Any | None, ...]


class SequentialNeuron(nn.Module):
    """Pass each value through ordered neuron modules with isolated states.

    This optional helper uses ``nn.Module`` children so PyTorch can discover
    their parameters. The core ``Neuron`` protocol itself has no such base
    class requirement.
    """

    def __init__(self, neurons: Sequence[nn.Module]) -> None:
        super().__init__()
        if not neurons:
            raise ValueError("SequentialNeuron requires at least one module")
        if any(not callable(getattr(neuron, "step", None)) for neuron in neurons):
            raise TypeError("every child must implement step(inputs, state=...)")
        self.neurons = nn.ModuleList(neurons)

    def step(
        self,
        inputs: Any,
        *,
        state: NeuronStateBundle | None = None,
    ) -> NeuronResult[Any, NeuronStateBundle]:
        if state is None:
            old_states: NeuronStateBundle = (None,) * len(self.neurons)
        elif not isinstance(state, tuple) or len(state) != len(self.neurons):
            raise ValueError("state must be a tuple with one entry per neuron")
        else:
            old_states = state

        value = inputs
        next_states = []
        for neuron, old_state in zip(self.neurons, old_states, strict=True):
            result = neuron.step(value, state=old_state)
            if not isinstance(result, NeuronResult):
                raise TypeError("each neuron step must return NeuronResult")
            value = result.value
            next_states.append(result.state)
        return NeuronResult(value, tuple(next_states))

    def forward(
        self,
        inputs: Any,
        *,
        state: NeuronStateBundle | None = None,
    ) -> NeuronResult[Any, NeuronStateBundle]:
        return self.step(inputs, state=state)


class ModelNeuron(nn.Module):
    """Shared model architecture applied independently at each neuron position.

    A graph stores one set of model weights. Each logical neuron position has
    its own recurrent state, while variable-degree message aggregation is part
    of this model's forward behavior.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        message_size: int,
        num_edge_types: int = 1,
        message_aggregator: nn.Module | None = None,
        exact_input_sum_readout: bool = False,
        exact_input_sum_output_channel: int | None = None,
        aggregate_only_output_channels: tuple[int, ...] = (),
        use_edge_type_embedding: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("hidden_size", hidden_size),
            ("message_size", message_size),
            ("num_edge_types", num_edge_types),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(exact_input_sum_readout, bool):
            raise TypeError("exact_input_sum_readout must be a bool")
        if not isinstance(use_edge_type_embedding, bool):
            raise TypeError("use_edge_type_embedding must be a bool")
        if exact_input_sum_output_channel is not None and (
            not isinstance(exact_input_sum_output_channel, int)
            or isinstance(exact_input_sum_output_channel, bool)
            or not 0 <= exact_input_sum_output_channel < message_size
        ):
            raise ValueError("exact_input_sum_output_channel must index a message output channel")
        if not isinstance(aggregate_only_output_channels, tuple) or any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or not 0 <= channel < message_size
            for channel in aggregate_only_output_channels
        ):
            raise ValueError("aggregate_only_output_channels must be a tuple of valid output channels")
        if len(set(aggregate_only_output_channels)) != len(aggregate_only_output_channels):
            raise ValueError("aggregate_only_output_channels cannot contain duplicates")
        if exact_input_sum_output_channel in aggregate_only_output_channels:
            raise ValueError("exact sum and aggregate-only output channels must be disjoint")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.num_edge_types = num_edge_types
        self.message_aggregator = message_aggregator
        self.exact_input_sum_readout = exact_input_sum_readout
        self.use_edge_type_embedding = use_edge_type_embedding
        self.exact_input_sum_output_channel = exact_input_sum_output_channel
        self.aggregate_only_output_channels = aggregate_only_output_channels
        self.message_projection = nn.Linear(message_size, hidden_size, bias=False)
        edge_type_embedding = nn.Embedding(num_edge_types, hidden_size)
        self.edge_type_embedding = edge_type_embedding if use_edge_type_embedding else None
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.state_update = nn.GRUCell(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, message_size)
        self.aggregate_output_projection = nn.Linear(hidden_size, message_size)
        nn.init.zeros_(self.aggregate_output_projection.weight)
        nn.init.zeros_(self.aggregate_output_projection.bias)
        if exact_input_sum_readout:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @property
    def architecture_signature(self) -> tuple[int, int, int, int, str]:
        implementation = type(self).__qualname__
        if self.exact_input_sum_readout:
            implementation += "+ExactInputSumReadout"
        if not self.use_edge_type_embedding:
            implementation += "+NoEdgeTypeEmbedding"
        if self.exact_input_sum_output_channel is not None:
            implementation += f"+ExactInputSumChannel{self.exact_input_sum_output_channel}"
        if self.aggregate_only_output_channels:
            implementation += f"+AggregateOnlyChannels{self.aggregate_only_output_channels}"
        if self.message_aggregator is not None:
            implementation += f"+{type(self.message_aggregator).__qualname__}"
            aggregator_signature = getattr(self.message_aggregator, "architecture_signature", None)
            if aggregator_signature is not None:
                implementation += f"[{aggregator_signature!r}]"
        return (
            self.input_size,
            self.hidden_size,
            self.message_size,
            self.num_edge_types,
            implementation,
        )

    def initial_state(self, inputs: Tensor) -> Tensor:
        return inputs.new_zeros((inputs.shape[0], self.hidden_size))

    def step(
        self,
        inputs: Tensor,
        *,
        incoming_messages: Tensor,
        incoming_strengths: Tensor,
        incoming_edge_types: Tensor,
        incoming_group_ids: Tensor | None = None,
        state: Tensor | None = None,
    ) -> NeuronResult[Tensor, Tensor]:
        """Update one model instance from its variable-size contact set.

        ``incoming_messages[b, i]`` is one contact payload. Rows are a
        multiset: repeated rows remain distinct contacts. Contact or group
        identity that affects computation must be encoded in the payload;
        ``incoming_edge_types`` carries the relation type for each contact.
        Optional integer ``incoming_group_ids`` carries exact group identity
        for relation-aware aggregators; values are opaque equality labels.
        The number of contacts is set by the input tensor, with no top-k cap.
        """
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError(f"inputs must have shape [batch, {self.input_size}]")
        if incoming_messages.ndim != 3 or incoming_messages.shape[0] != inputs.shape[0]:
            raise ValueError("incoming_messages must have shape [batch, incoming, message_size]")
        if incoming_messages.shape[-1] != self.message_size:
            raise ValueError("incoming message width does not match message_size")
        edge_count = incoming_messages.shape[1]
        edge_data_shape = (inputs.shape[0], edge_count)
        if incoming_strengths.shape != edge_data_shape or not incoming_strengths.is_floating_point():
            raise ValueError("incoming_strengths must align with each row's messages")
        if incoming_edge_types.shape != edge_data_shape or incoming_edge_types.dtype != torch.long:
            raise ValueError("incoming_edge_types must be long and align with each row's messages")
        if incoming_group_ids is not None and (
            incoming_group_ids.shape != edge_data_shape or incoming_group_ids.dtype != torch.long
        ):
            raise ValueError("incoming_group_ids must be long and align with each row's messages")
        if any(
            value.device != inputs.device
            for value in (incoming_messages, incoming_strengths, incoming_edge_types)
        ):
            raise ValueError("all inputs and incoming edge data must share a device")
        if incoming_group_ids is not None and incoming_group_ids.device != inputs.device:
            raise ValueError("incoming_group_ids must share the input device")
        if incoming_messages.dtype != inputs.dtype or incoming_strengths.dtype != inputs.dtype:
            raise ValueError("input, message, and strength dtypes must match")
        if state is None:
            state = self.initial_state(inputs)
        if state.shape != (inputs.shape[0], self.hidden_size):
            raise ValueError("state shape does not match batch and hidden_size")
        if state.device != inputs.device or state.dtype != inputs.dtype:
            raise ValueError("state device and dtype must match inputs")

        if edge_count:
            projected = self.message_projection(incoming_messages)
            if self.edge_type_embedding is None:
                weighted = projected * incoming_strengths.unsqueeze(-1)
            else:
                relation = gather_rows(
                    self.edge_type_embedding.weight, incoming_edge_types.reshape(-1)
                ).reshape(*edge_data_shape, self.hidden_size)
                weighted = (projected + relation) * incoming_strengths.unsqueeze(-1)
            degree = incoming_strengths.sum(dim=1, keepdim=True).clamp_min(1.0).sqrt()
            if self.message_aggregator is None:
                aggregate = weighted.sum(dim=1) / degree
            else:
                mask = incoming_strengths != 0
                contextual_aggregate = getattr(
                    self.message_aggregator, "forward_batched_with_context", None
                )
                relation_aggregate = getattr(
                    self.message_aggregator, "forward_batched_with_relations", None
                )
                batched_aggregate = getattr(self.message_aggregator, "forward_batched", None)
                if callable(relation_aggregate):
                    if incoming_group_ids is None:
                        raise ValueError("this message aggregator requires incoming_group_ids")
                    aggregate = relation_aggregate(
                        weighted,
                        edge_types=incoming_edge_types,
                        group_ids=incoming_group_ids,
                        mask=mask,
                        context=inputs,
                    ) / degree
                elif callable(contextual_aggregate):
                    aggregate = contextual_aggregate(weighted, mask=mask, context=inputs) / degree
                elif callable(batched_aggregate):
                    aggregate = batched_aggregate(weighted, mask=mask) / degree
                else:
                    aggregate = torch.stack(
                        [
                            self.message_aggregator(weighted[row][mask[row]]).reshape(
                                self.hidden_size
                            )
                            / degree[row]
                            for row in range(inputs.shape[0])
                        ]
                    )
            readout_aggregate = aggregate * degree
        else:
            aggregate = inputs.new_zeros((inputs.shape[0], self.hidden_size))
            readout_aggregate = aggregate
        drive = torch.tanh(self.input_projection(inputs) + aggregate)
        next_state = self.state_update(drive, state)
        state_value = self.output_projection(next_state)
        aggregate_value = self.aggregate_output_projection(readout_aggregate)
        value = state_value + aggregate_value
        if self.aggregate_only_output_channels:
            channel_mask = torch.zeros(self.message_size, dtype=torch.bool, device=value.device)
            channel_mask[list(self.aggregate_only_output_channels)] = True
            aggregate_only_value = F.linear(
                readout_aggregate, self.aggregate_output_projection.weight, bias=None
            )
            value = torch.where(channel_mask.unsqueeze(0), aggregate_only_value, value)
        if self.exact_input_sum_output_channel is not None:
            raw_sum = (incoming_messages * incoming_strengths.unsqueeze(-1)).sum(dim=1)
            channel_mask = torch.arange(self.message_size, device=value.device).eq(
                self.exact_input_sum_output_channel
            )
            value = torch.where(channel_mask.unsqueeze(0), raw_sum, value)
        if self.exact_input_sum_readout:
            value = value + (incoming_messages * incoming_strengths.unsqueeze(-1)).sum(dim=1)
        return NeuronResult(value, next_state)


@dataclass(frozen=True, slots=True)
class ModelNeuronGraphState:
    """Caller-owned state rows and value snapshot bound to stable IDs."""

    neuron_states: Tensor
    previous_values: Tensor
    node_ids: Tensor


@dataclass(frozen=True, slots=True)
class ModelNeuronGraphOutput:
    values: Tensor
    state: ModelNeuronGraphState


class ModelNeuronGraph(nn.Module):
    """Apply one shared neuron model to any number of logical neuron positions.

    The sparse topology stores only connections and per-neuron state is caller
    owned. There is exactly one registered model parameter set regardless of
    node count; aggregation and update remain inside ``neuron``.
    """

    def __init__(self, neuron: ModelNeuron) -> None:
        super().__init__()
        if not isinstance(neuron, ModelNeuron):
            raise TypeError("neuron must be a ModelNeuron")
        self.neuron = neuron
        self.architecture_signature = neuron.architecture_signature
        self._adjacency_cache_key: tuple[object, ...] | None = None
        self._adjacency_cache: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None

    def _adjacency(
        self,
        topology: NeuronGraphTopology,
        device: torch.device,
        candidate_edges: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        node_ids = topology.node_ids
        if node_ids is None:
            raise ValueError("model neuron graphs require stable node_ids")
        tensors = (topology.edge_index, topology.edge_type, topology.strength, node_ids)
        if topology.edge_group_ids is not None:
            tensors = (*tensors, topology.edge_group_ids)
        if candidate_edges is not None and (
            candidate_edges.dtype != torch.long
            or candidate_edges.ndim != 2
            or candidate_edges.shape[0] != 2
            or candidate_edges.device != device
        ):
            raise ValueError("candidate_edges must be long [2, candidates] on the graph device")
        key = (
            topology.nodes_per_graph,
            str(device),
            *(value.data_ptr() for value in tensors),
            *(value._version for value in tensors),
            None if candidate_edges is None else candidate_edges.data_ptr(),
            None if candidate_edges is None else candidate_edges._version,
            None if candidate_edges is None else candidate_edges.shape[1],
        )
        if key == self._adjacency_cache_key and self._adjacency_cache is not None:
            topology.validate(
                batch_size=1,
                nodes_per_graph=topology.nodes_per_graph,
                num_edge_types=self.neuron.num_edge_types,
                device=device,
                causal=False,
                check_values=False,
            )
            return self._adjacency_cache

        topology.validate(
            batch_size=1,
            nodes_per_graph=topology.nodes_per_graph,
            num_edge_types=self.neuron.num_edge_types,
            device=device,
            causal=False,
        )
        if torch.any(node_ids < 0):
            raise ValueError("every model neuron must have a registered non-negative node ID")
        neuron_count = topology.nodes_per_graph
        edge_rows = torch.stack(
            (topology.edge_index[0], topology.edge_index[1], topology.edge_type)
        ).detach().cpu().tolist()
        edge_strengths = topology.strength.detach().cpu().tolist()
        edge_group_ids = (
            topology.edge_group_ids.detach().cpu().tolist()
            if topology.edge_group_ids is not None else [0] * topology.edge_count
        )
        candidate_rows = (
            candidate_edges.detach().cpu().t().tolist()
            if candidate_edges is not None
            else []
        )
        candidate_keys = {(source, target) for source, target in candidate_rows}
        if len(candidate_keys) != len(candidate_rows):
            raise ValueError("candidate_edges cannot contain duplicate directed edges")
        if any(
            source == target or source < 0 or target < 0 or source >= neuron_count or target >= neuron_count
            for source, target in candidate_rows
        ):
            raise ValueError("candidate edge references an invalid neuron pair")
        static_incoming: list[list[tuple[int, int, float, int]]] = [[] for _ in range(neuron_count)]
        for source, target, edge_type, strength, group_id in zip(
            *edge_rows, edge_strengths, edge_group_ids, strict=True
        ):
            if (source, target) not in candidate_keys:
                static_incoming[target].append((source, edge_type, strength, group_id))
        candidate_incoming: list[list[tuple[int, int, int, int]]] = [[] for _ in range(neuron_count)]
        for candidate_index, (source, target) in enumerate(candidate_rows):
            for edge_type in range(self.neuron.num_edge_types):
                candidate_incoming[target].append(
                    (source, edge_type, candidate_index * self.neuron.num_edge_types + edge_type, 0)
                )

        def pad_rows(
            rows: list[list[tuple[int, int, object, int]]], *, padding_value: object = 0
        ) -> tuple[list[list[int]], list[list[int]], list[list[object]], list[list[int]]]:
            max_degree = max((len(row) for row in rows), default=0)
            source_rows = [[0] * max_degree for _ in range(neuron_count)]
            type_rows = [[0] * max_degree for _ in range(neuron_count)]
            value_rows: list[list[object]] = [
                [padding_value] * max_degree for _ in range(neuron_count)
            ]
            group_rows = [[0] * max_degree for _ in range(neuron_count)]
            for target, row in enumerate(rows):
                for column, (source, edge_type, value, group_id) in enumerate(row):
                    source_rows[target][column] = source
                    type_rows[target][column] = edge_type
                    value_rows[target][column] = value
                    group_rows[target][column] = group_id
            return source_rows, type_rows, value_rows, group_rows

        static_sources, static_types, static_strengths, static_groups = pad_rows(static_incoming)
        candidate_sources, candidate_types, candidate_slots, _ = pad_rows(
            candidate_incoming, padding_value=-1
        )
        static_degree = len(static_sources[0]) if static_sources else 0
        candidate_degree = len(candidate_sources[0]) if candidate_sources else 0
        self._adjacency_cache = (
            torch.tensor(static_sources, dtype=torch.long, device=device).reshape(neuron_count, static_degree),
            torch.tensor(static_types, dtype=torch.long, device=device).reshape(neuron_count, static_degree),
            torch.tensor(static_strengths, dtype=topology.strength.dtype, device=device).reshape(
                neuron_count, static_degree
            ),
            torch.tensor(static_groups, dtype=torch.long, device=device).reshape(neuron_count, static_degree),
            torch.tensor(candidate_sources, dtype=torch.long, device=device).reshape(
                neuron_count, candidate_degree
            ),
            torch.tensor(candidate_types, dtype=torch.long, device=device).reshape(
                neuron_count, candidate_degree
            ),
            torch.tensor(candidate_slots, dtype=torch.long, device=device).reshape(
                neuron_count, candidate_degree
            ),
        )
        self._adjacency_cache_key = key
        return self._adjacency_cache

    def step(
        self,
        inputs: Tensor,
        topology: NeuronGraphTopology,
        *,
        state: ModelNeuronGraphState | None = None,
        candidate_edges: Tensor | None = None,
        candidate_type_weights: Tensor | None = None,
        candidate_group_ids: Tensor | None = None,
    ) -> ModelNeuronGraphOutput:
        if inputs.ndim != 3 or inputs.shape[-1] != self.neuron.input_size:
            raise ValueError("inputs must have shape [batch, neuron_count, input_size]")
        batch_size, neuron_count, _ = inputs.shape
        if neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if topology.batch_size != 1 or topology.nodes_per_graph != neuron_count:
            raise ValueError("topology must describe one shared graph matching neuron_count")
        if topology.edge_index.device != inputs.device:
            raise ValueError("topology and inputs must share a device")
        if (candidate_edges is None) != (candidate_type_weights is None):
            raise ValueError("candidate_edges and candidate_type_weights must be supplied together")
        candidate_count = 0 if candidate_edges is None else candidate_edges.shape[1]
        if candidate_group_ids is not None and (
            candidate_group_ids.dtype != torch.long
            or candidate_group_ids.device != inputs.device
            or candidate_group_ids.shape not in {(candidate_count,), (batch_size, candidate_count)}
        ):
            raise ValueError("candidate_group_ids must be long [candidates] or [batch, candidates]")
        relation_aware = callable(
            getattr(self.neuron.message_aggregator, "forward_batched_with_relations", None)
        )
        if relation_aware and topology.edge_count and topology.edge_group_ids is None:
            raise ValueError("relation-aware graph execution requires topology.edge_group_ids")
        if relation_aware and candidate_count and candidate_group_ids is None:
            raise ValueError("relation-aware candidate edges require candidate_group_ids")
        if candidate_type_weights is not None:
            if candidate_type_weights.ndim == 2:
                if candidate_type_weights.shape != (candidate_count, self.neuron.num_edge_types):
                    raise ValueError("candidate type weights must align with candidate edges")
                candidate_type_weights = candidate_type_weights.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            elif candidate_type_weights.shape != (
                batch_size,
                candidate_count,
                self.neuron.num_edge_types,
            ):
                raise ValueError("batched candidate type weights must align with batch and edges")
            if candidate_type_weights.device != inputs.device or not candidate_type_weights.is_floating_point():
                raise ValueError("candidate type weights must be floating and share the input device")
            if candidate_type_weights.dtype != inputs.dtype:
                raise ValueError("candidate type weights dtype must match inputs")
        (
            static_sources,
            static_types,
            static_strengths,
            static_group_ids,
            candidate_sources,
            candidate_types,
            candidate_slots,
        ) = self._adjacency(topology, inputs.device, candidate_edges)
        assert topology.node_ids is not None
        if state is None:
            previous_values = inputs.new_zeros(
                (batch_size, neuron_count, self.neuron.message_size)
            )
            previous_states = inputs.new_zeros((batch_size, neuron_count, self.neuron.hidden_size))
        else:
            if state.node_ids.shape != (neuron_count,) or state.node_ids.device != inputs.device:
                raise ValueError("graph state identities changed; remap state by node ID first")
            if not torch.equal(state.node_ids, topology.node_ids[0]):
                raise ValueError("graph state identities are reordered; remap state by node ID first")
            if state.neuron_states.shape != (
                batch_size,
                neuron_count,
                self.neuron.hidden_size,
            ):
                raise ValueError("neuron_states shape does not match graph inputs")
            if state.previous_values.shape != (
                batch_size,
                neuron_count,
                self.neuron.message_size,
            ):
                raise ValueError("previous_values shape does not match graph inputs")
            if state.previous_values.device != inputs.device or state.previous_values.dtype != inputs.dtype:
                raise ValueError("graph state device and dtype must match inputs")
            previous_values = state.previous_values
            previous_states = state.neuron_states

        batch_edge_offsets = (
            torch.arange(batch_size, device=inputs.device, dtype=torch.long).view(batch_size, 1, 1)
            * neuron_count
        )
        static_degree = static_sources.shape[1]
        candidate_degree = candidate_sources.shape[1]
        source_indices = torch.cat((static_sources, candidate_sources), dim=1)
        edge_types = torch.cat((static_types, candidate_types), dim=1)
        fixed_strengths = static_strengths.to(dtype=inputs.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        fixed_group_ids = static_group_ids.unsqueeze(0).expand(batch_size, -1, -1)
        if candidate_type_weights is None:
            dynamic_strengths = inputs.new_zeros((batch_size, neuron_count, candidate_degree))
        else:
            flat_weights = candidate_type_weights.reshape(batch_size, -1)
            safe_slots = candidate_slots.clamp_min(0)
            weight_offsets = (
                torch.arange(batch_size, device=inputs.device, dtype=torch.long).view(batch_size, 1, 1)
                * (candidate_count * self.neuron.num_edge_types)
            )
            weight_indices = (safe_slots.unsqueeze(0) + weight_offsets).reshape(-1)
            dynamic_strengths = gather_rows(flat_weights.reshape(-1), weight_indices).reshape(
                batch_size, neuron_count, candidate_degree
            )
            dynamic_strengths = dynamic_strengths * candidate_slots.ge(0).unsqueeze(0)
        strengths = torch.cat((fixed_strengths, dynamic_strengths), dim=2).reshape(
            batch_size * neuron_count, static_degree + candidate_degree
        )
        if candidate_group_ids is None:
            dynamic_group_ids = torch.zeros(
                (batch_size, neuron_count, candidate_degree), dtype=torch.long, device=inputs.device
            )
        else:
            candidate_group_ids = candidate_group_ids.reshape(1, -1).expand(batch_size, -1) \
                if candidate_group_ids.ndim == 1 else candidate_group_ids
            candidate_indices = candidate_slots.clamp_min(0) // self.neuron.num_edge_types
            group_offsets = torch.arange(batch_size, device=inputs.device).view(batch_size, 1, 1) * candidate_count
            flat_candidate_groups = candidate_group_ids.reshape(-1)
            dynamic_group_ids = gather_rows(
                flat_candidate_groups,
                (candidate_indices.unsqueeze(0) + group_offsets).reshape(-1),
            ).reshape(batch_size, neuron_count, candidate_degree)
            dynamic_group_ids = dynamic_group_ids * candidate_slots.ge(0).unsqueeze(0)
        group_ids = torch.cat((fixed_group_ids, dynamic_group_ids), dim=2).reshape(
            batch_size * neuron_count, static_degree + candidate_degree
        )
        max_degree = static_degree + candidate_degree
        message_indices = (source_indices.unsqueeze(0) + batch_edge_offsets).reshape(-1)
        message_indices = message_indices.reshape(batch_size, neuron_count, max_degree).reshape(-1)
        messages = gather_rows(
            previous_values.reshape(batch_size * neuron_count, self.neuron.message_size),
            message_indices,
        ).reshape(batch_size * neuron_count, max_degree, self.neuron.message_size)
        typed_edges = edge_types.unsqueeze(0).expand(batch_size, -1, -1)
        typed_edges = typed_edges.reshape(batch_size * neuron_count, max_degree)
        result = self.neuron.step(
            inputs.reshape(batch_size * neuron_count, self.neuron.input_size),
            incoming_messages=messages,
            incoming_strengths=strengths,
            incoming_edge_types=typed_edges,
            incoming_group_ids=group_ids if topology.edge_group_ids is not None or candidate_group_ids is not None else None,
            state=previous_states.reshape(batch_size * neuron_count, self.neuron.hidden_size),
        )
        values = result.value.reshape(batch_size, neuron_count, self.neuron.message_size)
        return ModelNeuronGraphOutput(
            values,
            ModelNeuronGraphState(
                result.state.reshape(batch_size, neuron_count, self.neuron.hidden_size),
                values,
                topology.node_ids[0].detach().clone(),
            ),
        )

    def remap_state(
        self,
        state: ModelNeuronGraphState,
        new_node_ids: Tensor,
    ) -> ModelNeuronGraphState:
        """Reorder or resize state by identity; unseen IDs start with zero state."""
        if new_node_ids.dtype != torch.long or new_node_ids.ndim != 2 or new_node_ids.shape[0] != 1:
            raise ValueError("new_node_ids must have shape [1, new_nodes] and dtype long")
        if new_node_ids.device != state.node_ids.device:
            raise ValueError("new_node_ids and graph state must share a device")
        old_ids = state.node_ids.detach().cpu().tolist()
        next_ids = new_node_ids[0].detach().cpu().tolist()
        if any(identity < 0 for identity in next_ids) or len(set(next_ids)) != len(next_ids):
            raise ValueError("model neuron identities must be unique and non-negative")
        old_positions = {identity: index for index, identity in enumerate(old_ids)}
        next_states: list[Tensor] = []
        next_values: list[Tensor] = []
        for identity in next_ids:
            old_position = old_positions.get(identity)
            if old_position is None:
                next_states.append(
                    state.neuron_states.new_zeros(
                        (state.neuron_states.shape[0], self.neuron.hidden_size)
                    )
                )
                next_values.append(
                    state.previous_values.new_zeros(
                        (state.previous_values.shape[0], self.neuron.message_size)
                    )
                )
            else:
                next_states.append(state.neuron_states[:, old_position])
                next_values.append(state.previous_values[:, old_position])
        previous_values = (
            torch.stack(next_values, dim=1)
            if next_values
            else state.previous_values.new_empty(
                (state.previous_values.shape[0], 0, self.neuron.message_size)
            )
        )
        return ModelNeuronGraphState(
            torch.stack(next_states, dim=1), previous_values, new_node_ids[0].detach().clone()
        )

    def forward(
        self,
        inputs: Tensor,
        topology: NeuronGraphTopology,
        *,
        state: ModelNeuronGraphState | None = None,
        candidate_edges: Tensor | None = None,
        candidate_type_weights: Tensor | None = None,
        candidate_group_ids: Tensor | None = None,
    ) -> ModelNeuronGraphOutput:
        return self.step(
            inputs,
            topology,
            state=state,
            candidate_edges=candidate_edges,
            candidate_type_weights=candidate_type_weights,
            candidate_group_ids=candidate_group_ids,
        )


__all__ = [
    "ModelNeuron",
    "ModelNeuronGraph",
    "ModelNeuronGraphOutput",
    "ModelNeuronGraphState",
    "NeuronStateBundle",
    "SequentialNeuron",
]
