"""Variable-degree sparse neuron graphs and caller-owned topology mutations."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


class NeuronIdentityRegistry:
    """Allocate stable neuron IDs and persist their external-key mapping.

    IDs encode a 15-bit namespace and a 48-bit monotonically increasing local
    sequence, remaining within signed int64. Entries are never recycled.
    Persistence is caller-owned storage, with atomic replacement for local files.
    """

    SCHEMA = "newera-neuron-identity-registry-v1"
    _MAX_NAMESPACE = (1 << 15) - 1
    _LOCAL_BITS = 48
    _MAX_LOCAL_ID = (1 << _LOCAL_BITS) - 1

    def __init__(self, namespace: int) -> None:
        if not isinstance(namespace, int) or isinstance(namespace, bool) or not 0 <= namespace <= self._MAX_NAMESPACE:
            raise ValueError("namespace must be an integer in [0, 32767]")
        self.namespace = namespace
        self._next_local_id = 1
        self._key_to_id: dict[str, int] = {}

    def register(self, key: str) -> int:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("identity key must be a nonempty string")
        existing = self._key_to_id.get(key)
        if existing is not None:
            return existing
        if self._next_local_id > self._MAX_LOCAL_ID:
            raise OverflowError("identity namespace is exhausted")
        neuron_id = (self.namespace << self._LOCAL_BITS) | self._next_local_id
        self._next_local_id += 1
        self._key_to_id[key] = neuron_id
        return neuron_id

    def resolve(self, key: str) -> int | None:
        return self._key_to_id.get(key)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "namespace": self.namespace,
            "next_local_id": self._next_local_id,
            "entries": dict(sorted(self._key_to_id.items())),
        }

    @classmethod
    def from_state_dict(cls, state: object) -> NeuronIdentityRegistry:
        if not isinstance(state, dict) or state.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported neuron identity registry schema")
        namespace = state.get("namespace")
        registry = cls(namespace)  # validates type and range
        entries = state.get("entries")
        next_local_id = state.get("next_local_id")
        if not isinstance(entries, dict):
            raise TypeError("registry entries must be a mapping")
        if not isinstance(next_local_id, int) or isinstance(next_local_id, bool):
            raise TypeError("next_local_id must be an integer")
        prefix = namespace << cls._LOCAL_BITS
        seen_ids: set[int] = set()
        max_local = 0
        for key, neuron_id in entries.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("registry keys must be nonempty strings")
            if not isinstance(neuron_id, int) or isinstance(neuron_id, bool):
                raise TypeError("registered neuron IDs must be integers")
            local_id = neuron_id - prefix
            if not 1 <= local_id <= cls._MAX_LOCAL_ID or neuron_id in seen_ids:
                raise ValueError("registered neuron ID is invalid or duplicated")
            seen_ids.add(neuron_id)
            max_local = max(max_local, local_id)
        if not 1 <= next_local_id <= cls._MAX_LOCAL_ID + 1 or next_local_id <= max_local:
            raise ValueError("next_local_id must exceed every registered local ID")
        registry._key_to_id = dict(entries)
        registry._next_local_id = next_local_id
        return registry

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> NeuronIdentityRegistry:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_state_dict(json.load(stream))


def _validate_dimensions(batch_size: int, nodes_per_graph: int) -> None:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(nodes_per_graph, int) or isinstance(nodes_per_graph, bool) or nodes_per_graph <= 0:
        raise ValueError("nodes_per_graph must be a positive integer")


def _device_matches(requested: torch.device, actual: torch.device) -> bool:
    if requested.type != actual.type:
        return False
    if requested.index is not None:
        return requested.index == actual.index
    if requested.type == "cuda":
        return actual.index == torch.cuda.current_device()
    return actual.index is None or requested.type == "mps"


@dataclass(frozen=True, slots=True)
class NeuronGraphTopology:
    """Directed edges over a batch of graphs, stored as one sparse edge list.

    Node IDs in ``edge_index`` are flattened as ``batch * nodes_per_graph + node``.
    Edge count is variable and has no per-node degree limit. Optional ``node_ids``
    bind each local slot to a caller-owned stable identity; ``-1`` marks padding.
    """

    batch_size: int
    nodes_per_graph: int
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    strength: torch.Tensor
    node_ids: torch.Tensor | None = None
    edge_group_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.node_ids is not None:
            object.__setattr__(self, "node_ids", self.node_ids.detach().clone())
        if self.edge_group_ids is not None:
            object.__setattr__(self, "edge_group_ids", self.edge_group_ids.detach().clone())

    @property
    def edge_count(self) -> int:
        return int(self.edge_type.numel())

    def remap_node_ids(
        self, new_node_ids: torch.Tensor, *, causal: bool = False
    ) -> NeuronGraphTopology:
        """Reindex a graph by persistent identity, preserving surviving links.

        New identities start disconnected. Removed identities and their incident
        links disappear. The same identity may occur in separate batch graphs,
        but identities must be unique within each graph.
        """
        if self.node_ids is None:
            raise ValueError("identity remapping requires existing node_ids")
        if (
            new_node_ids.dtype != torch.long
            or new_node_ids.ndim != 2
            or new_node_ids.shape[0] != self.batch_size
            or new_node_ids.device != self.edge_index.device
        ):
            raise ValueError("new_node_ids must be long [batch, new_nodes] on the topology device")
        if torch.any(new_node_ids < -1):
            raise ValueError("node IDs must be non-negative or -1 for padding")
        for row in new_node_ids:
            valid = row[row >= 0]
            if torch.unique(valid).numel() != valid.numel():
                raise ValueError("node IDs must be unique within each graph")
        source_values = self.edge_index[0].detach().cpu().tolist()
        target_values = self.edge_index[1].detach().cpu().tolist()
        old_ids = self.node_ids.detach().cpu()
        next_ids = new_node_ids.detach().cpu()
        source_indices: list[int] = []
        target_indices: list[int] = []
        kept_edges: list[int] = []
        new_node_count = new_node_ids.shape[1]
        for edge_id, (source, target) in enumerate(zip(source_values, target_values, strict=True)):
            source_batch, source_slot = divmod(source, self.nodes_per_graph)
            target_batch, target_slot = divmod(target, self.nodes_per_graph)
            if source_batch != target_batch:
                raise ValueError("topology edges cannot cross batch graph boundaries")
            source_identity = int(old_ids[source_batch, source_slot])
            target_identity = int(old_ids[target_batch, target_slot])
            if source_identity < 0 or target_identity < 0:
                continue
            row = next_ids[source_batch]
            source_matches = (row == source_identity).nonzero(as_tuple=True)[0]
            target_matches = (row == target_identity).nonzero(as_tuple=True)[0]
            if not source_matches.numel() or not target_matches.numel():
                continue
            source_new = int(source_matches[0])
            target_new = int(target_matches[0])
            if causal and source_new >= target_new:
                raise ValueError("identity remapping would violate causal edge direction")
            source_indices.append(source_batch * new_node_count + source_new)
            target_indices.append(target_batch * new_node_count + target_new)
            kept_edges.append(edge_id)
        device = self.edge_index.device
        if kept_edges:
            edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long, device=device)
            edge_slots = torch.tensor(kept_edges, dtype=torch.long, device=device)
            edge_type = self.edge_type.index_select(0, edge_slots)
            strength = self.strength.index_select(0, edge_slots)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_slots = torch.empty((0,), dtype=torch.long, device=device)
            edge_type = self.edge_type.new_empty((0,))
            strength = self.strength.new_empty((0,))
        return NeuronGraphTopology(
            self.batch_size, new_node_count, edge_index, edge_type, strength, new_node_ids,
            None if self.edge_group_ids is None else self.edge_group_ids.index_select(0, edge_slots),
        )

    @classmethod
    def empty(
        cls,
        batch_size: int,
        nodes_per_graph: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        node_ids: torch.Tensor | None = None,
    ) -> NeuronGraphTopology:
        _validate_dimensions(batch_size, nodes_per_graph)
        return cls(
            batch_size,
            nodes_per_graph,
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            node_ids,
        )

    @classmethod
    def causal_chain(
        cls,
        batch_size: int,
        nodes_per_graph: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        node_ids: torch.Tensor | None = None,
    ) -> NeuronGraphTopology:
        """Create a causal initial wiring; callers can add any number of links."""

        _validate_dimensions(batch_size, nodes_per_graph)
        if nodes_per_graph <= 1:
            return cls.empty(
                batch_size, nodes_per_graph, device=device, dtype=dtype, node_ids=node_ids
            )
        source = torch.arange(nodes_per_graph - 1, device=device)
        offsets = torch.arange(batch_size, device=device)[:, None] * nodes_per_graph
        source = (source[None, :] + offsets).reshape(-1)
        target = source + 1
        edge_index = torch.stack((source, target))
        count = source.numel()
        return cls(
            batch_size,
            nodes_per_graph,
            edge_index,
            torch.zeros(count, dtype=torch.long, device=device),
            torch.ones(count, dtype=dtype, device=device),
            node_ids,
        )

    def validate(
        self,
        *,
        batch_size: int,
        nodes_per_graph: int,
        num_edge_types: int,
        device: torch.device,
        causal: bool,
        check_values: bool = True,
    ) -> None:
        if (self.batch_size, self.nodes_per_graph) != (batch_size, nodes_per_graph):
            raise ValueError("topology dimensions must match the node tensor")
        if self.edge_index.dtype != torch.long or self.edge_index.shape != (2, self.edge_count):
            raise ValueError("edge_index must be a long tensor with shape [2, edges]")
        if self.edge_type.dtype != torch.long or self.edge_type.shape != (self.edge_count,):
            raise ValueError("edge_type must be a long tensor with one value per edge")
        if self.strength.shape != (self.edge_count,) or not self.strength.is_floating_point():
            raise ValueError("strength must be a floating tensor with one value per edge")
        if self.edge_group_ids is not None and (
            self.edge_group_ids.dtype != torch.long or self.edge_group_ids.shape != (self.edge_count,)
        ):
            raise ValueError("edge_group_ids must be long with one value per edge")
        if any(value.device != device for value in (self.edge_index, self.edge_type, self.strength)):
            raise ValueError("topology tensors must be on the node tensor's device")
        if self.edge_group_ids is not None and self.edge_group_ids.device != device:
            raise ValueError("edge_group_ids must be on the node tensor's device")
        if self.node_ids is not None and (
            self.node_ids.dtype != torch.long
            or self.node_ids.shape != (batch_size, nodes_per_graph)
            or self.node_ids.device != device
        ):
            raise ValueError("topology node_ids must be long [batch, nodes] on the topology device")
        if check_values and self.node_ids is not None:
            if torch.any(self.node_ids < -1):
                raise ValueError("topology node_ids must be non-negative or -1 for padding")
            for row in self.node_ids:
                valid_node_ids = row[row >= 0]
                if torch.unique(valid_node_ids).numel() != valid_node_ids.numel():
                    raise ValueError("topology node_ids must be unique within each graph")
        if self.edge_count == 0 or not check_values:
            return
        node_count = batch_size * nodes_per_graph
        if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= node_count:
            raise ValueError("edge references a node outside the graph batch")
        if int(self.edge_type.min()) < 0 or int(self.edge_type.max()) >= num_edge_types:
            raise ValueError("edge_type is outside the configured relation range")
        source, target = self.edge_index
        if torch.any(source == target):
            raise ValueError("self influence is represented by the neuron update, not a link")
        if torch.any(source // nodes_per_graph != target // nodes_per_graph):
            raise ValueError("edges cannot cross batch graph boundaries")
        if causal and torch.any(source % nodes_per_graph > target % nodes_per_graph):
            raise ValueError("causal topology cannot send information from future nodes")
        if not torch.isfinite(self.strength).all():
            raise ValueError("edge strengths must be finite")
        if torch.any(self.strength < 0):
            raise ValueError("edge strengths must be non-negative")
        ordered = source * node_count + target
        if torch.unique(ordered).numel() != self.edge_count:
            raise ValueError("topology cannot contain duplicate directed edges")

    def filter_context(
        self,
        *,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        num_edge_types: int | None = None,
        causal: bool = True,
    ) -> NeuronGraphTopology:
        """Return edges whose endpoints are valid and belong to one sample segment."""

        if attention_mask is None and segment_ids is None and num_edge_types is None:
            return self
        device = self.edge_index.device
        valid = torch.ones((self.batch_size, self.nodes_per_graph), dtype=torch.bool, device=device)
        if attention_mask is not None:
            if attention_mask.shape != valid.shape or attention_mask.device != device:
                raise ValueError("attention_mask must match topology batch/sequence on its device")
            valid &= attention_mask.to(torch.bool)
        if segment_ids is not None and (segment_ids.shape != valid.shape or segment_ids.device != device):
            raise ValueError("segment_ids must match topology batch/sequence on its device")
        if self.edge_count == 0:
            return self
        source, target = self.edge_index
        node_count = self.batch_size * self.nodes_per_graph
        source_batch = source // self.nodes_per_graph
        target_batch = target // self.nodes_per_graph
        source_pos = source % self.nodes_per_graph
        target_pos = target % self.nodes_per_graph
        in_range = (source >= 0) & (source < node_count) & (target >= 0) & (target < node_count)
        safe_source_batch = source_batch.clamp(0, self.batch_size - 1)
        safe_target_batch = target_batch.clamp(0, self.batch_size - 1)
        safe_source_pos = source_pos.clamp(0, self.nodes_per_graph - 1)
        safe_target_pos = target_pos.clamp(0, self.nodes_per_graph - 1)
        keep = in_range & (
            (source_batch == target_batch)
            & (source_pos != target_pos)
            & valid[safe_source_batch, safe_source_pos]
            & valid[safe_target_batch, safe_target_pos]
        )
        if causal:
            keep &= source_pos < target_pos
        keep &= torch.isfinite(self.strength) & (self.strength >= 0) & (self.edge_type >= 0)
        if num_edge_types is not None:
            keep &= self.edge_type < num_edge_types
        if segment_ids is not None:
            source_segment = segment_ids[safe_source_batch, safe_source_pos]
            target_segment = segment_ids[safe_target_batch, safe_target_pos]
            keep &= source_segment.ge(0) & source_segment.eq(target_segment)
        return NeuronGraphTopology(
            self.batch_size,
            self.nodes_per_graph,
            self.edge_index[:, keep],
            self.edge_type[keep],
            self.strength[keep],
            self.node_ids,
            None if self.edge_group_ids is None else self.edge_group_ids[keep],
        )

    @staticmethod
    def candidate_context_mask(
        candidate_edges: torch.Tensor,
        *,
        batch_size: int,
        nodes_per_graph: int,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mask noncausal, cross-sample, padding, and cross-segment candidates."""

        if candidate_edges.dtype != torch.long or candidate_edges.ndim != 2 or candidate_edges.shape[0] != 2:
            raise ValueError("candidate_edges must be a long tensor with shape [2, candidates]")
        source, target = candidate_edges
        source_batch, target_batch = source // nodes_per_graph, target // nodes_per_graph
        source_pos, target_pos = source % nodes_per_graph, target % nodes_per_graph
        node_count = batch_size * nodes_per_graph
        in_range = (source >= 0) & (source < node_count) & (target >= 0) & (target < node_count)
        safe_source_batch = source_batch.clamp(0, batch_size - 1)
        safe_target_batch = target_batch.clamp(0, batch_size - 1)
        safe_source_pos = source_pos.clamp(0, nodes_per_graph - 1)
        safe_target_pos = target_pos.clamp(0, nodes_per_graph - 1)
        keep = in_range & (source_batch == target_batch) & (source_pos < target_pos)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, nodes_per_graph):
                raise ValueError("attention_mask must match candidate batch/sequence")
            valid = attention_mask.to(device=candidate_edges.device, dtype=torch.bool)
            keep &= valid[safe_source_batch, safe_source_pos] & valid[safe_target_batch, safe_target_pos]
        if segment_ids is not None:
            if segment_ids.shape != (batch_size, nodes_per_graph):
                raise ValueError("segment_ids must match candidate batch/sequence")
            segments = segment_ids.to(candidate_edges.device)
            source_segment = segments[safe_source_batch, safe_source_pos]
            target_segment = segments[safe_target_batch, safe_target_pos]
            keep &= source_segment.ge(0) & source_segment.eq(target_segment)
        return keep

    @staticmethod
    def filter_candidates(
        candidate_edges: torch.Tensor,
        *,
        batch_size: int,
        nodes_per_graph: int,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        keep = NeuronGraphTopology.candidate_context_mask(
            candidate_edges,
            batch_size=batch_size,
            nodes_per_graph=nodes_per_graph,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        return candidate_edges[:, keep]

    @classmethod
    def causal_candidates(
        cls,
        batch_size: int,
        nodes_per_graph: int,
        *,
        device: torch.device | str,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        max_candidates_per_graph: int | None = None,
        randomize: bool = True,
        candidate_offset: int = 0,
    ) -> torch.Tensor:
        """Generate valid causal pairs; optional budget applies per graph.

        ``max_candidates_per_graph`` limits search work, not stored graph
        degree. Deterministic evaluation can advance ``candidate_offset`` to
        scan the complete valid causal pair list over multiple calls. Offsets
        past the end return no candidates; they never wrap to earlier pairs.
        """

        _validate_dimensions(batch_size, nodes_per_graph)
        if max_candidates_per_graph is not None and (
            not isinstance(max_candidates_per_graph, int)
            or isinstance(max_candidates_per_graph, bool)
            or max_candidates_per_graph <= 0
        ):
            raise ValueError("max_candidates_per_graph must be a positive integer or None")
        if not isinstance(candidate_offset, int) or isinstance(candidate_offset, bool) or candidate_offset < 0:
            raise ValueError("candidate_offset must be a non-negative integer")
        device = torch.device(device)
        if attention_mask is not None:
            expected = (batch_size, nodes_per_graph)
            if attention_mask.shape != expected or not _device_matches(device, attention_mask.device):
                raise ValueError("attention_mask must match candidate batch/sequence on its device")
            valid = attention_mask.to(torch.bool)
        else:
            valid = torch.ones((batch_size, nodes_per_graph), dtype=torch.bool, device=device)
        if segment_ids is not None:
            expected = (batch_size, nodes_per_graph)
            if segment_ids.shape != expected or not _device_matches(device, segment_ids.device):
                raise ValueError("segment_ids must match candidate batch/sequence on its device")
            valid &= segment_ids.ge(0)
        selected_graphs = []
        for batch_index in range(batch_size):
            positions = torch.arange(nodes_per_graph, device=device)[valid[batch_index]]
            if segment_ids is None:
                position_groups = (positions,)
            else:
                segment_values = segment_ids[batch_index, positions]
                position_groups = tuple(
                    positions[segment_values == segment]
                    for segment in torch.unique(segment_values, sorted=True)
                )
            groups = []
            for group in position_groups:
                if group.numel() < 2:
                    continue
                group_size = group.numel()
                pair_count = group_size * (group_size - 1) // 2
                groups.append((group, pair_count))
            total_pairs = sum(pair_count for _, pair_count in groups)
            if max_candidates_per_graph is None or total_pairs <= max_candidates_per_graph:
                pair_ids = torch.arange(total_pairs, device=device)
            elif randomize:
                count = max_candidates_per_graph
                if total_pairs <= 4 * count:
                    pair_ids = torch.randperm(total_pairs, device=device)[:count]
                else:
                    pair_ids = torch.empty((0,), dtype=torch.long, device=device)
                    while pair_ids.numel() < count:
                        draws = torch.randint(
                            total_pairs,
                            (max(2 * count, 32),),
                            device=device,
                        )
                        pair_ids = torch.unique(torch.cat((pair_ids, draws)), sorted=True)[:count]
            else:
                start = candidate_offset
                stop = min(total_pairs, start + max_candidates_per_graph)
                pair_ids = torch.arange(start, stop, device=device)
            if pair_ids.numel() == 0:
                selected_graphs.append(torch.empty((2, 0), dtype=torch.long, device=device))
                continue
            group_sizes = torch.tensor(
                [pair_count for _, pair_count in groups], dtype=torch.long, device=device
            )
            boundaries = group_sizes.cumsum(0)
            group_indices = torch.searchsorted(boundaries, pair_ids, right=True)
            graph_parts = []
            start_pair = 0
            for group_index, (group, pair_count) in enumerate(groups):
                belongs = group_indices == group_index
                local_ids = pair_ids[belongs] - start_pair
                start_pair += pair_count
                if local_ids.numel() == 0:
                    continue
                group_size = group.numel()
                row_counts = group_size - 1 - torch.arange(group_size - 1, device=device)
                row_starts = torch.cat(
                    (torch.zeros(1, dtype=torch.long, device=device), row_counts.cumsum(0)[:-1])
                )
                source_index = torch.searchsorted(row_starts, local_ids, right=True) - 1
                target_index = source_index + 1 + local_ids - row_starts[source_index]
                graph_parts.append(
                    torch.stack((group[source_index], group[target_index]))
                    + batch_index * nodes_per_graph
                )
            selected_graphs.append(
                torch.cat(graph_parts, dim=1)
                if graph_parts
                else torch.empty((2, 0), dtype=torch.long, device=device)
            )
        candidates = torch.cat(selected_graphs, dim=1)
        if candidates.numel() == 0:
            return candidates
        node_count = batch_size * nodes_per_graph
        keys = candidates[0] * node_count + candidates[1]
        keys = torch.unique(keys, sorted=True)
        candidates = torch.stack((keys // node_count, keys % node_count))
        return candidates

    def append_causal_nodes(
        self,
        count: int = 1,
        *,
        edge_type: int = 0,
        strength: float = 1.0,
        node_ids: torch.Tensor | None = None,
    ) -> NeuronGraphTopology:
        """Extend every batch graph by ``count`` positions and wire a causal chain."""

        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("count must be a positive integer")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and non-negative")
        old_size = self.nodes_per_graph
        new_size = old_size + count
        if self.node_ids is not None:
            if (
                node_ids is None
                or node_ids.dtype != torch.long
                or node_ids.shape != (self.batch_size, count)
                or node_ids.device != self.edge_index.device
            ):
                raise ValueError("appending identified nodes requires matching new node_ids")
            combined_node_ids = torch.cat((self.node_ids, node_ids), dim=1)
            valid_ids = combined_node_ids[combined_node_ids >= 0]
            if torch.any(node_ids < 0) or torch.unique(valid_ids).numel() != valid_ids.numel():
                raise ValueError("appended node_ids must be globally unique")
        elif node_ids is not None:
            raise ValueError("cannot add identities to a topology whose old nodes are unidentified")
        else:
            combined_node_ids = None
        batch_ids = torch.arange(self.batch_size, device=self.edge_index.device)
        batch_offsets_new = batch_ids * new_size
        local_source = self.edge_index[0] - (self.edge_index[0] // old_size) * old_size
        local_target = self.edge_index[1] - (self.edge_index[1] // old_size) * old_size
        source_batch = self.edge_index[0] // old_size
        target_batch = self.edge_index[1] // old_size
        if torch.any(source_batch != target_batch):
            raise ValueError("cannot extend topology with cross-batch edges")
        remapped = torch.stack((source_batch * new_size + local_source, target_batch * new_size + local_target))
        new_sources = []
        new_targets = []
        for offset in range(count):
            source_local = old_size - 1 + offset
            target_local = source_local + 1
            new_sources.append(batch_offsets_new + source_local)
            new_targets.append(batch_offsets_new + target_local)
        chain = torch.stack((torch.cat(new_sources), torch.cat(new_targets)))
        edge_index = torch.cat((remapped, chain), dim=1)
        new_types = torch.full((self.batch_size * count,), edge_type, dtype=torch.long, device=edge_index.device)
        new_strengths = torch.full(
            (self.batch_size * count,), strength, dtype=self.strength.dtype, device=edge_index.device
        )
        return NeuronGraphTopology(
            self.batch_size,
            new_size,
            edge_index,
            torch.cat((self.edge_type, new_types)),
            torch.cat((self.strength, new_strengths)),
            combined_node_ids,
            None if self.edge_group_ids is None else torch.cat((
                self.edge_group_ids,
                torch.zeros(self.batch_size * count, dtype=torch.long, device=edge_index.device),
            )),
        )


@dataclass(frozen=True, slots=True)
class LinkMutation:
    """A caller- or policy-proposed add, remove, or retype operation."""

    operation: Literal["add", "remove", "retype", "set_strength"]
    batch_index: int
    source: int
    target: int
    edge_type: int | None = None
    strength: float = 1.0
    group_id: int | None = None


class TopologyController:
    """Apply discrete topology changes between propagation steps.

    ``max_edges`` is an optional whole-batch resource budget, never a degree cap.
    Graph topology remains caller-owned and can be carried between model calls.
    """

    def __init__(self, *, max_edges: int | None = None, causal: bool = True) -> None:
        if max_edges is not None and (
            not isinstance(max_edges, int) or isinstance(max_edges, bool) or max_edges <= 0
        ):
            raise ValueError("max_edges must be a positive integer or None")
        if not isinstance(causal, bool):
            raise TypeError("causal must be bool")
        self.max_edges = max_edges
        self.causal = causal

    def apply(
        self,
        topology: NeuronGraphTopology,
        mutations: list[LinkMutation] | tuple[LinkMutation, ...],
        *,
        num_edge_types: int,
    ) -> NeuronGraphTopology:
        if not isinstance(num_edge_types, int) or isinstance(num_edge_types, bool) or num_edge_types <= 0:
            raise ValueError("num_edge_types must be a positive integer")
        topology.validate(
            batch_size=topology.batch_size,
            nodes_per_graph=topology.nodes_per_graph,
            num_edge_types=num_edge_types,
            device=topology.edge_index.device,
            causal=self.causal,
        )
        source_ids = topology.edge_index[0].detach().cpu().tolist()
        target_ids = topology.edge_index[1].detach().cpu().tolist()
        edge_types = topology.edge_type.detach().cpu().tolist()
        strengths = topology.strength.detach().cpu().tolist()
        group_ids = (
            topology.edge_group_ids.detach().cpu().tolist()
            if topology.edge_group_ids is not None
            else [None] * topology.edge_count
        )
        edges: dict[tuple[int, int], tuple[int, float, int | None]] = {
            (source, target): (kind, strength, group_id)
            for source, target, kind, strength, group_id in zip(
                source_ids, target_ids, edge_types, strengths, group_ids, strict=True
            )
        }
        if self.max_edges is not None and len(edges) > self.max_edges:
            raise ValueError("topology exceeds the global edge budget")

        for mutation in mutations:
            self._validate_mutation(mutation, topology, num_edge_types)
            offset = mutation.batch_index * topology.nodes_per_graph
            source, target = offset + mutation.source, offset + mutation.target
            key = (source, target)
            exists = key in edges
            if mutation.operation == "add":
                if exists:
                    raise ValueError("cannot add a link that already exists")
                assert mutation.edge_type is not None
                edges[key] = (mutation.edge_type, float(mutation.strength), mutation.group_id)
            elif mutation.operation == "remove":
                if not exists:
                    raise ValueError("cannot remove a link that does not exist")
                del edges[key]
            elif mutation.operation == "set_strength":
                if not exists:
                    raise ValueError("cannot update the strength of a link that does not exist")
                edges[key] = (edges[key][0], float(mutation.strength), edges[key][2])
            else:
                if not exists:
                    raise ValueError("cannot retype a link that does not exist")
                assert mutation.edge_type is not None
                group_id = edges[key][2] if mutation.group_id is None else mutation.group_id
                edges[key] = (mutation.edge_type, float(mutation.strength), group_id)

        if self.max_edges is not None and len(edges) > self.max_edges:
            raise ValueError("mutation exceeds the global edge budget")
        ordered_edges = list(edges.items())
        device = topology.edge_index.device
        dtype = topology.strength.dtype
        if ordered_edges:
            edge_index = torch.tensor(
                [[edge[0][0] for edge in ordered_edges], [edge[0][1] for edge in ordered_edges]],
                dtype=torch.long,
                device=device,
            )
            next_types = torch.tensor([edge[1][0] for edge in ordered_edges], dtype=torch.long, device=device)
            next_strengths = torch.tensor([edge[1][1] for edge in ordered_edges], dtype=dtype, device=device)
            has_group_ids = topology.edge_group_ids is not None or any(edge[1][2] is not None for edge in ordered_edges)
            next_group_ids = (
                torch.tensor([0 if edge[1][2] is None else edge[1][2] for edge in ordered_edges], dtype=torch.long, device=device)
                if has_group_ids else None
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            next_types = torch.empty((0,), dtype=torch.long, device=device)
            next_strengths = torch.empty((0,), dtype=dtype, device=device)
            next_group_ids = topology.edge_group_ids.new_empty((0,)) if topology.edge_group_ids is not None else None
        result = NeuronGraphTopology(
            topology.batch_size,
            topology.nodes_per_graph,
            edge_index,
            next_types,
            next_strengths,
            topology.node_ids,
            next_group_ids,
        )
        result.validate(
            batch_size=result.batch_size,
            nodes_per_graph=result.nodes_per_graph,
            num_edge_types=num_edge_types,
            device=device,
            causal=self.causal,
        )
        return result

    def _validate_mutation(
        self,
        mutation: LinkMutation,
        topology: NeuronGraphTopology,
        num_edge_types: int,
    ) -> None:
        if mutation.operation not in {"add", "remove", "retype", "set_strength"}:
            raise ValueError("unsupported topology mutation")
        if not isinstance(mutation.batch_index, int) or isinstance(mutation.batch_index, bool) or not 0 <= mutation.batch_index < topology.batch_size:
            raise ValueError("batch_index is outside the graph batch")
        if not isinstance(mutation.source, int) or isinstance(mutation.source, bool) or not 0 <= mutation.source < topology.nodes_per_graph:
            raise ValueError("source is outside its graph")
        if not isinstance(mutation.target, int) or isinstance(mutation.target, bool) or not 0 <= mutation.target < topology.nodes_per_graph:
            raise ValueError("target is outside its graph")
        if mutation.source == mutation.target:
            raise ValueError("self influence is represented by the neuron update, not a link")
        if self.causal and mutation.source > mutation.target:
            raise ValueError("causal topology cannot send information from future nodes")
        if mutation.operation in {"add", "retype"} and (
            not isinstance(mutation.edge_type, int)
            or isinstance(mutation.edge_type, bool)
            or not 0 <= mutation.edge_type < num_edge_types
        ):
            raise ValueError("add/retype requires a valid edge_type")
        if mutation.operation == "set_strength" and mutation.edge_type is not None:
            raise ValueError("set_strength cannot change the edge_type")
        if mutation.group_id is not None and (
            not isinstance(mutation.group_id, int) or isinstance(mutation.group_id, bool)
        ):
            raise ValueError("group_id must be an integer or None")
        if mutation.operation == "set_strength" and mutation.group_id is not None:
            raise ValueError("set_strength cannot change the group_id")
        if mutation.operation == "add" and topology.edge_group_ids is not None and mutation.group_id is None:
            raise ValueError("adding to a grouped topology requires group_id")
        if mutation.operation in {"add", "retype", "set_strength"} and (
            not math.isfinite(mutation.strength) or mutation.strength < 0
        ):
            raise ValueError("edge strength must be finite and non-negative")


__all__ = ["LinkMutation", "NeuronGraphTopology", "TopologyController"]
