"""Typed tensors and codec helpers for AST flow matching."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import torch

from floorflow.nlp_flow.ast_codec import ASTNode, render_source

AST_LEVELS = ("root", "subtree", "leaf")
PAD = "<pad>"


@dataclass(frozen=True)
class ASTTensorBatch:
    node_mask: torch.Tensor
    node_types: torch.Tensor
    edge_fields: torch.Tensor
    lexemes: torch.Tensor
    gaps: torch.Tensor
    parent_indices: torch.Tensor
    levels: torch.Tensor
    level_masks: dict[str, torch.Tensor]
    parseability_targets: torch.Tensor
    template_ids: torch.Tensor
    source_token_ids: torch.Tensor
    source_token_mask: torch.Tensor
    source_token_node_ids: torch.Tensor
    source_token_phase_ids: torch.Tensor

    def to(self, device: torch.device | str) -> ASTTensorBatch:
        return ASTTensorBatch(
            node_mask=self.node_mask.to(device),
            node_types=self.node_types.to(device),
            edge_fields=self.edge_fields.to(device),
            lexemes=self.lexemes.to(device),
            gaps=self.gaps.to(device),
            parent_indices=self.parent_indices.to(device),
            levels=self.levels.to(device),
            level_masks={
                name: value.to(device)
                for name, value in self.level_masks.items()
            },
            parseability_targets=self.parseability_targets.to(device),
            template_ids=self.template_ids.to(device),
            source_token_ids=self.source_token_ids.to(device),
            source_token_mask=self.source_token_mask.to(device),
            source_token_node_ids=self.source_token_node_ids.to(device),
            source_token_phase_ids=self.source_token_phase_ids.to(device),
        )


@dataclass(frozen=True)
class ASTFlowOutput:
    velocity: torch.Tensor
    flow_endpoint: torch.Tensor
    endpoint: torch.Tensor
    node_type_logits: torch.Tensor
    edge_type_logits: torch.Tensor
    lexical_logits: torch.Tensor
    gap_logits: torch.Tensor
    level_logits: torch.Tensor
    parent_logits: torch.Tensor
    root_logits: torch.Tensor
    node_count_logits: torch.Tensor
    source_length_logits: torch.Tensor
    uncertainty_log_std: torch.Tensor
    template_logits: torch.Tensor
    parseability_logits: torch.Tensor
    level_velocities: dict[str, torch.Tensor]
    source_token_logits: torch.Tensor | None = None
    source_token_free_logits: torch.Tensor | None = None
    source_token_teacher_logits: torch.Tensor | None = None
    source_token_ids: torch.Tensor | None = None
    executed_nfe: int = 0

    def as_training_dict(self) -> dict[str, torch.Tensor]:
        output = {
            "velocity": self.velocity,
            "flow_endpoint": self.flow_endpoint,
            "endpoint": self.endpoint,
            "node_type_logits": self.node_type_logits,
            "edge_type_logits": self.edge_type_logits,
            "lexical_logits": self.lexical_logits,
            "gap_logits": self.gap_logits,
            "level_logits": self.level_logits,
            "parent_logits": self.parent_logits,
            "root_logits": self.root_logits,
            "node_count_logits": self.node_count_logits,
            "source_length_logits": self.source_length_logits,
            "uncertainty_log_std": self.uncertainty_log_std,
            "template_logits": self.template_logits,
            "parseability_logits": self.parseability_logits,
        }
        if self.source_token_logits is not None:
            output["source_token_logits"] = self.source_token_logits
        if self.source_token_free_logits is not None:
            output["source_token_free_logits"] = self.source_token_free_logits
        if self.source_token_teacher_logits is not None:
            output["source_token_teacher_logits"] = (
                self.source_token_teacher_logits
            )
        return output

    def quality_score(self, node_mask: torch.Tensor) -> torch.Tensor:
        """Return a confidence proxy for downstream Stage-B gating."""

        if node_mask.shape != self.node_type_logits.shape[:2]:
            raise ValueError("node_mask must match AST output")
        mask = node_mask.to(self.node_type_logits.device)

        def masked_confidence(logits: torch.Tensor) -> torch.Tensor:
            values = torch.softmax(logits, dim=-1).amax(dim=-1)
            return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        node = masked_confidence(self.node_type_logits)
        edge = masked_confidence(self.edge_type_logits)
        lexical = masked_confidence(self.lexical_logits)
        gap = masked_confidence(self.gap_logits)
        parse = torch.sigmoid(self.parseability_logits)
        node_count = torch.softmax(self.node_count_logits, dim=-1).amax(dim=-1)
        root = torch.sigmoid(self.root_logits)
        root_confidence = (root * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return (
            (node + edge + lexical + gap) * 0.15
            + parse * 0.15
            + node_count * 0.15
            + root_confidence * 0.1
        ).clamp(0.0, 1.0)

    def predicted_node_count(self) -> torch.Tensor:
        return self.node_count_logits.argmax(dim=-1).clamp_min(1)

    def predicted_node_mask(
        self,
        *,
        max_nodes: int | None = None,
    ) -> torch.Tensor:
        limit = (
            self.node_count_logits.shape[-1] - 1
            if max_nodes is None
            else int(max_nodes)
        )
        if not 1 <= limit <= self.node_count_logits.shape[-1] - 1:
            raise ValueError("max_nodes is outside the node-count head")
        counts = self.predicted_node_count().clamp_max(limit)
        slots = torch.arange(limit, device=counts.device)[None]
        return slots.lt(counts[:, None])


@dataclass(frozen=True)
class ASTObjectiveLosses:
    total: torch.Tensor
    velocity: torch.Tensor
    endpoint: torch.Tensor
    node_type: torch.Tensor
    edge_type: torch.Tensor
    lexical: torch.Tensor
    gap: torch.Tensor
    level: torch.Tensor
    parent: torch.Tensor
    root: torch.Tensor
    node_count: torch.Tensor
    source_length: torch.Tensor
    uncertainty: torch.Tensor
    template: torch.Tensor
    parseability: torch.Tensor
    source_tokens: torch.Tensor
    source_tokens_free: torch.Tensor
    source_tokens_teacher: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "velocity": self.velocity,
            "endpoint": self.endpoint,
            "node_type": self.node_type,
            "edge_type": self.edge_type,
            "lexical": self.lexical,
            "gap": self.gap,
            "level": self.level,
            "parent": self.parent,
            "root": self.root,
            "node_count": self.node_count,
            "source_length": self.source_length,
            "uncertainty": self.uncertainty,
            "template": self.template,
            "parseability": self.parseability,
            "source_tokens": self.source_tokens,
            "source_tokens_free": self.source_tokens_free,
            "source_tokens_teacher": self.source_tokens_teacher,
        }


@dataclass(frozen=True)
class ASTCodecVocabulary:
    node_types: tuple[str, ...]
    edge_fields: tuple[str, ...]
    lexemes: tuple[str, ...]
    gaps: tuple[str, ...]
    templates: tuple[str, ...]
    source_tokens: tuple[str, ...] = ()
    node_types_by_language: dict[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )
    edge_fields_by_language: dict[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )
    lexemes_by_language: dict[str, tuple[str, ...]] = dataclass_field(
        default_factory=dict
    )

    @classmethod
    def from_graphs(cls, graphs: list[ASTNode] | tuple[ASTNode, ...]) -> ASTCodecVocabulary:
        node_types: set[str] = {PAD}
        edge_fields: set[str] = {PAD}
        lexemes: set[str] = {PAD}
        gaps: set[str] = {PAD}
        templates: set[str] = set()
        source_tokens: set[str] = {"<pad>", "<bos>", "<eos>"}
        node_types_by_language: dict[str, set[str]] = {}
        edge_fields_by_language: dict[str, set[str]] = {}
        lexemes_by_language: dict[str, set[str]] = {}
        for graph in graphs:
            templates.add(render_source(graph))
            language = graph.language
            if language is None:
                raise ValueError("AST graph root must declare a language")
            language_nodes = node_types_by_language.setdefault(
                language,
                set(),
            )
            language_edges = edge_fields_by_language.setdefault(
                language,
                set(),
            )
            language_lexemes = lexemes_by_language.setdefault(
                language,
                set(),
            )
            for _path, node in _walk(graph):
                node_types.add(node.node_type)
                language_nodes.add(node.node_type)
                if node.lexical_value is not None:
                    lexemes.add(node.lexical_value)
                    language_lexemes.add(node.lexical_value)
                    gaps.add(node.gap_before)
                    source_tokens.add(f"gap:{node.gap_before}")
                    source_tokens.add(f"lex:{node.lexical_value}")
            for field, _child in _walk_edges(graph):
                edge_fields.add(field)
                language_edges.add(field)
        return cls(
            node_types=tuple(sorted(node_types)),
            edge_fields=tuple(sorted(edge_fields)),
            lexemes=tuple(sorted(lexemes)),
            gaps=tuple(sorted(gaps)),
            templates=tuple(sorted(templates)),
            source_tokens=(
                "<pad>",
                "<bos>",
                "<eos>",
                *sorted(source_tokens - {"<pad>", "<bos>", "<eos>"}),
            ),
            node_types_by_language={
                language: tuple(sorted(values))
                for language, values in node_types_by_language.items()
            },
            edge_fields_by_language={
                language: tuple(sorted(values))
                for language, values in edge_fields_by_language.items()
            },
            lexemes_by_language={
                language: tuple(sorted(values))
                for language, values in lexemes_by_language.items()
            },
        )

    @property
    def node_type_to_id(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.node_types)}

    @property
    def edge_field_to_id(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.edge_fields)}

    @property
    def lexeme_to_id(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.lexemes)}

    @property
    def num_lexemes(self) -> int:
        return len(self.lexemes)

    @property
    def gap_to_id(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.gaps)}

    @property
    def num_gaps(self) -> int:
        return len(self.gaps)

    @property
    def template_to_id(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.templates)}


@dataclass(frozen=True)
class _FlatNode:
    node: ASTNode
    parent_index: int
    edge_field: str
    level: int


def _walk(node: ASTNode) -> list[tuple[tuple[int, ...], ASTNode]]:
    output = []

    def visit(current: ASTNode, path: tuple[int, ...]) -> None:
        output.append((path, current))
        for index, (_field, child) in enumerate(current.children):
            visit(child, path + (index,))

    visit(node, ())
    return output


def _walk_edges(node: ASTNode) -> list[tuple[str, ASTNode]]:
    output = []
    for field, child in node.children:
        output.append((field, child))
        output.extend(_walk_edges(child))
    return output


def _flatten_graph(root: ASTNode) -> list[_FlatNode]:
    output: list[_FlatNode] = []

    def visit(current: ASTNode, parent_index: int, edge_field: str) -> int:
        index = len(output)
        leaf = not current.children
        level = 0 if not output else 2 if leaf else 1
        output.append(_FlatNode(current, parent_index, edge_field, level))
        for field, child in current.children:
            visit(child, index, field)
        return index

    visit(root, -1, PAD)
    return output


def encode_ast_graph(
    graph: ASTNode,
    vocabulary: ASTCodecVocabulary,
    *,
    max_nodes: int,
    template_id: int = 0,
) -> ASTTensorBatch:
    flat = _flatten_graph(graph)
    if len(flat) > max_nodes:
        raise ValueError(f"AST has {len(flat)} nodes, exceeding max_nodes={max_nodes}")

    node_types = torch.zeros(max_nodes, dtype=torch.long)
    edge_fields = torch.zeros(max_nodes, dtype=torch.long)
    lexemes = torch.zeros(max_nodes, dtype=torch.long)
    gaps = torch.zeros(max_nodes, dtype=torch.long)
    parent_indices = torch.full((max_nodes,), -1, dtype=torch.long)
    levels = torch.full((max_nodes,), 3, dtype=torch.long)
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_type_to_id = vocabulary.node_type_to_id
    edge_field_to_id = vocabulary.edge_field_to_id
    lexeme_to_id = vocabulary.lexeme_to_id
    gap_to_id = vocabulary.gap_to_id
    for index, item in enumerate(flat):
        node_mask[index] = True
        node_types[index] = node_type_to_id[item.node.node_type]
        edge_fields[index] = edge_field_to_id[item.edge_field]
        lexemes[index] = lexeme_to_id[item.node.lexical_value or PAD]
        gaps[index] = gap_to_id[item.node.gap_before]
        parent_indices[index] = item.parent_index
        levels[index] = item.level

    source_token_ids, source_token_node_ids, source_token_phase_ids = (
        _source_token_data(
            graph,
            vocabulary,
            max_nodes=max_nodes,
        )
    )
    return ASTTensorBatch(
        node_mask=node_mask,
        node_types=node_types,
        edge_fields=edge_fields,
        lexemes=lexemes,
        gaps=gaps,
        parent_indices=parent_indices,
        levels=levels,
        level_masks={
            "root": node_mask & levels.eq(0),
            "subtree": node_mask & levels.eq(1),
            "leaf": node_mask & levels.eq(2),
        },
        parseability_targets=torch.ones(1, dtype=torch.bool),
        template_ids=torch.tensor([template_id], dtype=torch.long),
        source_token_ids=torch.tensor(source_token_ids, dtype=torch.long),
        source_token_mask=torch.tensor(
            [value != 0 for value in source_token_ids],
            dtype=torch.bool,
        ),
        source_token_node_ids=torch.tensor(
            source_token_node_ids,
            dtype=torch.long,
        ),
        source_token_phase_ids=torch.tensor(
            source_token_phase_ids,
            dtype=torch.long,
        ),
    )


def _source_leaves(node: ASTNode) -> list[ASTNode]:
    if not node.children:
        return [node]
    output = []
    for _field, child in node.children:
        output.extend(_source_leaves(child))
    return output


def _source_token_data(
    graph: ASTNode,
    vocabulary: ASTCodecVocabulary,
    *,
    max_nodes: int,
) -> tuple[list[int], list[int], list[int]]:
    token_to_id = {
        token: index for index, token in enumerate(vocabulary.source_tokens)
    }
    values = [token_to_id["<bos>"]]
    node_ids = [0]
    phase_ids = [0]
    for leaf_index, leaf in zip(
        _source_leaf_indices(graph),
        _source_leaves(graph),
        strict=True,
    ):
        values.append(token_to_id[f"gap:{leaf.gap_before}"])
        node_ids.append(leaf_index)
        phase_ids.append(1)
        if leaf.lexical_value is None:
            raise ValueError("AST source leaf requires a lexical value")
        values.append(token_to_id[f"lex:{leaf.lexical_value}"])
        node_ids.append(leaf_index)
        phase_ids.append(2)
    values.append(token_to_id["<eos>"])
    node_ids.append(0)
    phase_ids.append(0)
    maximum = max_nodes * 2 + 1
    if len(values) > maximum:
        raise ValueError("AST source token sequence exceeds the node capacity")
    padding = maximum - len(values)
    return (
        values + [0] * padding,
        node_ids + [0] * padding,
        phase_ids + [0] * padding,
    )


def _source_leaf_indices(graph: ASTNode) -> list[int]:
    indices = []

    def visit(node: ASTNode, index: int) -> int:
        next_index = index + 1
        if not node.children:
            indices.append(index)
        for _field, child in node.children:
            next_index = visit(child, next_index)
        return next_index

    visit(graph, 0)
    return indices


def collate_ast_batches(batches: list[ASTTensorBatch]) -> ASTTensorBatch:
    if not batches:
        raise ValueError("cannot collate an empty AST batch")
    return ASTTensorBatch(
        node_mask=torch.stack([batch.node_mask for batch in batches]),
        node_types=torch.stack([batch.node_types for batch in batches]),
        edge_fields=torch.stack([batch.edge_fields for batch in batches]),
        lexemes=torch.stack([batch.lexemes for batch in batches]),
        gaps=torch.stack([batch.gaps for batch in batches]),
        parent_indices=torch.stack([batch.parent_indices for batch in batches]),
        levels=torch.stack([batch.levels for batch in batches]),
        level_masks={
            name: torch.stack([batch.level_masks[name] for batch in batches])
            for name in AST_LEVELS
        },
        parseability_targets=torch.cat(
            [batch.parseability_targets.reshape(-1) for batch in batches]
        ),
        template_ids=torch.cat([batch.template_ids.reshape(-1) for batch in batches]),
        source_token_ids=torch.stack(
            [batch.source_token_ids for batch in batches]
        ),
        source_token_mask=torch.stack(
            [batch.source_token_mask for batch in batches]
        ),
        source_token_node_ids=torch.stack(
            [batch.source_token_node_ids for batch in batches]
        ),
        source_token_phase_ids=torch.stack(
            [batch.source_token_phase_ids for batch in batches]
        ),
    )


def prompt_context(
    prompt: str,
    hidden_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode prompt bytes without consulting any target AST."""

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    return (values / math.sqrt(hidden_size)).to(dtype=dtype)


__all__ = [
    "AST_LEVELS",
    "PAD",
    "ASTCodecVocabulary",
    "ASTFlowOutput",
    "ASTObjectiveLosses",
    "ASTTensorBatch",
    "collate_ast_batches",
    "encode_ast_graph",
    "prompt_context",
]
