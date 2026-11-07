"""Typed Stage-B conditioning from the exact parser-receipted AST graph."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ASTNode
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import ParsedASTReceipt, full_ordered_graph_sha256


@dataclass(frozen=True)
class ParsedASTEncoding:
    states: torch.Tensor
    mask: torch.Tensor
    node_types: tuple[str, ...]
    parent_indices: torch.Tensor
    edge_fields: tuple[str, ...]
    lexical_values: tuple[str | None, ...]


@dataclass(frozen=True)
class StageBConditioning:
    """Separate prompt, parser-graph and test channels with no latent endpoint slot."""

    prompt_states: torch.Tensor
    prompt_mask: torch.Tensor
    structure_states: torch.Tensor
    structure_mask: torch.Tensor
    test_states: torch.Tensor
    test_mask: torch.Tensor

    def __post_init__(self) -> None:
        channels = (
            (self.prompt_states, self.prompt_mask),
            (self.structure_states, self.structure_mask),
            (self.test_states, self.test_mask),
        )
        width = self.prompt_states.shape[-1]
        batch_size = self.prompt_states.shape[0]
        for states, mask in channels:
            if (
                states.ndim != 3
                or states.shape[0] != batch_size
                or mask.shape != states.shape[:2]
                or states.shape[-1] != width
            ):
                raise ValueError("Stage-B conditioning channels must be separately masked [batch, items, width]")
            if mask.dtype != torch.bool:
                raise TypeError("Stage-B channel masks must be boolean")


def _flatten(graph: ASTGraph) -> tuple[list[ASTNode], list[int], list[str]]:
    nodes: list[ASTNode] = []
    parents: list[int] = []
    edges: list[str] = []

    def visit(node: ASTNode, parent: int, edge: str) -> None:
        index = len(nodes)
        nodes.append(node)
        parents.append(parent)
        edges.append(edge)
        for field, child in node.children:
            visit(child, index, field)

    visit(graph.root, -1, "__root__")
    return nodes, parents, edges


class ParsedASTEncoder(nn.Module):
    """Encode topology and lexical values from a receipt-checked AST graph only."""

    def __init__(
        self,
        node_type_ids: Mapping[str, int],
        edge_field_ids: Mapping[str, int],
        scope_type_ids: Mapping[str, int],
        *,
        hidden_size: int = 128,
        heads: int = 4,
        layers: int = 2,
        max_nodes: int = 4096,
        lexical_buckets: int = 8192,
    ) -> None:
        super().__init__()
        if hidden_size % heads or min(hidden_size, layers, max_nodes, lexical_buckets) < 1:
            raise ValueError("invalid parsed AST encoder dimensions")
        if any(value <= 0 for mapping in (node_type_ids, edge_field_ids, scope_type_ids) for value in mapping.values()):
            raise ValueError("parsed AST vocabularies reserve ID zero for padding")
        self.node_type_ids = dict(node_type_ids)
        self.edge_field_ids = dict(edge_field_ids)
        self.scope_type_ids = dict(scope_type_ids)
        self.max_nodes = max_nodes
        self.lexical_buckets = lexical_buckets
        self.node_embedding = nn.Embedding(max(self.node_type_ids.values(), default=0) + 2, hidden_size)
        self.parent_type_embedding = nn.Embedding(max(self.node_type_ids.values(), default=0) + 2, hidden_size)
        self.edge_embedding = nn.Embedding(max(self.edge_field_ids.values(), default=0) + 2, hidden_size)
        self.scope_embedding = nn.Embedding(max(self.scope_type_ids.values(), default=0) + 2, hidden_size)
        self.lexical_embedding = nn.Embedding(lexical_buckets + 1, hidden_size)
        self.position_embedding = nn.Embedding(max_nodes, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            heads,
            hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.RMSNorm(hidden_size)

    def _id(self, value: str, vocabulary: Mapping[str, int]) -> int:
        return vocabulary.get(value, 1)

    def _lexical_id(self, value: str | None) -> int:
        if value is None:
            return 0
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return 1 + int.from_bytes(digest[:8], "big") % self.lexical_buckets

    def forward(self, graph: ASTGraph, receipt: ParsedASTReceipt) -> ParsedASTEncoding:
        if (
            receipt.language != graph.language
            or receipt.source_sha256 != graph.source_sha256
            or receipt.full_ordered_canonical_graph_sha256 != full_ordered_graph_sha256(graph)
        ):
            raise ValueError("parsed AST graph does not match its parser receipt")
        nodes, parents, edges = _flatten(graph)
        if not nodes or len(nodes) > self.max_nodes:
            raise ValueError("parsed AST graph exceeds the encoder node limit")
        scope_types = {scope.scope_id: scope.node_type for scope in graph.scopes}
        device = self.node_embedding.weight.device
        node_ids = torch.tensor(
            [self._id(node.node_type, self.node_type_ids) for node in nodes],
            dtype=torch.long,
            device=device,
        )
        parent_ids = torch.tensor(
            [self._id(nodes[parent].node_type, self.node_type_ids) if parent >= 0 else 0 for parent in parents],
            dtype=torch.long,
            device=device,
        )
        edge_ids = torch.tensor(
            [self._id(edge, self.edge_field_ids) for edge in edges], dtype=torch.long, device=device
        )
        scope_ids = torch.tensor(
            [self._id(scope_types.get(node.scope_id, ""), self.scope_type_ids) for node in nodes],
            dtype=torch.long,
            device=device,
        )
        lexical_ids = torch.tensor(
            [self._lexical_id(node.lexical_value) for node in nodes], dtype=torch.long, device=device
        )
        positions = torch.arange(len(nodes), dtype=torch.long, device=device)
        values = (
            self.node_embedding(node_ids)
            + self.parent_type_embedding(parent_ids)
            + self.edge_embedding(edge_ids)
            + self.scope_embedding(scope_ids)
            + self.lexical_embedding(lexical_ids)
            + self.position_embedding(positions)
        )
        states = self.norm(self.encoder(values[None]))
        return ParsedASTEncoding(
            states=states,
            mask=torch.ones((1, len(nodes)), dtype=torch.bool, device=states.device),
            node_types=tuple(node.node_type for node in nodes),
            parent_indices=torch.tensor(parents, dtype=torch.long, device=states.device)[None],
            edge_fields=tuple(edges),
            lexical_values=tuple(node.lexical_value for node in nodes),
        )


__all__ = ["ParsedASTEncoder", "ParsedASTEncoding", "StageBConditioning"]
