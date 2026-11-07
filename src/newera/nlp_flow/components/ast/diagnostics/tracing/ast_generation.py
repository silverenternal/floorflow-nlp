"""Structure-aware decoding for Stage-A AST flow outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from newera.nlp_flow.components.ast.flow.core.ast_cfm_data import PAD, ASTCodecVocabulary, ASTFlowOutput
from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTNode, parse_ast, render_source
from newera.nlp_flow.components.generation.trees.typed.typed_tree import from_ast_node, tree_signature

_ROOT_TYPES = {
    "python": "module",
    "rust": "source_file",
    "cpp": "translation_unit",
}


@dataclass(frozen=True)
class StructuredASTDecoding:
    source: str
    tree: ASTNode
    node_mask: torch.Tensor
    parent_indices: torch.Tensor
    level_ids: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.node_mask.sum())


def predict_generated_topology(
    output: ASTFlowOutput,
    node_mask: torch.Tensor,
    *,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = node_mask[batch_index].bool()
    active_indices = active.nonzero(as_tuple=False).flatten()
    if not active_indices.numel():
        raise ValueError("generated AST has no active nodes")
    root_index = int(active_indices[0])
    parents = torch.full_like(output.root_logits[batch_index], -2, dtype=torch.long)
    parents[root_index] = -1
    for index in active_indices.tolist():
        if index == root_index:
            continue
        if index == 0:
            raise ValueError("generated AST second root has no preceding parent")
        candidates = torch.arange(index, device=parents.device)
        candidates = candidates[active[:index]]
        if not candidates.numel():
            raise ValueError("generated AST node has no legal parent")
        parent_scores = output.parent_logits[
            batch_index,
            index,
            candidates + 1,
        ]
        parents[index] = candidates[int(parent_scores.argmax())]
    levels = output.level_logits[batch_index].argmax(dim=-1)
    return parents, levels


def predict_generated_topology_batch(
    output: ASTFlowOutput,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    levels = []
    for batch_index in range(node_mask.shape[0]):
        parents, level_ids = predict_generated_topology(
            output,
            node_mask,
            batch_index=batch_index,
        )
        rows.append(parents)
        levels.append(level_ids)
    return torch.stack(rows), torch.stack(levels)


def source_token_plan(
    node_mask: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_mask.shape != parent_indices.shape:
        raise ValueError("node mask and parent indices must align")
    node_rows = []
    phase_rows = []
    for batch_index in range(node_mask.shape[0]):
        active = node_mask[batch_index].bool()
        parents = parent_indices[batch_index]
        children: dict[int, list[int]] = {
            index: []
            for index in active.nonzero(as_tuple=False).flatten().tolist()
        }
        for index in children:
            parent = int(parents[index])
            if parent >= 0:
                children[parent].append(index)
        leaves = []

        def visit(
            index: int,
            children=children,
            leaves=leaves,
        ) -> None:
            if not children[index]:
                leaves.append(index)
                return
            for child in children[index]:
                visit(child)

        root = next(
            (
                index
                for index in children
                if int(parents[index]) == -1
            ),
            None,
        )
        if root is None:
            raise ValueError("source token plan requires one AST root")
        visit(root)
        node_ids = [root]
        phases = [0]
        for leaf in leaves:
            node_ids.extend((leaf, leaf))
            phases.extend((1, 2))
        node_ids.append(root)
        phases.append(0)
        if len(node_ids) > max_tokens:
            raise ValueError("source token plan exceeds decoder capacity")
        padding = max_tokens - len(node_ids)
        node_rows.append(node_ids + [0] * padding)
        phase_rows.append(phases + [0] * padding)
    return (
        torch.tensor(node_rows, dtype=torch.long, device=node_mask.device),
        torch.tensor(phase_rows, dtype=torch.long, device=node_mask.device),
    )


def decode_structured_ast(
    output: ASTFlowOutput,
    vocabulary: ASTCodecVocabulary,
    language: Literal["python", "rust", "cpp"],
    *,
    node_mask: torch.Tensor | None = None,
    batch_index: int = 0,
) -> StructuredASTDecoding:
    """Build and verify a tree from predicted topology, fields, and lexemes."""

    if node_mask is None:
        node_mask = output.predicted_node_mask()
    if node_mask.shape != output.node_type_logits.shape[:2]:
        raise ValueError("node_mask must match generated AST slots")
    if not 0 <= batch_index < node_mask.shape[0]:
        raise ValueError("batch_index is outside generated AST output")
    mask = node_mask[batch_index].bool()
    parents, levels = predict_generated_topology(
        output,
        node_mask,
        batch_index=batch_index,
    )
    active_indices = mask.nonzero(as_tuple=False).flatten().tolist()
    root_index = next(
        index for index in active_indices if int(parents[index]) == -1
    )
    node_names = vocabulary.node_types
    edge_names = vocabulary.edge_fields
    lexical_names = vocabulary.lexemes
    language_nodes = vocabulary.node_types_by_language.get(
        language,
        node_names,
    )
    language_edges = vocabulary.edge_fields_by_language.get(
        language,
        edge_names,
    )
    language_lexemes = vocabulary.lexemes_by_language.get(
        language,
        lexical_names,
    )
    node_allowed = [
        node_names.index(value)
        for value in language_nodes
    ]
    edge_allowed = [
        edge_names.index(value)
        for value in language_edges
    ]
    lexical_allowed = [
        lexical_names.index(value)
        for value in language_lexemes
    ]
    node_logits = output.node_type_logits[batch_index].index_select(
        dim=-1,
        index=torch.tensor(node_allowed, device=output.node_type_logits.device),
    )
    predicted_nodes = torch.tensor(
        node_allowed,
        device=output.node_type_logits.device,
    )[node_logits.argmax(dim=-1)]
    root_type_id = node_names.index(_ROOT_TYPES[language])
    if root_type_id not in node_allowed:
        raise ValueError("language root type is absent from the AST vocabulary")
    predicted_nodes[root_index] = root_type_id
    predicted_edges = torch.tensor(
        edge_allowed,
        device=output.edge_type_logits.device,
    )[
        output.edge_type_logits[batch_index]
        .index_select(
            dim=-1,
            index=torch.tensor(
                edge_allowed,
                device=output.edge_type_logits.device,
            ),
        )
        .argmax(dim=-1)
    ]
    pad_index = vocabulary.lexemes.index(PAD)
    lexical_allowed = [
        index for index in lexical_allowed
        if index != pad_index
    ]
    if not lexical_allowed:
        raise ValueError("language has no non-padding AST lexemes")
    lexical_logits = output.lexical_logits[batch_index].index_select(
        dim=-1,
        index=torch.tensor(
            lexical_allowed,
            device=output.lexical_logits.device,
        ),
    )
    predicted_lexemes = torch.tensor(
        lexical_allowed,
        device=output.lexical_logits.device,
    )[lexical_logits.argmax(dim=-1)]
    predicted_gaps = output.gap_logits[batch_index].argmax(dim=-1)
    by_parent: dict[int, list[int]] = {index: [] for index in active_indices}
    for index in active_indices:
        parent = int(parents[index])
        if parent >= 0:
            by_parent[parent].append(index)

    def build(index: int, *, is_root: bool = False) -> ASTNode:
        children = tuple(
            (
                vocabulary.edge_fields[int(predicted_edges[child])],
                build(child),
            )
            for child in by_parent[index]
        )
        lexical_id = int(predicted_lexemes[index])
        lexical = vocabulary.lexemes[lexical_id]
        gap = vocabulary.gaps[int(predicted_gaps[index])]
        if gap == PAD:
            gap = ""
        value = None if children or lexical == PAD else lexical
        if not children and value is None:
            raise ValueError("generated AST leaf has no lexical token")
        return ASTNode(
            node_type=vocabulary.node_types[int(predicted_nodes[index])],
            start_byte=0,
            end_byte=0,
            named=True,
            children=children,
            lexical_value=value,
            gap_before=gap,
            language=language if is_root else None,
        )

    predicted_root = build(root_index, is_root=True)
    if predicted_root.node_type != _ROOT_TYPES[language]:
        raise ValueError(
            "generated AST root type does not match the declared language: "
            f"{predicted_root.node_type}"
        )
    source = render_source(predicted_root)
    try:
        parsed = parse_ast(source, language)
    except ValueError as error:
        raise ValueError(
            f"{language} generated AST source does not parse: {source!r}"
        ) from error
    if tree_signature(from_ast_node(predicted_root)) != tree_signature(
        from_ast_node(parsed)
    ):
        raise ValueError(
            "predicted node/edge structure disagrees with parsed source"
        )
    return StructuredASTDecoding(
        source=source,
        tree=parsed,
        node_mask=mask[None],
        parent_indices=parents[mask],
        level_ids=levels[mask],
    )


__all__ = [
    "StructuredASTDecoding",
    "decode_structured_ast",
    "predict_generated_topology",
    "predict_generated_topology_batch",
    "source_token_plan",
]
