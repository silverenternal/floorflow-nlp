"""Decoding helpers for Stage-A AST flow outputs."""

from __future__ import annotations

from typing import Literal

import torch

from floorflow.nlp_flow.ast_cfm_data import PAD, ASTCodecVocabulary, ASTFlowOutput
from floorflow.nlp_flow.ast_codec import ASTNode, ast_metrics, parse_ast
from floorflow.nlp_flow.structured_generation import TypedTreeGateway
from floorflow.nlp_flow.typed_tree import TypedTreeNode


def decode_generated_source(
    output: ASTFlowOutput,
    node_mask: torch.Tensor,
    vocabulary: ASTCodecVocabulary,
    *,
    batch_index: int = 0,
) -> str:
    """Decode leaf lexemes selected by the generated level head."""

    if output.lexical_logits.ndim != 3:
        raise ValueError("lexical logits must be [batch, nodes, vocabulary]")
    if node_mask.shape != output.lexical_logits.shape[:2]:
        raise ValueError("node_mask must match output slots")
    active = node_mask[batch_index]
    predicted_lexemes = output.lexical_logits[batch_index].argmax(dim=-1)
    predicted_gaps = output.gap_logits[batch_index].argmax(dim=-1)
    values = []
    for index in active.nonzero(as_tuple=False).flatten().tolist():
        gap_value = vocabulary.gaps[int(predicted_gaps[index])]
        lexical_value = vocabulary.lexemes[int(predicted_lexemes[index])]
        if gap_value != PAD:
            values.append(gap_value)
        if lexical_value != PAD:
            values.append(lexical_value)
    if not values:
        raise ValueError("generated AST has no lexical content")
    return "".join(values)


def decode_generated_template(
    output: ASTFlowOutput,
    vocabulary: ASTCodecVocabulary,
    *,
    batch_index: int = 0,
    allow_template_shortcut: bool = False,
) -> str:
    """Select a parser-validated grammar shell from native model logits."""

    if not allow_template_shortcut:
        raise ValueError(
            "grammar-template shortcuts are disabled for AST synthesis"
        )
    template_id = int(output.template_logits[batch_index].argmax())
    return vocabulary.templates[template_id]


def validate_generated_ast(
    output: ASTFlowOutput,
    node_mask: torch.Tensor,
    vocabulary: ASTCodecVocabulary,
    language: Literal["python", "rust", "cpp"],
    reference: ASTNode,
) -> tuple[ASTNode | None, dict[str, float | bool] | None, str]:
    try:
        source = decode_generated_source(output, node_mask, vocabulary)
        generated = parse_ast(source, language)
    except (TypeError, ValueError, UnicodeError):
        return None, None, ""
    return generated, ast_metrics(generated, reference), source


def decode_generated_typed_tree(
    output: ASTFlowOutput,
    node_mask: torch.Tensor,
    vocabulary: ASTCodecVocabulary,
    language: Literal["python", "rust", "cpp"],
) -> tuple[TypedTreeNode, str]:
    source = decode_generated_source(output, node_mask, vocabulary)
    structured = TypedTreeGateway.from_ast(source, language)
    return structured.tree, structured.text


__all__ = [
    "decode_generated_source",
    "decode_generated_template",
    "decode_generated_typed_tree",
    "validate_generated_ast",
]
