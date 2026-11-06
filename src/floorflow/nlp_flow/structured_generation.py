"""Single structured-generation gateway over the shared typed-tree contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.json_tree import build_json_tree, strict_json_loads
from floorflow.nlp_flow.output_contract import (
    OutputContractDecision,
    validate_output_contract,
)
from floorflow.nlp_flow.typed_tree import (
    MODES,
    TypedTreeNode,
    TypedTreeValidity,
    from_ast_node,
    from_json_node,
    render_typed_tree,
)


@dataclass(frozen=True)
class StructuredGenerationResult:
    mode: str
    text: str
    tree: TypedTreeNode
    validity: TypedTreeValidity


class TypedTreeGateway:
    """Route text/code/JSON/AST results through one typed-tree finish path."""

    @staticmethod
    def select_mode(mode_logits: torch.Tensor) -> str:
        if mode_logits.ndim == 3:
            mode_logits = mode_logits[:, -1]
        if mode_logits.ndim != 2 or mode_logits.shape[-1] != len(MODES):
            raise ValueError("mode logits must be [batch, 4] or [batch, seq, 4]")
        return MODES[int(mode_logits[0].argmax())]

    @staticmethod
    def from_ast(source: str, language: str) -> StructuredGenerationResult:
        ast = parse_ast(source, language)
        tree = from_ast_node(ast)
        return StructuredGenerationResult(
            mode="ast",
            text=render_typed_tree(tree),
            tree=tree,
            validity=TypedTreeValidity(
                structural=True,
                schema=True,
                executable=False,
            ),
        )

    @staticmethod
    def from_json(
        value_or_text: Any,
        *,
        domain: str = "json",
        schema_valid: bool = False,
    ) -> StructuredGenerationResult:
        if domain not in {"json", "patch", "tool_action"}:
            raise ValueError(f"JSON-tree gateway does not support {domain}")
        value = (
            strict_json_loads(value_or_text)
            if isinstance(value_or_text, str)
            else value_or_text
        )
        tree = from_json_node(build_json_tree(value), domain=domain)
        return StructuredGenerationResult(
            mode=domain,
            text=render_typed_tree(tree),
            tree=tree,
            validity=TypedTreeValidity(
                structural=True,
                schema=schema_valid,
                executable=False,
            ),
        )

    @staticmethod
    def contract_decision(
        result: StructuredGenerationResult,
    ) -> OutputContractDecision:
        return validate_output_contract(result.mode, result.validity)

    @staticmethod
    def require_contract(
        result: StructuredGenerationResult,
    ) -> StructuredGenerationResult:
        TypedTreeGateway.contract_decision(result).require_valid()
        return result

    @staticmethod
    def with_validity(
        result: StructuredGenerationResult,
        *,
        schema: bool,
        executable: bool,
    ) -> StructuredGenerationResult:
        return StructuredGenerationResult(
            mode=result.mode,
            text=result.text,
            tree=result.tree,
            validity=TypedTreeValidity(
                structural=result.validity.structural,
                schema=schema,
                executable=executable,
            ),
        )


__all__ = [
    "MODES",
    "StructuredGenerationResult",
    "TypedTreeGateway",
]
