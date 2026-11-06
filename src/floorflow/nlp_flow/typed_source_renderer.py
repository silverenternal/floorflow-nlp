"""Pure source projection from committed AST and typed lexical decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from floorflow.nlp_flow.ast_codec import ASTGraph, ASTNode, parse_ast_graph
from floorflow.nlp_flow.ast_leaf_classifier import classify_leaf
from floorflow.nlp_flow.typed_lexical_contract import TypedLexicalValue


class TypedSourceRenderError(ValueError):
    """Base error for invalid source projection decisions."""


class MissingLexicalDecisionError(TypedSourceRenderError):
    pass


class InvalidLexicalDecisionError(TypedSourceRenderError):
    pass


class SourceReplayMismatchError(TypedSourceRenderError):
    pass


@dataclass(frozen=True)
class RenderedSource:
    language: str
    source: str
    source_sha256: str
    canonical_signature: tuple[object, ...]
    parsed_graph: ASTGraph


def _leaf_ids(root: ASTNode) -> tuple[int, ...]:
    output = []

    def visit(node: ASTNode, next_id: int) -> int:
        current = next_id
        next_id += 1
        if not node.children:
            output.append(current)
        for _field, child in node.children:
            next_id = visit(child, next_id)
        return next_id

    visit(root, 0)
    return tuple(output)


def _expected_signature(
    node: ASTNode,
    decisions: Mapping[int, TypedLexicalValue],
    counter: list[int],
) -> tuple[object, ...]:
    node_id = counter[0]
    counter[0] += 1
    if node.children:
        lexical = None
        gap = None
        children = tuple(
            (field, _expected_signature(child, decisions, counter))
            for field, child in node.children
        )
    else:
        value = decisions[node_id]
        lexical = value.value
        gap = node.gap_before
        children = ()
    return node.node_type, node.named, node.scope_id, lexical, gap, children


def _parsed_signature(node: ASTNode) -> tuple[object, ...]:
    children = tuple((field, _parsed_signature(child)) for field, child in node.children)
    return (
        node.node_type,
        node.named,
        node.scope_id,
        node.lexical_value if not node.children else None,
        node.gap_before if not node.children else None,
        children,
    )


def render_typed_source(
    graph: ASTGraph,
    lexical_values: Sequence[TypedLexicalValue] | Mapping[int, TypedLexicalValue],
) -> RenderedSource:
    """Render every leaf exactly once, then require parser identity replay."""
    if isinstance(lexical_values, Mapping):
        decisions = dict(lexical_values)
    else:
        decisions = {}
        for value in lexical_values:
            if value.node_id in decisions:
                raise InvalidLexicalDecisionError(f"duplicate lexical decision for node {value.node_id}")
            decisions[value.node_id] = value
    expected_ids = set(_leaf_ids(graph.root))
    observed_ids = set(decisions)
    if expected_ids != observed_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        error = MissingLexicalDecisionError if missing else InvalidLexicalDecisionError
        raise error(f"lexical decision set differs from AST leaves; missing={missing}, extra={extra}")

    parts: list[str] = []

    def emit(node: ASTNode, counter: list[int]) -> None:
        node_id = counter[0]
        counter[0] += 1
        if not node.children:
            decision = decisions[node_id]
            if (
                decision.node_id != node_id
                or decision.node_kind != node.node_type
                or decision.scope_id != node.scope_id
                or decision.origin not in {"user_available", "generated"}
                or decision.minimum_length < 0
                or decision.maximum_length < decision.minimum_length
                or not decision.minimum_length <= len(decision.value) <= decision.maximum_length
                or classify_leaf(node.node_type, decision.value) != decision.leaf_class
            ):
                raise InvalidLexicalDecisionError(f"typed lexical decision does not match AST leaf {node_id}")
            parts.extend((node.gap_before, decision.value))
            return
        if node.lexical_value is not None:
            raise InvalidLexicalDecisionError("internal AST nodes cannot carry lexical text")
        for _field, child in node.children:
            emit(child, counter)

    emit(graph.root, [0])
    source = "".join(parts)
    try:
        replay = parse_ast_graph(source, graph.language)
    except (TypeError, ValueError, UnicodeError) as error:
        raise SourceReplayMismatchError(f"rendered source does not parse: {error}") from error
    signature = _expected_signature(graph.root, decisions, [0])
    if _parsed_signature(replay.root) != signature:
        raise SourceReplayMismatchError("rendered source changed the committed AST structure or lexical leaves")
    return RenderedSource(
        language=graph.language,
        source=source,
        source_sha256=replay.source_sha256,
        canonical_signature=signature,
        parsed_graph=replay,
    )


__all__ = [
    "InvalidLexicalDecisionError",
    "MissingLexicalDecisionError",
    "RenderedSource",
    "SourceReplayMismatchError",
    "TypedSourceRenderError",
    "render_typed_source",
]
