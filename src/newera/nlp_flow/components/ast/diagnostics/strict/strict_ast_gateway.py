"""Commit typed AST decisions to deterministic source and a parser receipt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from newera.nlp_flow.components.ast.representation.codec.ast_codec import SUPPORTED_LANGUAGES, ASTGraph
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
    ModelQualificationBinding,
    ParsedASTReceipt,
    canonicalizer_runtime_identity,
    issue_parsed_ast_receipt,
    parser_runtime_identity,
    verify_parsed_ast_receipt,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import TreeAction
from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import TypedLexicalValue
from newera.nlp_flow.components.generation.trees.typed.typed_source_renderer import (
    InvalidLexicalDecisionError,
    MissingLexicalDecisionError,
    SourceReplayMismatchError,
    TypedSourceRenderError,
    render_typed_source,
)

FailureKind = Literal["unsupported", "invalid_decision", "parse_or_replay_failure", "receipt_failure"]


@dataclass(frozen=True)
class ASTGatewaySuccess:
    source: str
    source_sha256: str
    parsed_graph: ASTGraph
    receipt: ParsedASTReceipt
    original_actions: tuple[TreeAction, ...]


@dataclass(frozen=True)
class ASTGatewayFailure:
    kind: FailureKind
    message: str
    original_actions: tuple[TreeAction, ...]


def commit_typed_ast(
    committed_graph: ASTGraph,
    lexical_values: tuple[TypedLexicalValue, ...] | list[TypedLexicalValue],
    original_actions: tuple[TreeAction, ...] | list[TreeAction],
    *,
    task_id: str,
    track: str,
    input_sha256: str,
    qualification: ModelQualificationBinding,
    parser_version: str,
    canonicalizer_version: str,
    grammar_version: str,
) -> ASTGatewaySuccess | ASTGatewayFailure:
    """Render one committed IR, retain the single replay graph, and bind its receipt."""
    actions = tuple(original_actions)
    if committed_graph.language not in SUPPORTED_LANGUAGES:
        return ASTGatewayFailure("unsupported", "language is not supported by the frozen parser", actions)
    if (
        committed_graph.language not in qualification.languages
        or track not in qualification.tracks
    ):
        return ASTGatewayFailure("receipt_failure", "model qualification scope does not cover this task", actions)
    if qualification.mode == "production":
        parser_id, parser_hash = parser_runtime_identity(committed_graph.language)
        canonicalizer_id, canonicalizer_hash = canonicalizer_runtime_identity()
        grammar_id = parser_id.rsplit(":grammar-", maxsplit=1)[-1]
        if parser_version != parser_id or qualification.parser_sha256 != parser_hash:
            return ASTGatewayFailure("receipt_failure", "runtime parser differs from the frozen qualification", actions)
        if (
            canonicalizer_version != canonicalizer_id
            or qualification.canonicalizer_sha256 != canonicalizer_hash
        ):
            return ASTGatewayFailure("receipt_failure", "runtime canonicalizer differs from the frozen qualification", actions)
        if grammar_version != grammar_id:
            return ASTGatewayFailure("receipt_failure", "runtime grammar differs from the receipt version", actions)
    try:
        rendered = render_typed_source(committed_graph, lexical_values)
    except NotImplementedError as error:
        return ASTGatewayFailure("unsupported", str(error), actions)
    except (MissingLexicalDecisionError, InvalidLexicalDecisionError) as error:
        return ASTGatewayFailure("invalid_decision", str(error), actions)
    except SourceReplayMismatchError as error:
        return ASTGatewayFailure("parse_or_replay_failure", str(error), actions)
    except TypedSourceRenderError as error:
        return ASTGatewayFailure("invalid_decision", str(error), actions)
    except (TypeError, ValueError, UnicodeError) as error:
        return ASTGatewayFailure("parse_or_replay_failure", str(error), actions)

    try:
        receipt = issue_parsed_ast_receipt(
            rendered.parsed_graph,
            rendered.source,
            task_id=task_id,
            track=track,
            input_sha256=input_sha256,
            qualification=qualification,
            parser_version=parser_version,
            canonicalizer_version=canonicalizer_version,
            grammar_version=grammar_version,
            mode=qualification.mode,
        )
        verify_parsed_ast_receipt(
            receipt,
            rendered.parsed_graph,
            rendered.source,
            task_id=task_id,
            track=track,
            input_sha256=input_sha256,
            qualification=qualification,
            parser_version=parser_version,
            canonicalizer_version=canonicalizer_version,
            grammar_version=grammar_version,
        )
    except (TypeError, ValueError) as error:
        return ASTGatewayFailure("receipt_failure", str(error), actions)
    return ASTGatewaySuccess(
        source=rendered.source,
        source_sha256=rendered.source_sha256,
        parsed_graph=rendered.parsed_graph,
        receipt=receipt,
        original_actions=actions,
    )


__all__ = ["ASTGatewayFailure", "ASTGatewaySuccess", "commit_typed_ast"]
