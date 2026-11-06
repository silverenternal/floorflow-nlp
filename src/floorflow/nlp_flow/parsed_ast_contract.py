"""Exact parser receipts and model qualification bindings for Stage-B handoff."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from floorflow.nlp_flow.ast_codec import SERIALIZATION_VERSION, ASTGraph, ASTNode, parser_fingerprint

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ReceiptMode = Literal["fixture", "production"]


def _hash(value: str, field: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _node_payload(node: ASTNode) -> dict[str, object]:
    return {
        "type": node.node_type,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "named": node.named,
        "lexical_value": node.lexical_value,
        "gap_before": node.gap_before,
        "scope_id": node.scope_id,
        "symbol_name": node.symbol_name,
        "symbol_kind": node.symbol_kind,
        "language": node.language,
        "children": [
            {"field": field, "node": _node_payload(child)}
            for field, child in node.children
        ],
    }


def full_ordered_graph_sha256(graph: ASTGraph) -> str:
    """Hash every ordered node, edge, lexical, scope and symbol value."""
    return _digest(
        {
            "language": graph.language,
            "source_sha256": graph.source_sha256,
            "root": _node_payload(graph.root),
            "scopes": [
                {
                    "scope_id": scope.scope_id,
                    "node_type": scope.node_type,
                    "start_byte": scope.start_byte,
                    "end_byte": scope.end_byte,
                    "parent_scope_id": scope.parent_scope_id,
                    "path": list(scope.path),
                }
                for scope in graph.scopes
            ],
            "symbols": [
                {
                    "name": symbol.name,
                    "kind": symbol.kind,
                    "node_type": symbol.node_type,
                    "scope_id": symbol.scope_id,
                    "scope_path": list(symbol.scope_path),
                    "start_byte": symbol.start_byte,
                    "end_byte": symbol.end_byte,
                }
                for symbol in graph.symbols
            ],
        }
    )


@dataclass(frozen=True)
class ModelQualificationBinding:
    checkpoint_sha256: str
    tokenizer_sha256: str
    parser_sha256: str
    canonicalizer_sha256: str
    evaluator_sha256: str
    languages: tuple[str, ...]
    tracks: tuple[str, ...]
    freeze_manifest_sha256: str
    mode: ReceiptMode

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256", "tokenizer_sha256", "parser_sha256",
            "canonicalizer_sha256", "evaluator_sha256", "freeze_manifest_sha256",
        ):
            _hash(getattr(self, name), name)
        if not self.languages or tuple(sorted(set(self.languages))) != self.languages:
            raise ValueError("qualification languages must be unique and sorted")
        if not self.tracks or tuple(sorted(set(self.tracks))) != self.tracks:
            raise ValueError("qualification tracks must be unique and sorted")
        if self.mode not in {"fixture", "production"}:
            raise ValueError("unknown model qualification mode")

    @property
    def qualification_id(self) -> str:
        return _digest(
            {
                "checkpoint": self.checkpoint_sha256,
                "tokenizer": self.tokenizer_sha256,
                "parser": self.parser_sha256,
                "canonicalizer": self.canonicalizer_sha256,
                "evaluator": self.evaluator_sha256,
                "languages": self.languages,
                "tracks": self.tracks,
                "freeze_manifest": self.freeze_manifest_sha256,
                "mode": self.mode,
            }
        )


@dataclass(frozen=True)
class ParsedASTReceipt:
    source_sha256: str
    full_ordered_canonical_graph_sha256: str
    task_id: str
    task_and_input_fingerprint: str
    model_qualification_id: str
    language: str
    track: str
    parser_version: str
    canonicalizer_version: str
    grammar_version: str
    mode: ReceiptMode

    def __post_init__(self) -> None:
        for name in (
            "source_sha256", "full_ordered_canonical_graph_sha256",
            "task_and_input_fingerprint", "model_qualification_id",
        ):
            _hash(getattr(self, name), name)
        if not all((self.task_id, self.language, self.track, self.parser_version, self.canonicalizer_version, self.grammar_version)):
            raise ValueError("receipt task, language, and version bindings are required")
        if self.mode not in {"fixture", "production"}:
            raise ValueError("unknown receipt mode")


def task_input_fingerprint(task_id: str, input_sha256: str) -> str:
    if not task_id:
        raise ValueError("task identity cannot be empty")
    return _digest({"task_id": task_id, "input_sha256": _hash(input_sha256, "input_sha256")})


def parser_runtime_identity(language: str) -> tuple[str, str]:
    fingerprint = parser_fingerprint(language)
    details = {
        "language": fingerprint.language,
        "package": fingerprint.package,
        "package_version": fingerprint.package_version,
        "abi_version": fingerprint.abi_version,
        "semantic_version": fingerprint.semantic_version,
        "native_sha256": fingerprint.native_sha256,
        "grammar_sha256": fingerprint.grammar_sha256,
    }
    version = (
        f"{fingerprint.package}@{fingerprint.package_version}:abi{fingerprint.abi_version}"
        f":grammar-{fingerprint.grammar_sha256}"
    )
    return version, _digest(details)


def canonicalizer_runtime_identity() -> tuple[str, str]:
    from floorflow.nlp_flow import ast_codec, ast_codec_io

    sources = {}
    for module in (ast_codec, ast_codec_io):
        path = Path(module.__file__ or "")
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return SERIALIZATION_VERSION, _digest(sources)


def issue_parsed_ast_receipt(
    graph: ASTGraph,
    source: str,
    *,
    task_id: str,
    track: str,
    input_sha256: str,
    qualification: ModelQualificationBinding,
    parser_version: str,
    canonicalizer_version: str,
    grammar_version: str,
    mode: ReceiptMode,
) -> ParsedASTReceipt:
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_hash != graph.source_sha256:
        raise ValueError("parser graph source hash differs from the supplied source")
    if graph.language not in qualification.languages:
        raise ValueError("model qualification does not cover this language")
    if track not in qualification.tracks:
        raise ValueError("model qualification does not cover this track")
    if mode != qualification.mode:
        raise ValueError("receipt mode differs from the model qualification mode")
    if mode == "fixture" and qualification.mode == "production":
        raise ValueError("fixture receipts cannot be issued from production qualification")
    return ParsedASTReceipt(
        source_sha256=source_hash,
        full_ordered_canonical_graph_sha256=full_ordered_graph_sha256(graph),
        task_id=task_id,
        task_and_input_fingerprint=task_input_fingerprint(task_id, input_sha256),
        model_qualification_id=qualification.qualification_id,
        language=graph.language,
        track=track,
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=grammar_version,
        mode=mode,
    )


def verify_parsed_ast_receipt(
    receipt: ParsedASTReceipt,
    graph: ASTGraph,
    source: str,
    *,
    task_id: str,
    track: str,
    input_sha256: str,
    qualification: ModelQualificationBinding,
    parser_version: str,
    canonicalizer_version: str,
    grammar_version: str,
) -> None:
    expected = issue_parsed_ast_receipt(
        graph,
        source,
        task_id=task_id,
        track=track,
        input_sha256=input_sha256,
        qualification=qualification,
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=grammar_version,
        mode=receipt.mode,
    )
    if receipt != expected:
        raise ValueError("parsed AST receipt is stale or bound to another task, input, graph, or version")


__all__ = [
    "ModelQualificationBinding",
    "ParsedASTReceipt",
    "canonicalizer_runtime_identity",
    "full_ordered_graph_sha256",
    "issue_parsed_ast_receipt",
    "parser_runtime_identity",
    "task_input_fingerprint",
    "verify_parsed_ast_receipt",
]
