"""Single-parse, receipt-checked handoff from generated source to Stage B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from floorflow.nlp_flow.ast_codec import ASTGraph, parse_ast_graph
from floorflow.nlp_flow.parsed_ast_contract import (
    ModelQualificationBinding,
    ParsedASTReceipt,
    canonicalizer_runtime_identity,
    issue_parsed_ast_receipt,
    parser_runtime_identity,
    task_input_fingerprint,
    verify_parsed_ast_receipt,
)
from floorflow.nlp_flow.parsed_ast_encoder import ParsedASTEncoder
from floorflow.nlp_flow.result_cfm import GeneratedASTResultCFM, ResultCFMOutput

Language = Literal["python", "rust", "cpp"]


@dataclass(frozen=True)
class ParsedASTRuntimeRequest:
    task_ids: tuple[str, ...]
    input_sha256s: tuple[str, ...]
    track: str
    qualification: ModelQualificationBinding
    parser_version: str
    canonicalizer_version: str
    grammar_version: str

    def __post_init__(self) -> None:
        if not self.task_ids or len(self.task_ids) != len(self.input_sha256s):
            raise ValueError("runtime request task and input hash batches must be non-empty and aligned")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("runtime task IDs must be unique within a batch")
        for task_id, input_hash in zip(self.task_ids, self.input_sha256s, strict=True):
            task_input_fingerprint(task_id, input_hash)
        if not all((self.track, self.parser_version, self.canonicalizer_version, self.grammar_version)):
            raise ValueError("runtime track and parser/canonicalizer/grammar versions are required")
        if self.qualification.mode != "production":
            raise ValueError("final Stage-B handoff requires a production model qualification")


@dataclass(frozen=True)
class ParsedASTBatch:
    graphs: tuple[ASTGraph, ...]
    receipts: tuple[ParsedASTReceipt, ...]
    states: torch.Tensor
    mask: torch.Tensor
    parent_indices: torch.Tensor
    node_levels: torch.Tensor


@dataclass(frozen=True)
class ParsedASTStageBResult:
    parsed_ast: ParsedASTBatch
    result: ResultCFMOutput


def _node_levels(parent_indices: torch.Tensor) -> torch.Tensor:
    levels = torch.zeros_like(parent_indices)
    for index in range(parent_indices.shape[1]):
        parent = int(parent_indices[0, index])
        if parent >= 0:
            levels[0, index] = min(3, int(levels[0, parent]) + 1)
    return levels


def parse_and_encode_generated_sources(
    sources: tuple[str, ...] | list[str],
    *,
    language: Language,
    request: ParsedASTRuntimeRequest,
    encoder: ParsedASTEncoder,
) -> ParsedASTBatch:
    """Parse each source once, issue/verify a receipt, then encode that exact graph."""
    if len(sources) != len(request.task_ids):
        raise ValueError("generated sources must match the receipt request batch")
    graphs = tuple(parse_ast_graph(source, language) for source in sources)
    receipts = tuple(
        issue_parsed_ast_receipt(
            graph,
            source,
            task_id=task_id,
            track=request.track,
            input_sha256=input_hash,
            qualification=request.qualification,
            parser_version=request.parser_version,
            canonicalizer_version=request.canonicalizer_version,
            grammar_version=request.grammar_version,
            mode="production",
        )
        for graph, source, task_id, input_hash in zip(
            graphs, sources, request.task_ids, request.input_sha256s, strict=True
        )
    )
    return encode_verified_parsed_graphs(
        graphs,
        receipts,
        sources,
        language=language,
        request=request,
        encoder=encoder,
    )


def encode_verified_parsed_graphs(
    graphs: tuple[ASTGraph, ...] | list[ASTGraph],
    receipts: tuple[ParsedASTReceipt, ...] | list[ParsedASTReceipt],
    sources: tuple[str, ...] | list[str],
    *,
    language: Language,
    request: ParsedASTRuntimeRequest,
    encoder: ParsedASTEncoder,
) -> ParsedASTBatch:
    """Verify and encode an existing receipt graph without parsing source again."""
    if not (len(graphs) == len(receipts) == len(sources) == len(request.task_ids)):
        raise ValueError("parsed graphs, receipts, sources and runtime request must align")
    if language not in request.qualification.languages:
        raise ValueError("production qualification does not cover the requested language")
    if request.track not in request.qualification.tracks:
        raise ValueError("production qualification does not cover this track")
    active_parser_version, active_parser_hash = parser_runtime_identity(language)
    active_canonicalizer_version, active_canonicalizer_hash = canonicalizer_runtime_identity()
    if (
        request.parser_version != active_parser_version
        or request.qualification.parser_sha256 != active_parser_hash
    ):
        raise ValueError("runtime parser differs from the frozen qualification")
    active_grammar_version = active_parser_version.rsplit(":grammar-", maxsplit=1)[-1]
    if request.grammar_version != active_grammar_version:
        raise ValueError("runtime grammar differs from the receipt version")
    if (
        request.canonicalizer_version != active_canonicalizer_version
        or request.qualification.canonicalizer_sha256 != active_canonicalizer_hash
    ):
        raise ValueError("runtime canonicalizer differs from the frozen qualification")
    graph_values = []
    receipt_values = []
    encodings = []
    for graph, receipt, source, task_id, input_hash in zip(
        graphs, receipts, sources, request.task_ids, request.input_sha256s, strict=True
    ):
        verify_parsed_ast_receipt(
            receipt,
            graph,
            source,
            task_id=task_id,
            track=request.track,
            input_sha256=input_hash,
            qualification=request.qualification,
            parser_version=request.parser_version,
            canonicalizer_version=request.canonicalizer_version,
            grammar_version=request.grammar_version,
        )
        graph_values.append(graph)
        receipt_values.append(receipt)
        encodings.append(encoder(graph, receipt))
    device = encodings[0].states.device
    hidden = encodings[0].states.shape[-1]
    max_nodes = max(encoding.states.shape[1] for encoding in encodings)
    states = encodings[0].states.new_zeros((len(encodings), max_nodes, hidden))
    mask = torch.zeros((len(encodings), max_nodes), dtype=torch.bool, device=device)
    parents = torch.full((len(encodings), max_nodes), -1, dtype=torch.long, device=device)
    levels = torch.zeros((len(encodings), max_nodes), dtype=torch.long, device=device)
    for index, encoding in enumerate(encodings):
        count = encoding.states.shape[1]
        states[index, :count] = encoding.states[0]
        mask[index, :count] = True
        parents[index, :count] = encoding.parent_indices[0]
        levels[index, :count] = _node_levels(encoding.parent_indices)[0]
    return ParsedASTBatch(tuple(graph_values), tuple(receipt_values), states, mask, parents, levels)


@torch.no_grad()
def sample_stage_b_from_parsed_sources(
    stage_b: GeneratedASTResultCFM,
    prompt_context: torch.Tensor,
    sources: tuple[str, ...] | list[str],
    *,
    language: Language,
    request: ParsedASTRuntimeRequest,
    encoder: ParsedASTEncoder,
    result_length: int,
    steps: int | None = None,
    noise: torch.Tensor | None = None,
) -> ParsedASTStageBResult:
    parsed = parse_and_encode_generated_sources(
        sources,
        language=language,
        request=request,
        encoder=encoder,
    )
    if prompt_context.shape[0] != parsed.states.shape[0]:
        raise ValueError("prompt and parsed AST batches must align")
    encoder_device = next(encoder.parameters()).device
    if encoder_device != prompt_context.device:
        raise ValueError("parsed AST encoder and prompt must be on the same device")
    if parsed.states.shape[1] > stage_b.conditioner.position_embedding.num_embeddings:
        raise ValueError("parsed AST exceeds the Stage-B topology position capacity")
    result = stage_b.sample(
        prompt_context,
        parsed.states,
        parsed.mask,
        parent_indices=parsed.parent_indices,
        node_levels=parsed.node_levels,
        result_length=result_length,
        steps=steps,
        noise=noise,
    )
    return ParsedASTStageBResult(parsed, result)


@torch.no_grad()
def sample_stage_b_from_parsed_receipts(
    stage_b: GeneratedASTResultCFM,
    prompt_context: torch.Tensor,
    graphs: tuple[ASTGraph, ...] | list[ASTGraph],
    receipts: tuple[ParsedASTReceipt, ...] | list[ParsedASTReceipt],
    sources: tuple[str, ...] | list[str],
    *,
    language: Language,
    request: ParsedASTRuntimeRequest,
    encoder: ParsedASTEncoder,
    result_length: int,
    steps: int | None = None,
    noise: torch.Tensor | None = None,
) -> ParsedASTStageBResult:
    """Run Stage B from the exact AST graph/receipt committed by the strict gateway."""
    parsed = encode_verified_parsed_graphs(
        graphs,
        receipts,
        sources,
        language=language,
        request=request,
        encoder=encoder,
    )
    if prompt_context.shape[0] != parsed.states.shape[0]:
        raise ValueError("prompt and parsed AST batches must align")
    if next(encoder.parameters()).device != prompt_context.device:
        raise ValueError("parsed AST encoder and prompt must be on the same device")
    if parsed.states.shape[1] > stage_b.conditioner.position_embedding.num_embeddings:
        raise ValueError("parsed AST exceeds the Stage-B topology position capacity")
    result = stage_b.sample(
        prompt_context,
        parsed.states,
        parsed.mask,
        parent_indices=parsed.parent_indices,
        node_levels=parsed.node_levels,
        result_length=result_length,
        steps=steps,
        noise=noise,
    )
    return ParsedASTStageBResult(parsed, result)


__all__ = [
    "ParsedASTBatch",
    "ParsedASTRuntimeRequest",
    "ParsedASTStageBResult",
    "encode_verified_parsed_graphs",
    "parse_and_encode_generated_sources",
    "sample_stage_b_from_parsed_receipts",
    "sample_stage_b_from_parsed_sources",
]
