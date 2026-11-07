from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import pytest
import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
    ModelQualificationBinding,
    canonicalizer_runtime_identity,
    parser_runtime_identity,
)
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_encoder import ParsedASTEncoder
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_handoff import (
    ParsedASTRuntimeRequest,
    parse_and_encode_generated_sources,
    sample_stage_b_from_parsed_sources,
)
from newera.nlp_flow.components.ast.flow.runtime.result_cfm import GeneratedASTResultCFM, ResultCFMConfig

SOURCES = (
    "def add(left, right):\n    return left + right\n",
    "value = 3\n",
)


def _qualification(mode: str = "production", **changes) -> ModelQualificationBinding:
    _parser_version, parser_hash = parser_runtime_identity("python")
    _canonicalizer_version, canonicalizer_hash = canonicalizer_runtime_identity()
    values = {
        "checkpoint_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "parser_sha256": parser_hash,
        "canonicalizer_sha256": canonicalizer_hash,
        "evaluator_sha256": "5" * 64,
        "languages": ("python",),
        "tracks": ("code",),
        "freeze_manifest_sha256": "6" * 64,
        "mode": mode,
    }
    return ModelQualificationBinding(**(values | changes))


def _encoder() -> ParsedASTEncoder:
    graphs = tuple(parse_ast_graph(source, "python") for source in SOURCES)
    node_types = set()
    edge_fields = set()
    scope_types = set()

    def visit(node):
        node_types.add(node.node_type)
        for field, child in node.children:
            edge_fields.add(field)
            visit(child)

    for graph in graphs:
        visit(graph.root)
        scope_types.update(scope.node_type for scope in graph.scopes)
    return ParsedASTEncoder(
        {value: index + 1 for index, value in enumerate(sorted(node_types))},
        {value: index + 1 for index, value in enumerate(sorted(edge_fields))},
        {value: index + 1 for index, value in enumerate(sorted(scope_types))},
        hidden_size=16,
        heads=4,
        layers=1,
        max_nodes=256,
    )


def _request(qualification=None) -> ParsedASTRuntimeRequest:
    parser_version, _parser_hash = parser_runtime_identity("python")
    canonicalizer_version, _canonicalizer_hash = canonicalizer_runtime_identity()
    return ParsedASTRuntimeRequest(
        task_ids=("case-1", "case-2"),
        input_sha256s=(hashlib.sha256(b"prompt-1").hexdigest(), hashlib.sha256(b"prompt-2").hexdigest()),
        track="code",
        qualification=qualification or _qualification(),
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=parser_version.rsplit(":grammar-", maxsplit=1)[-1],
    )


def test_generated_source_is_parsed_once_per_case_and_exact_graph_conditions_stage_b(monkeypatch) -> None:
    import newera.nlp_flow.components.ast.representation.parsed.parsed_ast_handoff as handoff_module

    real_parser = handoff_module.parse_ast_graph
    calls = []

    def count_parse(source: str, language: str):
        calls.append((source, language))
        return real_parser(source, language)

    monkeypatch.setattr(handoff_module, "parse_ast_graph", count_parse)
    encoder = _encoder()
    model = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=4)).eval()
    prompt = torch.randn(2, 16)
    noise = torch.randn(2, 5, 16)

    output = sample_stage_b_from_parsed_sources(
        model,
        prompt,
        SOURCES,
        language="python",
        request=_request(),
        encoder=encoder,
        result_length=5,
        steps=1,
        noise=noise,
    )

    assert calls == [(source, "python") for source in SOURCES]
    assert all(int(count) > 0 for count in output.parsed_ast.mask.sum(dim=1))
    assert len(output.parsed_ast.receipts) == 2
    assert output.result.endpoint.shape == noise.shape
    assert output.parsed_ast.states.shape[0] == prompt.shape[0]


def test_final_gateway_rejects_fixture_qualification_and_wrong_track() -> None:
    with pytest.raises(ValueError, match="production model qualification"):
        _request(_qualification(mode="fixture"))
    with pytest.raises(ValueError, match="does not cover this track"):
        parse_and_encode_generated_sources(
            SOURCES[:1],
            language="python",
            request=ParsedASTRuntimeRequest(
                task_ids=("case-1",),
                input_sha256s=("a" * 64,),
                track="engineering",
                qualification=_qualification(),
                parser_version="parser-v1",
                canonicalizer_version="canonical-v1",
                grammar_version="grammar-v1",
            ),
            encoder=_encoder(),
        )


def test_final_gateway_rejects_stale_parser_canonicalizer_and_grammar_identity() -> None:
    encoder = _encoder()
    request = _request()
    cases = (
        (
            replace(request, qualification=replace(request.qualification, parser_sha256="9" * 64)),
            "runtime parser differs",
        ),
        (
            replace(request, qualification=replace(request.qualification, canonicalizer_sha256="8" * 64)),
            "runtime canonicalizer differs",
        ),
        (replace(request, grammar_version="stale-grammar"), "runtime grammar differs"),
    )
    for stale_request, message in cases:
        with pytest.raises(ValueError, match=message):
            parse_and_encode_generated_sources(
                SOURCES,
                language="python",
                request=stale_request,
                encoder=encoder,
            )


def test_stage_b_gateway_api_has_no_raw_latent_or_predicted_topology_input() -> None:
    parameters = inspect.signature(sample_stage_b_from_parsed_sources).parameters
    assert {"sources", "request", "encoder"} <= set(parameters)
    assert not any("endpoint" in key or "latent" in key or "parent_indices" in key for key in parameters)
