from __future__ import annotations

import hashlib
import inspect

import torch

from newera.nlp_flow.components.ast.diagnostics.strict.strict_ast_gateway import ASTGatewaySuccess, commit_typed_ast
from newera.nlp_flow.components.ast.flow.runtime.result_cfm import GeneratedASTResultCFM, ResultCFMConfig
from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
    ModelQualificationBinding,
    canonicalizer_runtime_identity,
    parser_runtime_identity,
)
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_encoder import ParsedASTEncoder
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_handoff import (
    ParsedASTRuntimeRequest,
    sample_stage_b_from_parsed_receipts,
)
from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf
from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import TypedLexicalValue

SOURCE = "def add(left, right):\n    return left + right\n"


def _typed_values(graph):
    values = []
    index = 0

    def visit(node: ASTNode):
        nonlocal index
        node_id = index
        index += 1
        for _field, child in node.children:
            visit(child)
        if not node.children:
            values.append(
                TypedLexicalValue(
                    node_id=node_id,
                    node_kind=node.node_type,
                    leaf_class=classify_leaf(node.node_type, node.lexical_value or ""),
                    scope_id=node.scope_id,
                    value=node.lexical_value or "",
                    origin="generated",
                    minimum_length=0,
                    maximum_length=max(1, len(node.lexical_value or "")),
                )
            )

    visit(graph.root)
    return tuple(values)


def test_prompt_ir_source_receipt_and_stage_b_use_the_same_parsed_graph(monkeypatch) -> None:
    import newera.nlp_flow.components.ast.representation.parsed.parsed_ast_handoff as handoff_module
    import newera.nlp_flow.components.generation.trees.typed.typed_source_renderer as renderer_module

    parser_version, parser_hash = parser_runtime_identity("python")
    canonicalizer_version, canonicalizer_hash = canonicalizer_runtime_identity()
    qualification = ModelQualificationBinding(
        checkpoint_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        parser_sha256=parser_hash,
        canonicalizer_sha256=canonicalizer_hash,
        evaluator_sha256="5" * 64,
        languages=("python",),
        tracks=("code",),
        freeze_manifest_sha256="6" * 64,
        mode="production",
    )
    input_hash = hashlib.sha256(b"fixture user prompt").hexdigest()
    committed_graph = parse_ast_graph(SOURCE, "python")
    calls = []
    actual_parse = renderer_module.parse_ast_graph

    def count_parse(source: str, language: str):
        calls.append((source, language))
        return actual_parse(source, language)

    monkeypatch.setattr(renderer_module, "parse_ast_graph", count_parse)
    monkeypatch.setattr(
        handoff_module,
        "parse_ast_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("receipt handoff reparsed source")),
    )
    committed = commit_typed_ast(
        committed_graph,
        _typed_values(committed_graph),
        (),
        task_id="integration-case",
        track="code",
        input_sha256=input_hash,
        qualification=qualification,
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=parser_version.rsplit(":grammar-", maxsplit=1)[-1],
    )
    assert isinstance(committed, ASTGatewaySuccess)
    assert committed.source == SOURCE
    assert calls == [(SOURCE, "python")]

    node_kinds = set()
    edge_fields = set()
    scope_kinds = set()

    def collect(node):
        node_kinds.add(node.node_type)
        for field, child in node.children:
            edge_fields.add(field)
            collect(child)

    collect(committed.parsed_graph.root)
    scope_kinds.update(scope.node_type for scope in committed.parsed_graph.scopes)
    encoder = ParsedASTEncoder(
        {value: index + 1 for index, value in enumerate(sorted(node_kinds))},
        {value: index + 1 for index, value in enumerate(sorted(edge_fields))},
        {value: index + 1 for index, value in enumerate(sorted(scope_kinds))},
        hidden_size=16,
        heads=4,
        layers=1,
        max_nodes=256,
    )
    request = ParsedASTRuntimeRequest(
        task_ids=("integration-case",),
        input_sha256s=(input_hash,),
        track="code",
        qualification=qualification,
        parser_version=parser_version,
        canonicalizer_version=canonicalizer_version,
        grammar_version=parser_version.rsplit(":grammar-", maxsplit=1)[-1],
    )
    stage_b = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=4)).eval()
    result = sample_stage_b_from_parsed_receipts(
        stage_b,
        torch.randn(1, 16),
        (committed.parsed_graph,),
        (committed.receipt,),
        (committed.source,),
        language="python",
        request=request,
        encoder=encoder,
        result_length=5,
        steps=1,
        noise=torch.zeros(1, 5, 16),
    )
    assert result.parsed_ast.graphs[0] is committed.parsed_graph
    assert result.parsed_ast.receipts[0] == committed.receipt
    assert result.result.endpoint.shape == (1, 5, 16)
    assert calls == [(SOURCE, "python")]


def test_final_stage_b_entrypoint_has_no_raw_ast_prediction_or_gold_target_argument() -> None:
    parameters = inspect.signature(sample_stage_b_from_parsed_receipts).parameters
    assert {"graphs", "receipts", "sources", "request", "encoder"} <= set(parameters)
    assert not any("gold" in name or "target_ast" in name or "latent" in name for name in parameters)
