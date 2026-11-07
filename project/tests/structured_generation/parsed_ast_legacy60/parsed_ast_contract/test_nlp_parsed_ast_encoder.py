from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import ModelQualificationBinding, issue_parsed_ast_receipt
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_encoder import ParsedASTEncoder, StageBConditioning


def _qualification() -> ModelQualificationBinding:
    return ModelQualificationBinding(
        checkpoint_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        parser_sha256="3" * 64,
        canonicalizer_sha256="4" * 64,
        evaluator_sha256="5" * 64,
        languages=("python",),
        tracks=("code",),
        freeze_manifest_sha256="6" * 64,
        mode="fixture",
    )


def _vocab(graphs: tuple[ASTGraph, ...]):
    node_types: set[str] = set()
    edges: set[str] = set()
    scope_types: set[str] = set()

    def walk(node: ASTNode) -> None:
        node_types.add(node.node_type)
        for field, child in node.children:
            edges.add(field)
            walk(child)

    for graph in graphs:
        walk(graph.root)
        scope_types.update(scope.node_type for scope in graph.scopes)
    return (
        {name: index + 1 for index, name in enumerate(sorted(node_types))},
        {name: index + 1 for index, name in enumerate(sorted(edges))},
        {name: index + 1 for index, name in enumerate(sorted(scope_types))},
    )


def _encode(source: str = "def add(left, right):\n    return left + right\n"):
    graph = parse_ast_graph(source, "python")
    receipt = issue_parsed_ast_receipt(
        graph,
        source,
        task_id="encoder-fixture",
        track="code",
        input_sha256="a" * 64,
        qualification=_qualification(),
        parser_version="parser-v1",
        canonicalizer_version="canonical-v1",
        grammar_version="grammar-v1",
        mode="fixture",
    )
    encoder = ParsedASTEncoder(*_vocab((graph,)), hidden_size=16, heads=4, layers=1, max_nodes=256).eval()
    return graph, receipt, encoder, encoder(graph, receipt)


def test_parsed_graph_topology_and_lexical_values_condition_stage_b_states() -> None:
    _graph, baseline_receipt, encoder, baseline = _encode()
    lex_graph, lex_receipt, _unused, _lex_output = _encode("def add(lhs, right):\n    return lhs + right\n")
    topology_graph, topology_receipt, _unused, _topology_output = _encode(
        "def add(left, right):\n    if left:\n        return right\n    return 0\n"
    )
    lexical_change = encoder(lex_graph, lex_receipt)
    topology_change = encoder(topology_graph, topology_receipt)

    assert baseline.states.shape[0] == 1
    assert baseline.states.shape[-1] == 16
    assert not torch.allclose(baseline.states, lexical_change.states[:, : baseline.states.shape[1]])
    assert baseline.states.shape[1] != topology_change.states.shape[1]
    assert lex_receipt.full_ordered_canonical_graph_sha256 != baseline_receipt.full_ordered_canonical_graph_sha256
    assert topology_receipt.full_ordered_canonical_graph_sha256 != baseline_receipt.full_ordered_canonical_graph_sha256
    assert topology_graph.language == "python"
    assert encoder.node_embedding.weight.grad is None


def test_encoder_rejects_graph_mutations_after_receipt_creation() -> None:
    graph, receipt, encoder, _encoded = _encode()
    changed = replace(graph, root=replace(graph.root, lexical_value="stale"))

    with pytest.raises(ValueError, match="does not match"):
        encoder(changed, receipt)


def test_stage_b_channels_are_distinct_and_have_no_latent_endpoint_api() -> None:
    graph, receipt, _encoder, encoded = _encode()
    prompt_states = torch.randn(1, 3, 16)
    test_states = torch.randn(1, 2, 16)
    channels = StageBConditioning(
        prompt_states=prompt_states,
        prompt_mask=torch.ones(1, 3, dtype=torch.bool),
        structure_states=encoded.states,
        structure_mask=encoded.mask,
        test_states=test_states,
        test_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert channels.prompt_states is prompt_states
    assert channels.test_states is test_states
    assert channels.structure_states is encoded.states
    assert "latent" not in inspect.signature(ParsedASTEncoder.forward).parameters
    assert "endpoint" not in StageBConditioning.__dataclass_fields__
    assert receipt.language == graph.language
