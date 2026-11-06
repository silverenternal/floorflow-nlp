from __future__ import annotations

from floorflow.nlp_flow import ast_cfm, ast_cfm_data, ast_codec
from floorflow.nlp_flow.ast_codec_io import serialize_ast as direct_serialize_ast
from floorflow.nlp_flow.ast_codec_metrics import ast_metrics as direct_ast_metrics


def test_ast_cfm_data_public_symbols_remain_on_ast_cfm() -> None:
    assert ast_cfm.ASTTensorBatch is ast_cfm_data.ASTTensorBatch
    assert ast_cfm.ASTFlowOutput is ast_cfm_data.ASTFlowOutput
    assert ast_cfm.ASTObjectiveLosses is ast_cfm_data.ASTObjectiveLosses
    assert ast_cfm.ASTCodecVocabulary is ast_cfm_data.ASTCodecVocabulary
    assert ast_cfm.encode_ast_graph is ast_cfm_data.encode_ast_graph
    assert ast_cfm.collate_ast_batches is ast_cfm_data.collate_ast_batches
    assert ast_cfm.prompt_context is ast_cfm_data.prompt_context


def test_ast_codec_public_wrappers_match_direct_implementations() -> None:
    source = "def add(a, b):\n    return a + b\n"
    graph = ast_codec.parse_ast(source, "python")
    assert ast_codec.serialize_ast(graph) == direct_serialize_ast(graph)
    assert ast_codec.ast_metrics(graph, graph) == direct_ast_metrics(graph, graph)
    assert ast_codec.deserialize_ast(ast_codec.serialize_ast(graph)) == graph
