from __future__ import annotations

import torch

from floorflow.nlp_flow.ast_cfm import (
    ASTCodecVocabulary,
    ASTStageAFlow,
    collate_ast_batches,
    encode_ast_graph,
)
from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.ast_token_decoder import (
    ASTTokenEndpointDecoder,
    decode_source_tokens,
    source_token_loss,
)


def _batch():
    graph = parse_ast(
        "def add(a, b):\n    return a + b\n",
        "python",
    )
    vocabulary = ASTCodecVocabulary.from_graphs((graph,))
    return collate_ast_batches(
        [encode_ast_graph(graph, vocabulary, max_nodes=32, template_id=0)]
    ), vocabulary


def test_token_endpoint_decoder_trains_and_samples_from_ast_states() -> None:
    torch.manual_seed(53)
    batch, vocabulary = _batch()
    decoder = ASTTokenEndpointDecoder(
        vocabulary,
        hidden_size=16,
        max_tokens=65,
        layers=1,
    )
    node_states = torch.randn(1, 32, 16, requires_grad=True)

    logits = decoder(
        node_states,
        batch.node_mask,
        batch.source_token_ids[:, :-1],
        batch.source_token_node_ids[:, :-1],
        batch.source_token_phase_ids[:, :-1],
    )
    loss = source_token_loss(
        logits,
        batch.source_token_ids[:, 1:],
        batch.source_token_mask[:, 1:],
    )
    loss.backward()
    sampled = decoder.sample(
        node_states.detach(),
        batch.node_mask,
        source_token_node_ids=batch.source_token_node_ids,
        source_token_phase_ids=batch.source_token_phase_ids,
    )
    sampled_source = decode_source_tokens(sampled[0], vocabulary)

    assert logits.shape[0] == 1
    assert sampled.shape == (1, 65)
    assert sampled_source
    assert node_states.grad is not None
    assert decoder.output.weight.grad is not None


def test_source_token_round_trip_matches_target_spelling() -> None:
    batch, vocabulary = _batch()

    source = decode_source_tokens(batch.source_token_ids[0], vocabulary)

    assert source == "def add(a, b):\n    return a + b\n"


def test_stage_a_sample_emits_token_endpoint_ids() -> None:
    _batch_data, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=16,
        num_heads=4,
        max_nodes=32,
        layers=1,
    ).eval()

    output = model.sample_target_free(
        torch.randn(1, 16),
        steps=1,
        noise=torch.randn(1, 32, 16),
    )

    assert output.source_token_ids is not None
    assert output.source_token_ids.shape == (1, 65)


def test_lexical_copy_bias_uses_the_bound_leaf_node() -> None:
    batch, vocabulary = _batch()
    decoder = ASTTokenEndpointDecoder(
        vocabulary,
        hidden_size=16,
        max_tokens=65,
        layers=1,
    )
    node_states = torch.zeros(1, 32, 16)
    lexical_a = torch.zeros(1, 32, vocabulary.num_lexemes)
    lexical_b = lexical_a.clone()
    lexical_position = batch.source_token_phase_ids[0].eq(2).nonzero()[0]
    leaf_node = int(batch.source_token_node_ids[0, lexical_position])
    lexical_a[:, leaf_node, vocabulary.lexemes.index("add")] = 10.0
    lexical_b[:, leaf_node, vocabulary.lexemes.index("add")] = -10.0
    lexical_b[:, leaf_node, vocabulary.lexemes.index("def")] = 10.0

    first = decoder(
        node_states,
        batch.node_mask,
        batch.source_token_ids[:, :-1],
        batch.source_token_node_ids[:, :-1],
        batch.source_token_phase_ids[:, :-1],
        lexical_a,
        torch.zeros_like(lexical_a[..., : vocabulary.num_gaps]),
    )
    second = decoder(
        node_states,
        batch.node_mask,
        batch.source_token_ids[:, :-1],
        batch.source_token_node_ids[:, :-1],
        batch.source_token_phase_ids[:, :-1],
        lexical_b,
        torch.zeros_like(lexical_b[..., : vocabulary.num_gaps]),
    )

    assert not torch.equal(first, second)
