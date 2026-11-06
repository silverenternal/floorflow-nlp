from __future__ import annotations

import torch

from floorflow.nlp_flow.model.prompt_encoding import SemanticPromptEncoder


def test_semantic_prompt_encoder_reads_token_content_and_language() -> None:
    torch.manual_seed(31)
    encoder = SemanticPromptEncoder(
        vocab_size=64,
        hidden_size=16,
        max_tokens=12,
        layers=2,
    )
    first_ids = torch.tensor([[1, 2, 3, 4]])
    second_ids = torch.tensor([[1, 2, 5, 6]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    first = encoder(first_ids, mask, torch.tensor([1]))
    second = encoder(second_ids, mask, torch.tensor([1]))
    other_language = encoder(first_ids, mask, torch.tensor([2]))

    assert first.context.shape == (1, 16)
    assert first.token_states.shape == (1, 4, 16)
    assert not torch.equal(first.context, second.context)
    assert not torch.equal(first.context, other_language.context)


def test_prompt_encoding_gradients_reach_tokens_and_pooling() -> None:
    encoder = SemanticPromptEncoder(
        vocab_size=64,
        hidden_size=16,
        max_tokens=12,
        layers=1,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])

    state = encoder(input_ids)
    state.context.square().mean().backward()

    assert encoder.token_embedding.weight.grad is not None
    assert encoder.pool_query.grad is not None
