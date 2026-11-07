from __future__ import annotations

import torch

from newera.nlp_flow.model.components.tasks.conversation.conversation import (
    ConversationControlHead,
    TurnSegmentEmbedding,
)


def test_turn_embedding_is_identity_at_initialization() -> None:
    embedding = TurnSegmentEmbedding(32, 5)
    ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])

    assert torch.equal(embedding(ids), torch.zeros(2, 5, 32))


def test_conversation_heads_expose_all_control_targets() -> None:
    head = ConversationControlHead(32)
    output = head(torch.randn(2, 7, 32))

    assert output["continue_logits"].shape == (2, 7)
    assert output["turn_boundary_logits"].shape == (2, 7, 3)
    assert output["reply_mode_logits"].shape == (2, 7, 4)
    assert output["language_logits"].shape == (2, 7, 4)
    assert output["validity_logits"].shape == (2, 7, 3)
    assert ConversationControlHead.should_stop(
        torch.tensor([[-1.0, 1.0], [2.0, -2.0]])
    ).tolist() == [False, True]


def test_conversation_control_losses_are_finite_and_differentiable() -> None:
    head = ConversationControlHead(32)
    hidden = torch.randn(2, 5, 32)
    predictions = head(hidden)
    targets = {
        "continue_logits": torch.ones(2, 5),
        "turn_boundary_logits": torch.zeros(2, 5, dtype=torch.long),
        "reply_mode_logits": torch.ones(2, 5, dtype=torch.long),
        "language_logits": torch.full((2, 5), 1, dtype=torch.long),
        "validity_logits": torch.ones(2, 5, 3),
    }
    losses = head.losses(
        predictions,
        targets,
        masks={
            "continue_logits": torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, True],
                ]
            ),
            "turn_boundary_logits": torch.ones(2, 5, dtype=torch.bool),
            "reply_mode_logits": torch.ones(2, 5, dtype=torch.bool),
            "language_logits": torch.ones(2, 5, dtype=torch.bool),
            "validity_logits": torch.ones(2, 5, 3, dtype=torch.bool),
        },
    )

    sum(losses.values()).backward()
    assert set(losses) == {
        "continue",
        "validity",
        "turn_boundary",
        "reply_mode",
        "language",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert head.continue_head.weight.grad is not None
