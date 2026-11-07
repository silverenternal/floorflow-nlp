from __future__ import annotations

import torch

from newera.nlp_flow.model import TypedChannelMemoryEncoder


def test_typed_channel_memory_shapes_and_priors() -> None:
    encoder = TypedChannelMemoryEncoder(hidden_size=32, channel_size=8)
    hidden = torch.randn(2, 7, 32)

    memory = encoder(hidden)

    assert memory.global_memory.shape == (2, 5, 32)
    assert set(memory.token_states) == {
        "syntax",
        "symbol",
        "discourse",
        "execution",
        "engineering",
    }
    assert all(state.shape == (2, 7, 8) for state in memory.token_states.values())
    assert memory.relation_prior.shape == (5, 5)
    assert torch.allclose(memory.relation_prior, memory.relation_prior.T)


def test_channel_masks_change_only_their_channel_memory() -> None:
    encoder = TypedChannelMemoryEncoder(hidden_size=32, channel_size=8).eval()
    hidden = torch.randn(1, 6, 32)
    masks = {
        name: torch.ones(1, 6, dtype=torch.bool)
        for name in (
            "syntax",
            "symbol",
            "discourse",
            "execution",
            "engineering",
        )
    }
    masks["syntax"][:, 3:] = False

    with torch.no_grad():
        full = encoder(hidden)
        masked = encoder(hidden, masks)

    for index, name in enumerate(
        ("syntax", "symbol", "discourse", "execution", "engineering")
    ):
        if name == "syntax":
            assert not torch.allclose(
                full.global_memory[:, index],
                masked.global_memory[:, index],
            )
        else:
            assert torch.allclose(
                full.global_memory[:, index],
                masked.global_memory[:, index],
            )


def test_channel_encoders_receive_independent_gradients() -> None:
    encoder = TypedChannelMemoryEncoder(hidden_size=32, channel_size=8)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    memory = encoder(hidden)

    loss = memory.global_memory.square().mean()
    loss.backward()

    assert hidden.grad is not None
    gradients = [
        encoder.down[name].weight.grad
        for name in memory.token_states
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert len({float(gradient.norm()) for gradient in gradients}) > 1


def test_channel_encoder_stays_compact() -> None:
    encoder = TypedChannelMemoryEncoder(hidden_size=160, channel_size=32)
    parameters = sum(parameter.numel() for parameter in encoder.parameters())

    assert parameters < 100_000
