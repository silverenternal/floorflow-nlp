from __future__ import annotations

import torch

from newera.nlp_flow.model.components.training.objectives.channel_objectives import (
    ChannelObjectiveSpec,
    TypedChannelObjectiveBank,
    channel_objective_loss,
)

SPECS = {
    "syntax": ChannelObjectiveSpec("multiclass", 8),
    "symbol": ChannelObjectiveSpec("binary", 1),
    "discourse": ChannelObjectiveSpec("multiclass", 8),
    "execution": ChannelObjectiveSpec("multiclass", 4),
    "engineering": ChannelObjectiveSpec("multiclass", 8),
}


def test_channel_objectives_have_independent_shapes_and_gradients() -> None:
    bank = TypedChannelObjectiveBank(8, SPECS)
    states = {
        name: torch.randn(2, 6, 8, requires_grad=True)
        for name in SPECS
    }
    logits = bank(states)
    targets = {
        "syntax": torch.randint(0, 8, (2, 6)),
        "symbol": torch.randint(0, 2, (2, 6)),
        "discourse": torch.randint(0, 8, (2, 6)),
        "execution": torch.randint(0, 4, (2, 6)),
        "engineering": torch.randint(0, 8, (2, 6)),
    }
    loss, components = channel_objective_loss(
        logits,
        targets,
        None,
        SPECS,
        {name: 1.0 for name in SPECS},
    )
    loss.backward()

    assert logits["symbol"].shape == (2, 6, 1)
    assert set(components) == {f"{name}_loss" for name in SPECS} | {"loss"}
    for name in SPECS:
        assert states[name].grad is not None
        assert bank.heads[name].net[1].weight.grad is not None


def test_channel_loss_respects_masks() -> None:
    bank = TypedChannelObjectiveBank(8, SPECS)
    states = {name: torch.randn(2, 4, 8) for name in SPECS}
    logits = bank(states)
    targets = {
        "syntax": torch.zeros(2, 4, dtype=torch.long),
        "symbol": torch.zeros(2, 4, dtype=torch.long),
        "discourse": torch.zeros(2, 4, dtype=torch.long),
        "execution": torch.zeros(2, 4, dtype=torch.long),
        "engineering": torch.zeros(2, 4, dtype=torch.long),
    }
    masks = {name: torch.zeros(2, 4, dtype=torch.bool) for name in SPECS}
    masks["syntax"][:, 0] = True

    loss, _ = channel_objective_loss(
        logits,
        targets,
        masks,
        SPECS,
        {name: 1.0 for name in SPECS},
    )

    assert torch.isfinite(loss)
