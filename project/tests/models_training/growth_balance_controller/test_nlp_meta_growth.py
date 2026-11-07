from __future__ import annotations

import inspect

import pytest
import torch

from newera.nlp_flow.components.research.architecture.growth.meta_growth import (
    BirthUtilityNet,
    MetaGrowthConfig,
    MetaGrowthLoop,
    birth_utility_target,
)


def test_birth_utility_learns_positive_and_negative_outcomes() -> None:
    torch.manual_seed(19)
    generator = torch.Generator().manual_seed(23)
    features = torch.rand(512, 7, generator=generator)
    gain = torch.rand(512, generator=generator) - 0.15
    random_gain = torch.rand(512, generator=generator)
    fixed_moe_gain = torch.rand(512, generator=generator)
    parameter_cost = torch.rand(512, generator=generator)
    compute_cost = torch.rand(512, generator=generator)
    dead = (torch.rand(512, generator=generator) > 0.8).float()
    target = birth_utility_target(
        post_adaptation_gain=gain,
        random_expert_gain=random_gain,
        fixed_moe_gain=fixed_moe_gain,
        parameter_cost=parameter_cost,
        compute_cost=compute_cost,
        dead_module=dead,
        config=MetaGrowthConfig(),
    )
    model = BirthUtilityNet(hidden_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    for _step in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = model.utility_loss(features, target_utility=target)
        loss.total.backward()
        optimizer.step()

    predicted, logits = model(features)
    positive = predicted[target > 0.0].mean()
    negative = predicted[target <= 0.0].mean()

    assert positive > negative
    assert torch.isfinite(logits).all()
    assert not any(
        "target" in name for name in inspect.signature(model.forward).parameters
    )


def test_episode_is_bounded_and_validation_cannot_synthesize() -> None:
    model = BirthUtilityNet()
    with torch.no_grad():
        model.utility_head.bias.fill_(2.0)
        model.birth_head.bias.fill_(2.0)
    loop = MetaGrowthLoop(model, config=MetaGrowthConfig(max_adaptation_steps=4))

    result = loop.run_episode(
        torch.ones(1, 7),
        phase="meta_train",
        observed_gain=0.3,
        requested_adaptation_steps=99,
    )
    validation = loop.run_episode(
        torch.ones(1, 7),
        phase="meta_validation",
        observed_gain=1.0,
        requested_adaptation_steps=8,
    )

    assert result.committed
    assert result.adaptation_steps == 4
    assert result.events.count("adapt") == 4
    assert "synthesize" in result.events
    assert "commit" in result.events
    assert not validation.committed
    assert "synthesize" not in validation.events

    loop.freeze_factory()
    with pytest.raises(RuntimeError, match="frozen"):
        loop.run_episode(
            torch.ones(1, 7),
            phase="meta_train",
            observed_gain=1.0,
            requested_adaptation_steps=1,
        )
