from __future__ import annotations

import inspect

import pytest
import torch

from newera.nlp_flow.components.research.architecture.growth.growth_controller import (
    GROWTH_ACTIONS,
    BoundedGrowthManager,
    LearnedGrowthController,
    growth_action_utility,
    random_growth_action,
)


def _features(count: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 7, generator=generator)


def test_controller_learns_multiple_actions_and_beats_random_utility() -> None:
    torch.manual_seed(17)
    features = _features(512)
    utilities = growth_action_utility(features)
    targets = utilities.argmax(dim=-1)
    model = LearnedGrowthController(hidden_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _step in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss = model.policy_loss(
            features,
            action_targets=targets,
            utility_targets=utilities,
        )
        loss.total.backward()
        optimizer.step()

    predicted = model(features).action_logits.argmax(dim=-1)
    learned_utility = utilities.gather(1, predicted[:, None]).squeeze(1).mean()
    random_actions = random_growth_action(101, features.shape[0])
    random_utility = utilities.gather(1, random_actions[:, None]).squeeze(1).mean()

    assert set(predicted.tolist()) == set(range(len(GROWTH_ACTIONS)))
    assert learned_utility > random_utility
    assert not any("target" in name for name in inspect.signature(model.forward).parameters)


def test_growth_manager_rejects_net_growth_and_freezes_at_budget() -> None:
    manager = BoundedGrowthManager(
        max_growth_rounds=2,
        parameter_budget=1_000,
        initial_parameters=900,
    )

    with pytest.raises(ValueError, match="net parameter growth"):
        manager.apply("birth", birth_parameters=50, retired_parameters=20)

    event = manager.apply("birth", birth_parameters=50, retired_parameters=50)
    assert event["net_parameter_delta"] == 0
    assert manager.active_parameters == 900
    manager.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        manager.apply("retire", retired_parameters=10)


def test_growth_rounds_are_bounded_independently_of_parameter_budget() -> None:
    manager = BoundedGrowthManager(max_growth_rounds=1, parameter_budget=1_000)
    manager.apply("birth", birth_parameters=10, retired_parameters=10)

    with pytest.raises(ValueError, match="maximum growth rounds"):
        manager.apply("birth", birth_parameters=10, retired_parameters=10)
