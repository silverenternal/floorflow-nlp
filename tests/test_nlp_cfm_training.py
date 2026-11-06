from __future__ import annotations

import torch

from floorflow.nlp_flow.cfm_training import (
    CFMTrainingConfig,
    ModelEMA,
    flow_velocity_target,
    interpolate_flow,
    sample_flow_times,
    warmup_cosine_scale,
)


def test_logit_normal_timesteps_are_bounded_deterministic_and_not_uniform() -> None:
    config = CFMTrainingConfig(timestep_sampler="logit_normal")
    first = sample_flow_times(
        4096,
        config,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    second = sample_flow_times(
        4096,
        config,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )
    uniform = sample_flow_times(
        4096,
        CFMTrainingConfig(timestep_sampler="uniform"),
        device="cpu",
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(first, second)
    assert first.min() >= config.timestep_epsilon
    assert first.max() <= 1.0 - config.timestep_epsilon
    assert ((first >= 0.25) & (first <= 0.75)).float().mean() > (
        ((uniform >= 0.25) & (uniform <= 0.75)).float().mean()
    )


def test_flow_interpolation_and_target_are_consistent() -> None:
    noise = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)
    time = torch.tensor([0.0, 1.0])

    state = interpolate_flow(noise, target, time)

    assert torch.equal(state[0], noise[0])
    assert torch.equal(state[1], target[1])
    assert torch.equal(flow_velocity_target(noise, target), target)


def test_model_ema_tracks_float_parameters_and_copies_buffers() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
    )
    ema = ModelEMA(model, decay=0.5)
    initial = ema.state_dict()["0.weight"].clone()
    with torch.no_grad():
        model[0].weight.add_(2.0)
        model[1].running_mean.add_(3.0)

    ema.update(model)

    assert torch.allclose(
        ema.state_dict()["0.weight"],
        initial + 1.0,
    )
    assert torch.equal(
        ema.state_dict()["1.running_mean"],
        model[1].running_mean,
    )


def test_warmup_cosine_schedule_is_bounded() -> None:
    values = [
        warmup_cosine_scale(step, 100, warmup_steps=10)
        for step in (0, 5, 10, 50, 99)
    ]

    assert values[0] < values[1] < values[2]
    assert values[2] == 1.0
    assert values[-1] >= 0.05
    assert values[-1] < values[-2]
