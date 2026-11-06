from __future__ import annotations

import torch

from floorflow.nlp_flow.engineering import ScheduleFreeAdamW


def test_schedule_free_adamw_trains_without_an_external_scheduler() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    optimizer = ScheduleFreeAdamW(
        [parameter],
        lr=0.05,
    )
    initial = parameter.detach().square().sum()

    optimizer.train()
    for _step in range(20):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
    optimizer.eval()

    assert parameter.square().sum() < initial
    assert optimizer.param_groups[0]["k"] == 20


def test_schedule_free_adamw_switches_between_train_and_eval_iterates() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ScheduleFreeAdamW([parameter], lr=0.01)
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    train_value = float(parameter.detach())
    optimizer.eval()
    eval_value = float(parameter.detach())
    optimizer.train()

    assert eval_value != train_value
    assert float(parameter.detach()) == train_value
