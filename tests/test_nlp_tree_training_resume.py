from __future__ import annotations

import copy
import random

import pytest
import torch
from torch import nn

from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW
from floorflow.nlp_flow.nlp_tree_checkpoint import (
    capture_training_checkpoint,
    restore_training_checkpoint,
)

IDENTITY = {
    "data_sha256": "data-fixture",
    "parser_sha256": "parser-fixture",
    "model_sha256": "model-fixture",
    "tokenizer_sha256": "tokenizer-fixture",
}


def _build() -> tuple[nn.Module, ScheduleFreeAdamW]:
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01, warmup_steps=2)
    model.train()
    optimizer.train()
    return model, optimizer


def _step(model, optimizer, sampler, trace):
    indices = torch.randperm(12, generator=sampler)[:4]
    values = torch.randn(4, 3)
    targets = torch.randn(4, 2)
    trace.append((random.random(), indices.clone(), values.clone(), targets.clone()))
    optimizer.zero_grad(set_to_none=True)
    loss = (model(values) - targets).square().mean()
    loss.backward()
    optimizer.step()


def _run_updates(model, optimizer, sampler, trace, count):
    for _ in range(count):
        _step(model, optimizer, sampler, trace)


def test_checkpoint_resume_reproduces_next_batch_rng_scope_and_updates() -> None:
    torch.manual_seed(41)
    random.seed(41)
    sampler = torch.Generator().manual_seed(42)
    uninterrupted_model, uninterrupted_optimizer = _build()
    uninterrupted_trace = []
    _run_updates(uninterrupted_model, uninterrupted_optimizer, sampler, uninterrupted_trace, 4)

    torch.manual_seed(41)
    random.seed(41)
    split_sampler = torch.Generator().manual_seed(42)
    model, optimizer = _build()
    partial_trace = []
    _run_updates(model, optimizer, split_sampler, partial_trace, 2)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        sampler=split_sampler,
        identity=IDENTITY,
        counters={"optimizer_steps": 2, "examples_seen": 8},
        active_scopes=("fixture_head",),
        effective_batch_size=4,
    )

    resumed_sampler = torch.Generator().manual_seed(0)
    resumed_model, resumed_optimizer = _build()
    restored = restore_training_checkpoint(
        copy.deepcopy(checkpoint),
        resumed_model,
        resumed_optimizer,
        sampler=resumed_sampler,
        expected_identity=IDENTITY,
        expected_effective_batch_size=4,
    )
    resumed_trace = []
    _run_updates(resumed_model, resumed_optimizer, resumed_sampler, resumed_trace, 2)

    assert restored["counters"] == {"optimizer_steps": 2, "examples_seen": 8}
    assert restored["active_scopes"] == ("fixture_head",)
    assert restored["effective_batch_size"] == 4
    for actual_step, expected_step in zip(partial_trace + resumed_trace, uninterrupted_trace, strict=True):
        assert actual_step[0] == expected_step[0]
        for actual, expected in zip(actual_step[1:], expected_step[1:], strict=True):
            torch.testing.assert_close(actual, expected)
    assert resumed_model.training == uninterrupted_model.training
    assert [group["train_mode"] for group in resumed_optimizer.param_groups] == [True]
    for actual, expected in zip(resumed_model.parameters(), uninterrupted_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert resumed_optimizer.state_dict()["param_groups"] == uninterrupted_optimizer.state_dict()["param_groups"]


def test_checkpoint_restore_rejects_changed_data_or_parser_identity() -> None:
    model, optimizer = _build()
    sampler = torch.Generator().manual_seed(3)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        sampler=sampler,
        identity=IDENTITY,
        counters={"optimizer_steps": 0},
        active_scopes=("all",),
        effective_batch_size=1,
    )
    with pytest.raises(ValueError, match="identity changed"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            sampler=sampler,
            expected_identity={**IDENTITY, "parser_sha256": "changed"},
            expected_effective_batch_size=1,
        )


def test_checkpoint_restore_rejects_changed_effective_batch_size() -> None:
    model, optimizer = _build()
    sampler = torch.Generator().manual_seed(3)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        sampler=sampler,
        identity=IDENTITY,
        counters={"optimizer_steps": 0},
        active_scopes=("all",),
        effective_batch_size=4,
    )
    with pytest.raises(ValueError, match="effective batch size changed"):
        restore_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            sampler=sampler,
            expected_identity=IDENTITY,
            expected_effective_batch_size=8,
        )
