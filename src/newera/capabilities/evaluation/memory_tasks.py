"""Deterministic sequence-memory tasks with independent reference targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MemorySequenceSample:
    events: Tensor
    target: Tensor
    family: str


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _event(
    family: int,
    *,
    write: bool = False,
    query: bool = False,
    value: float = 0.0,
    symbol_a: bool = False,
    symbol_b: bool = False,
) -> Tensor:
    row = torch.zeros(8)
    row[family] = 1.0
    row[3] = float(write)
    row[4] = float(query)
    row[5] = value
    row[6] = float(symbol_a)
    row[7] = float(symbol_b)
    return row


def _validate_length(length: int) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length < 4:
        raise ValueError("sequence length must be an integer >= 4")


def make_delayed_recall(*, length: int, seed: int) -> MemorySequenceSample:
    """Recall a marked value after intervening, non-target events."""
    _validate_length(length)
    generator = _generator(seed)
    target = torch.randn((), generator=generator)
    events = [_event(0, write=True, value=float(target))]
    events.extend(
        _event(0, value=float(torch.randn((), generator=generator)))
        for _ in range(length - 2)
    )
    events.append(_event(0, query=True))
    return MemorySequenceSample(torch.stack(events), target, "delayed_recall")


def make_interference_recovery(*, length: int, seed: int) -> MemorySequenceSample:
    """Recall the latest marked write after distractors and an overwrite."""
    _validate_length(length)
    if length < 6:
        raise ValueError("interference recovery requires length >= 6")
    generator = _generator(seed)
    initial = torch.randn((), generator=generator)
    target = torch.randn((), generator=generator)
    overwrite_at = length // 2
    events = []
    for index in range(length - 1):
        if index == 0:
            events.append(_event(2, write=True, value=float(initial)))
        elif index == overwrite_at:
            events.append(_event(2, write=True, value=float(target)))
        else:
            events.append(_event(2, value=float(torch.randn((), generator=generator))))
    events.append(_event(2, query=True))
    return MemorySequenceSample(torch.stack(events), target, "interference_recovery")


def make_order_recognition(*, length: int, seed: int) -> MemorySequenceSample:
    """Report whether A or B appeared first after a variable delay."""
    _validate_length(length)
    generator = _generator(seed)
    positions = torch.randperm(length - 1, generator=generator)[:2]
    position_a, position_b = int(positions[0]), int(positions[1])
    events = []
    for index in range(length - 1):
        events.append(
            _event(
                1,
                value=float(torch.randn((), generator=generator)),
                symbol_a=index == position_a,
                symbol_b=index == position_b,
            )
        )
    events.append(_event(1, query=True))
    target = torch.tensor(float(position_a < position_b))
    return MemorySequenceSample(torch.stack(events), target, "order_recognition")


__all__ = [
    "MemorySequenceSample",
    "make_delayed_recall",
    "make_interference_recovery",
    "make_order_recognition",
]
