"""Cooperative wall-clock limits for matched NLP experiment runs."""

from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetDeadlineExceeded(RuntimeError):
    def __init__(self, *, scope: str, phase: str, elapsed_seconds: float) -> None:
        self.scope = scope
        self.phase = phase
        self.elapsed_seconds = elapsed_seconds
        self.partial_checkpoint: str | None = None
        self.training_metrics: dict[str, object] | None = None
        super().__init__(f"{scope} wall-time budget reached during {phase}")


@dataclass(frozen=True)
class ExperimentDeadline:
    started_at: float
    total_deadline: float
    seed_deadline: float

    def check(self, phase: str) -> None:
        now = time.monotonic()
        if now >= self.total_deadline:
            raise BudgetDeadlineExceeded(
                scope="total",
                phase=phase,
                elapsed_seconds=now - self.started_at,
            )
        if now >= self.seed_deadline:
            raise BudgetDeadlineExceeded(
                scope="per_seed",
                phase=phase,
                elapsed_seconds=now - self.started_at,
            )


__all__ = ["BudgetDeadlineExceeded", "ExperimentDeadline"]
