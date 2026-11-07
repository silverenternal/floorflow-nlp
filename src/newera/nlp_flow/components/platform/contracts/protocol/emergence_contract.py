"""Validity-gated novelty credit and one-mechanism experiment contracts."""

from __future__ import annotations

import math
from typing import Any

MECHANISM_IDS = frozenset(
    {"disabled", "fixed_competence", "random_competence", "fixed_pool", "generated_e"}
)


def valid_emergence_reward(
    novelty: float,
    *,
    supported: bool,
    holdout: bool,
    validity: dict[str, bool],
    required_validity: tuple[str, ...] = (
        "ast_valid",
        "native_schema_valid",
        "application_valid",
        "execution_valid",
    ),
) -> float:
    if not math.isfinite(novelty) or not 0.0 <= novelty <= 1.0:
        raise ValueError("novelty must be finite and in [0, 1]")
    if not supported or not holdout:
        return 0.0
    if any(validity.get(name) is not True for name in required_validity):
        return 0.0
    return novelty


def validate_mechanism_queue(payload: dict[str, Any]) -> None:
    if payload.get("run_one_mechanism_at_a_time") is not True:
        raise ValueError("mechanism comparisons must isolate one mechanism per run")
    if payload.get("execution", {}).get("run_enabled") is not False:
        raise ValueError("implementation queue cannot launch mechanism experiments")
    controls = payload.get("controls", [])
    identifiers = [row.get("id") for row in controls]
    if set(identifiers) != MECHANISM_IDS or len(identifiers) != len(MECHANISM_IDS):
        raise ValueError("mechanism queue must include each fixed and generated control once")
    seeds = payload.get("seed_set", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("mechanism queue needs unique registered seeds")
    policy = payload.get("emergence_objective", {})
    if policy.get("invalid_output_novelty_credit") != 0.0:
        raise ValueError("invalid outputs cannot receive novelty credit")
    if policy.get("unsupported_output_novelty_credit") != 0.0:
        raise ValueError("unsupported outputs cannot receive novelty credit")


__all__ = ["MECHANISM_IDS", "valid_emergence_reward", "validate_mechanism_queue"]
