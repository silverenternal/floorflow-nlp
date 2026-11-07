"""Fail-closed protocol definitions for the natural-language transfer route."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import (
    HIERARCHY_ROWS,
    TYPED_CHANNEL_NAMES,
)


@dataclass(frozen=True)
class HierarchyLevel:
    """One typed state level in the natural-language hierarchy."""

    level_id: str
    name: str
    role: str
    max_state_tokens: int


@dataclass(frozen=True)
class LanguageFlowProtocol:
    """Immutable summary of inputs, hierarchy, and target boundaries."""

    inference_inputs: frozenset[str]
    forbidden_inputs: frozenset[str]
    hierarchy: tuple[HierarchyLevel, ...]
    typed_channels: frozenset[str]
    stage_sequence: tuple[str, ...]

    @property
    def target_free(self) -> bool:
        return self.inference_inputs.isdisjoint(self.forbidden_inputs)


HIERARCHY_LEVELS = tuple(HierarchyLevel(*row) for row in HIERARCHY_ROWS)
EXPECTED_TYPED_CHANNELS = frozenset(TYPED_CHANNEL_NAMES)

EXPECTED_STAGES = (
    "NLP-PHASE-CONTRACT",
    "NLP-PHASE-INFRA",
    "NLP-PHASE-DATA",
    "NLP-PHASE-BENCHMARK",
    "NLP-PHASE-MODEL",
    "NLP-PHASE-SCREEN",
    "NLP-PHASE-SPECIALIZE",
    "NLP-PHASE-EVALUATE",
    "NLP-PHASE-CERTIFY",
)


def _string_set(payload: Any, field: str, errors: list[str]) -> frozenset[str]:
    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        errors.append(f"{field} must be a list of strings")
        return frozenset()
    return frozenset(payload)


def hierarchy_from_payload(payload: Any, errors: list[str]) -> tuple[HierarchyLevel, ...]:
    if not isinstance(payload, list):
        errors.append("hierarchy must be a list")
        return ()
    levels: list[HierarchyLevel] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            errors.append(f"hierarchy[{index}] must be an object")
            continue
        try:
            level = HierarchyLevel(
                level_id=str(row["id"]),
                name=str(row["name"]),
                role=str(row["role"]),
                max_state_tokens=int(row["max_state_tokens"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"hierarchy[{index}] is invalid: {exc}")
            continue
        if level.max_state_tokens <= 0:
            errors.append(f"hierarchy[{index}].max_state_tokens must be positive")
            continue
        levels.append(level)
    return tuple(levels)


def validate_protocol_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all contract violations; an empty tuple means fail-closed pass."""

    errors: list[str] = []
    contract = payload.get("inference_contract")
    if not isinstance(contract, Mapping):
        errors.append("inference_contract must be an object")
        contract = {}
    inference_inputs = _string_set(contract.get("required_inputs"), "required_inputs", errors)
    forbidden_inputs = _string_set(contract.get("forbidden_inputs"), "forbidden_inputs", errors)
    if contract.get("target_free") is not True:
        errors.append("inference_contract.target_free must be true")
    if overlap := sorted(inference_inputs & forbidden_inputs):
        errors.append(f"inference inputs overlap forbidden inputs: {overlap}")

    budget = payload.get("budget_contract")
    if not isinstance(budget, Mapping):
        errors.append("budget_contract must be an object")
    else:
        claim_cap = budget.get("primary_claim_parameter_cap")
        absolute_cap = budget.get("absolute_experiment_parameter_cap")
        if not isinstance(claim_cap, int) or claim_cap > 23_000_000:
            errors.append("primary claim parameter cap must not exceed 23M")
        if not isinstance(absolute_cap, int) or absolute_cap > 46_000_000:
            errors.append("absolute experiment cap must not exceed 46M")
        if budget.get("net_parameter_growth_allowed") is not False:
            errors.append("net parameter growth must be disabled")

    hierarchy = hierarchy_from_payload(payload.get("hierarchy"), errors)
    expected_ids = tuple(level.level_id for level in HIERARCHY_LEVELS)
    if tuple(level.level_id for level in hierarchy) != expected_ids:
        errors.append(f"hierarchy levels must be exactly {expected_ids}")

    typed_channels = _string_set(payload.get("typed_channels"), "typed_channels", errors)
    if typed_channels != EXPECTED_TYPED_CHANNELS:
        errors.append("typed_channels must cover syntax, symbol, discourse, execution, and engineering")

    stages = payload.get("stage_sequence")
    if not isinstance(stages, list) or tuple(stages) != EXPECTED_STAGES:
        errors.append(f"stage_sequence must be exactly {EXPECTED_STAGES}")

    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        errors.append("arms must be an object")
        arms = {}
    control = arms.get("control")
    candidate = arms.get("candidate")
    if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
        errors.append("arms.control and arms.candidate must be objects")
        control = control if isinstance(control, Mapping) else {}
        candidate = candidate if isinstance(candidate, Mapping) else {}
    shared_fields = _string_set(arms.get("shared_fields"), "arms.shared_fields", errors)
    for field in sorted(shared_fields):
        if control.get(field) != candidate.get(field):
            errors.append(f"matched arm field differs: {field}")
    if control.get("architecture") == candidate.get("architecture"):
        errors.append("control and candidate architectures must be distinct")

    metrics = _string_set(payload.get("required_task_metrics"), "required_task_metrics", errors)
    expected_metrics = {
        "held_out_token_nll",
        "perplexity",
        "constraint_success",
        "grammar_validity",
        "segment_boundary_f1",
        "unique_ngram_ratio",
        "heldout_coverage",
        "decoder_nfe",
        "tokens_per_second",
    }
    if not expected_metrics.issubset(metrics):
        errors.append(f"required_task_metrics is missing {sorted(expected_metrics - metrics)}")

    gates = payload.get("promotion_gates")
    if not isinstance(gates, Mapping) or len(gates) < 7:
        errors.append("promotion_gates must define the full leakage-to-replication gate")
    return tuple(errors)


def protocol_from_payload(payload: Mapping[str, Any]) -> LanguageFlowProtocol:
    """Materialize a protocol after validation."""

    errors = validate_protocol_payload(payload)
    if errors:
        raise ValueError("; ".join(errors))
    contract = payload["inference_contract"]
    return LanguageFlowProtocol(
        inference_inputs=frozenset(contract["required_inputs"]),
        forbidden_inputs=frozenset(contract["forbidden_inputs"]),
        hierarchy=hierarchy_from_payload(payload["hierarchy"], []),
        typed_channels=frozenset(payload["typed_channels"]),
        stage_sequence=tuple(payload["stage_sequence"]),
    )


__all__ = [
    "EXPECTED_STAGES",
    "HIERARCHY_LEVELS",
    "HierarchyLevel",
    "LanguageFlowProtocol",
    "hierarchy_from_payload",
    "protocol_from_payload",
    "validate_protocol_payload",
]
