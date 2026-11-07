"""Hard-cap checks and tunable-value provenance for dynamic NLP computation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class KnobSource(StrEnum):
    FIXED = "fixed"
    LEARNED = "learned"
    VALIDATION_SELECTED = "validation_selected"


REQUIRED_KNOBS = frozenset(
    {
        "optimizer.learning_rate",
        "optimizer.epsilon",
        "loss.ast_action",
        "budget.max_nodes",
        "budget.max_source_bytes",
        "budget.max_depth",
        "budget.max_lanes",
        "budget.max_nfe",
        "budget.max_generated_experts",
        "budget.stopping_source",
    }
)


@dataclass(frozen=True)
class BudgetCaps:
    max_nodes: int
    max_source_bytes: int
    max_depth: int
    max_lanes: int
    max_nfe: int
    max_generated_experts: int

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("all runtime budget caps must be positive")


@dataclass(frozen=True)
class BudgetRequest:
    nodes: int
    source_bytes: int
    depth: int
    lanes: int
    nfe: int
    generated_experts: int
    stopping_source: KnobSource


def check_budget(request: BudgetRequest, caps: BudgetCaps) -> dict[str, Any]:
    fields = (
        ("nodes", request.nodes, caps.max_nodes),
        ("source_bytes", request.source_bytes, caps.max_source_bytes),
        ("depth", request.depth, caps.max_depth),
        ("lanes", request.lanes, caps.max_lanes),
        ("nfe", request.nfe, caps.max_nfe),
        ("generated_experts", request.generated_experts, caps.max_generated_experts),
    )
    violations = [
        {"field": name, "requested": value, "cap": cap}
        for name, value, cap in fields
        if value < 0 or value > cap
    ]
    return {
        "feasible": not violations,
        "stopping_source": request.stopping_source.value,
        "requested": {name: value for name, value, _cap in fields},
        "caps": {name: cap for name, _value, cap in fields},
        "violations": violations,
    }


def validate_knob_ledger(payload: dict[str, Any]) -> None:
    if payload.get("zero_tuning_certificate") is not False:
        raise ValueError("knob provenance cannot certify zero tuning")
    if payload.get("per_task_answer_overrides") is not False:
        raise ValueError("per-task answer-derived overrides are forbidden")
    knobs = payload.get("knobs")
    if not isinstance(knobs, dict) or not knobs:
        raise ValueError("knob ledger must declare its values")
    if not REQUIRED_KNOBS.issubset(knobs):
        raise ValueError(f"knob ledger omits required values: {sorted(REQUIRED_KNOBS - set(knobs))}")
    for name, entry in knobs.items():
        if not isinstance(entry, dict) or "value" not in entry or not entry.get("evidence"):
            raise ValueError(f"knob {name} needs a value and evidence")
        try:
            source = KnobSource(entry["source"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"knob {name} has an unrecognized provenance") from error
        if (
            source == KnobSource.VALIDATION_SELECTED
            and (not entry.get("selection_report_sha256") or entry.get("validation_split") != "validation")
        ):
            raise ValueError(f"knob {name} lacks its validation selection receipt")
        if source == KnobSource.VALIDATION_SELECTED:
            digest = entry["selection_report_sha256"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"knob {name} selection receipt must be SHA-256")


__all__ = [
    "REQUIRED_KNOBS",
    "BudgetCaps",
    "BudgetRequest",
    "KnobSource",
    "check_budget",
    "validate_knob_ledger",
]
