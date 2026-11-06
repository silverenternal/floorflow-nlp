"""Types and validated protocol records for efficient architecture search."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from floorflow.nlp_flow._generated_protocol import ARCHITECTURE_AXIS_NAMES
from floorflow.nlp_flow.model import NLPModelConfig

SELECTION_SPLITS = frozenset({"validation", "heldout_validation"})


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class ArchitectureCandidate:
    candidate_id: str
    config: Mapping[str, object]
    parameters: int
    estimated_training_flops_per_token: int
    estimated_inference_flops: int
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        _positive_int(self.parameters, "parameters")
        _positive_int(
            self.estimated_training_flops_per_token,
            "estimated_training_flops_per_token",
        )
        _positive_int(
            self.estimated_inference_flops,
            "estimated_inference_flops",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "config": dict(self.config),
            "parameters": self.parameters,
            "estimated_training_flops_per_token": (
                self.estimated_training_flops_per_token
            ),
            "estimated_inference_flops": self.estimated_inference_flops,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    budget_tokens: int
    training_tokens: int
    quality_score: float
    metrics: Mapping[str, float]
    evaluation_split: str
    measured_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        _positive_int(self.budget_tokens, "budget_tokens")
        _positive_int(self.training_tokens, "training_tokens")
        if not math.isfinite(self.quality_score):
            raise ValueError("quality_score must be finite")
        if self.evaluation_split not in SELECTION_SPLITS:
            raise ValueError(
                "architecture selection accepts only validation splits, "
                f"got {self.evaluation_split!r}"
            )
        if self.measured_latency_ms is not None:
            _finite_nonnegative(
                self.measured_latency_ms,
                "measured_latency_ms",
            )
        if any(
            not isinstance(name, str)
            or not name
            or not math.isfinite(float(value))
            for name, value in self.metrics.items()
        ):
            raise ValueError("evaluation metrics must be finite named values")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "budget_tokens": self.budget_tokens,
            "training_tokens": self.training_tokens,
            "quality_score": self.quality_score,
            "metrics": dict(self.metrics),
            "evaluation_split": self.evaluation_split,
            "measured_latency_ms": self.measured_latency_ms,
        }


class ArchitectureEvaluator(Protocol):
    def evaluate(
        self,
        candidate: ArchitectureCandidate,
        budget_tokens: int,
    ) -> CandidateEvaluation: ...


@dataclass(frozen=True)
class SearchSpace:
    base_config: Mapping[str, object]
    axes: Mapping[str, tuple[object, ...]]
    max_parameters: int
    max_training_flops_per_token: int
    max_inference_flops: int
    candidate_limit: int
    sequence_length: int
    batch_size: int
    expected_depth_fraction: float
    seed: int
    controls: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "max_parameters",
            "max_training_flops_per_token",
            "max_inference_flops",
            "candidate_limit",
            "sequence_length",
            "batch_size",
        ):
            _positive_int(getattr(self, name), name)
        _finite_nonnegative(self.expected_depth_fraction, "expected_depth_fraction")
        if self.expected_depth_fraction > 1.0:
            raise ValueError("expected_depth_fraction cannot exceed one")
        valid_fields = set(NLPModelConfig.__dataclass_fields__)
        unknown = set(self.axes) - valid_fields
        if unknown:
            raise ValueError(f"unknown architecture axes: {sorted(unknown)}")
        undeclared = set(self.axes) - set(ARCHITECTURE_AXIS_NAMES)
        if undeclared:
            raise ValueError(
                "architecture axes are missing from the Rust contract: "
                f"{sorted(undeclared)}"
            )
        if any(not values for values in self.axes.values()):
            raise ValueError("search axes cannot be empty")
        for control in self.controls:
            if not isinstance(control, Mapping):
                raise TypeError("search controls must be objects")
            if not str(control.get("id", "")).strip():
                raise ValueError("search control requires a non-empty id")
            unknown_control = set(control) - valid_fields - {"id"}
            if unknown_control:
                raise ValueError(
                    "search control has unknown fields: "
                    f"{sorted(unknown_control)}"
                )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        root: Path,
    ) -> SearchSpace:
        base_path = Path(str(payload["base_config"]))
        if not base_path.is_absolute():
            base_path = root / base_path
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        base_config = base_payload["model"]
        raw_axes = payload.get("search_space", payload.get("axes"))
        if not isinstance(raw_axes, Mapping):
            raise TypeError("search_space must be an object")
        axes = {}
        for name, values in raw_axes.items():
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError(f"search axis {name} must be a list")
            axes[str(name)] = tuple(values)
        measurement = payload["measurement"]
        if not isinstance(measurement, Mapping):
            raise TypeError("measurement must be an object")
        raw_controls = payload.get("controls", [])
        if not isinstance(raw_controls, list) or not all(
            isinstance(control, Mapping)
            for control in raw_controls
        ):
            raise TypeError("controls must be a list of objects")
        return cls(
            base_config=dict(base_config),
            axes=axes,
            max_parameters=int(measurement["max_parameters"]),
            max_training_flops_per_token=int(
                measurement["max_training_flops_per_token"]
            ),
            max_inference_flops=int(measurement["max_inference_flops"]),
            candidate_limit=int(measurement["candidate_limit"]),
            sequence_length=int(measurement["sequence_length"]),
            batch_size=int(measurement["batch_size"]),
            expected_depth_fraction=float(
                measurement["expected_depth_fraction"]
            ),
            seed=int(measurement.get("seed", 0)),
            controls=tuple(dict(control) for control in raw_controls),
        )


__all__ = [
    "SELECTION_SPLITS",
    "ArchitectureCandidate",
    "ArchitectureEvaluator",
    "CandidateEvaluation",
    "SearchSpace",
]
