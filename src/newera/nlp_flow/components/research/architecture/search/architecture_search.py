"""Validation-only architecture search with efficiency Pareto selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from newera.nlp_flow.components.research.architecture.search.architecture_search_types import (
    ArchitectureCandidate,
    ArchitectureEvaluator,
    CandidateEvaluation,
    SearchSpace,
)
from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def _candidate_id(config: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"arch-{digest}"


def estimate_inference_flops(
    config: NLPModelConfig,
    *,
    batch_size: int,
    sequence_length: int,
    expected_depth_fraction: float,
) -> int:
    """Estimate forward FLOPs and expose every search-relevant term."""

    if sequence_length > config.max_seq_len:
        raise ValueError("sequence_length exceeds the candidate context")
    if config.enable_dynamic_stack:
        fractional_depth = (
            config.dynamic_min_steps
            + (config.dynamic_max_steps - config.dynamic_min_steps)
            * expected_depth_fraction
        )
        depth = max(1, math.ceil(fractional_depth))
    else:
        depth = config.num_layers
    hidden = config.hidden_size
    tokens = batch_size * sequence_length
    linear_per_token = (
        4 * hidden * hidden
        + 4 * hidden * config.kv_size
        + 6 * hidden * config.intermediate_size
    )
    attention = 4 * batch_size * sequence_length * sequence_length * hidden
    core = depth * (tokens * linear_per_token + attention)
    lane = 0
    if config.enable_dynamic_stack:
        selected = config.dynamic_parallel_top_k
        lane = depth * (
            tokens * selected * 4 * hidden * config.dynamic_lane_rank
            + tokens * 2 * hidden * config.dynamic_parallel_lanes
            + tokens * 2 * hidden
        )
    competence = 0
    if config.enable_unified_competence:
        planner = tokens * (
            2 * hidden * config.unified_planner_dim
            + 6 * config.unified_planner_dim * config.unified_planner_dim
            + 2 * config.unified_planner_dim * len(config.gap_names)
        )
        cells = (
            tokens
            * config.unified_top_k
            * 4
            * hidden
            * config.unified_cell_rank
        )
        generated = (
            tokens * 4 * hidden * config.unified_generated_rank
            if config.unified_generated_expert_mode == "online"
            else 0
        )
        competence = planner + cells + generated
    output = 2 * tokens * hidden * config.vocab_size
    return int(core + lane + competence + output)


class ArchitectureCandidateFactory:
    """Deterministically emits valid candidates under declared static budgets."""

    def __init__(self, space: SearchSpace) -> None:
        self.space = space

    def generate(self) -> tuple[ArchitectureCandidate, ...]:
        configs = self._candidate_configs()
        candidates = []
        seen = set()
        for config, tags in configs:
            candidate_id = _candidate_id(config)
            if candidate_id in seen:
                continue
            try:
                model_config = NLPModelConfig(**config)
                model = CompactNLPDecoder(model_config)
            except (TypeError, ValueError):
                continue
            parameters = count_unique_trainable_parameters(model)
            training_flops = 6 * parameters
            inference_flops = estimate_inference_flops(
                model_config,
                batch_size=self.space.batch_size,
                sequence_length=self.space.sequence_length,
                expected_depth_fraction=self.space.expected_depth_fraction,
            )
            if parameters > self.space.max_parameters:
                continue
            if training_flops > self.space.max_training_flops_per_token:
                continue
            if inference_flops > self.space.max_inference_flops:
                continue
            seen.add(candidate_id)
            candidates.append(
                ArchitectureCandidate(
                    candidate_id=candidate_id,
                    config=config,
                    parameters=parameters,
                    estimated_training_flops_per_token=training_flops,
                    estimated_inference_flops=inference_flops,
                    tags=tags,
                )
            )
            if len(candidates) >= self.space.candidate_limit:
                break
        return tuple(candidates)

    def _candidate_configs(
        self,
    ) -> list[tuple[dict[str, object], tuple[str, ...]]]:
        base = dict(self.space.base_config)
        configs = [(base, ("base",))]
        for control in self.space.controls:
            control_id = str(control["id"])
            overrides = {
                name: value
                for name, value in control.items()
                if name != "id"
            }
            configs.append(
                (
                    {**base, **overrides},
                    (f"control:{control_id}",),
                )
            )
        names = tuple(self.space.axes)
        if names:
            low = {**base, **{name: self.space.axes[name][0] for name in names}}
            high = {**base, **{name: self.space.axes[name][-1] for name in names}}
            configs.extend(((low, ("joint:low",)), (high, ("joint:high",))))
        width = max(
            (len(values) for values in self.space.axes.values()),
            default=0,
        )
        for value_index in range(width):
            for axis, values in self.space.axes.items():
                if value_index >= len(values):
                    continue
                config = {**base, axis: values[value_index]}
                configs.append((config, (f"one_factor:{axis}",)))

        combinations = list(
            itertools.product(*(self.space.axes[name] for name in names))
        )
        random.Random(self.space.seed).shuffle(combinations)
        for values in combinations:
            config = {**base, **dict(zip(names, values))}
            configs.append((config, ("joint",)))
        return configs


def pareto_front(
    candidates: Sequence[ArchitectureCandidate],
    evaluations: Mapping[str, CandidateEvaluation],
) -> tuple[ArchitectureCandidate, ...]:
    """Return quality-max and parameter/compute/latency-min non-dominated points."""

    points = [candidate for candidate in candidates if candidate.candidate_id in evaluations]
    front = []
    for candidate in points:
        evaluation = evaluations[candidate.candidate_id]
        latency = evaluation.measured_latency_ms
        dominated = False
        for other in points:
            if other.candidate_id == candidate.candidate_id:
                continue
            other_eval = evaluations[other.candidate_id]
            other_latency = other_eval.measured_latency_ms
            quality_ge = other_eval.quality_score >= evaluation.quality_score
            parameter_le = other.parameters <= candidate.parameters
            training_le = (
                other.estimated_training_flops_per_token
                <= candidate.estimated_training_flops_per_token
            )
            inference_le = (
                other.estimated_inference_flops
                <= candidate.estimated_inference_flops
            )
            latency_le = (
                other_latency is None
                or latency is None
                or other_latency <= latency
            )
            strictly_better = any(
                (
                    other_eval.quality_score > evaluation.quality_score,
                    other.parameters < candidate.parameters,
                    other.estimated_training_flops_per_token
                    < candidate.estimated_training_flops_per_token,
                    other.estimated_inference_flops
                    < candidate.estimated_inference_flops,
                    latency is not None
                    and other_latency is not None
                    and other_latency < latency,
                )
            )
            if (
                quality_ge
                and parameter_le
                and training_le
                and inference_le
                and latency_le
                and strictly_better
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(front)


class EfficientArchitectureSearch:
    """Successive-halving search with a final smallest-on-plateau selection."""

    def __init__(
        self,
        *,
        candidates: Sequence[ArchitectureCandidate],
        token_budgets: Sequence[int],
        keep_fraction: float = 0.5,
        quality_plateau: float = 0.0,
        required_metrics: Sequence[str] = (),
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate is required")
        if not token_budgets or any(
            not isinstance(value, int) or value <= 0
            for value in token_budgets
        ):
            raise ValueError("token_budgets must contain positive integers")
        if tuple(sorted(token_budgets)) != tuple(token_budgets):
            raise ValueError("token_budgets must be strictly increasing")
        if len(set(token_budgets)) != len(token_budgets):
            raise ValueError("token_budgets must be unique")
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be in (0, 1]")
        if not math.isfinite(quality_plateau) or quality_plateau < 0.0:
            raise ValueError("quality_plateau must be finite and non-negative")
        if len({candidate.candidate_id for candidate in candidates}) != len(
            candidates
        ):
            raise ValueError("candidate ids must be unique")
        if any(not isinstance(name, str) or not name for name in required_metrics):
            raise ValueError("required metrics must be non-empty strings")
        self.candidates = tuple(candidates)
        self.token_budgets = tuple(token_budgets)
        self.keep_fraction = float(keep_fraction)
        self.quality_plateau = float(quality_plateau)
        self.required_metrics = tuple(dict.fromkeys(required_metrics))

    def run(self, evaluator: ArchitectureEvaluator) -> dict[str, object]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        survivors = list(self.candidates)
        observations: dict[str, CandidateEvaluation] = {}
        rounds = []
        for budget in self.token_budgets:
            measured = []
            for candidate in survivors:
                evaluation = evaluator.evaluate(candidate, budget)
                if evaluation.candidate_id != candidate.candidate_id:
                    raise ValueError("evaluator returned a mismatched candidate id")
                if evaluation.budget_tokens != budget:
                    raise ValueError("evaluator changed the matched token budget")
                observations[candidate.candidate_id] = evaluation
                measured.append(candidate)
            measured.sort(
                key=lambda candidate: self._rank_key(
                    candidate,
                    observations[candidate.candidate_id],
                )
            )
            keep_count = max(1, math.ceil(len(measured) * self.keep_fraction))
            next_survivors = measured[:keep_count]
            discarded = [
                candidate.candidate_id
                for candidate in measured
                if candidate not in next_survivors
            ]
            discard = getattr(evaluator, "discard", None)
            if callable(discard):
                for candidate_id in discarded:
                    discard(candidate_id)
            rounds.append(
                {
                    "budget_tokens": budget,
                    "evaluated": [candidate.candidate_id for candidate in measured],
                    "survivors": [
                        candidate.candidate_id for candidate in next_survivors
                    ],
                    "discarded": discarded,
                    "results": [
                        observations[candidate.candidate_id].as_dict()
                        for candidate in measured
                    ],
                }
            )
            survivors = next_survivors

        front = pareto_front(survivors, observations)
        selected = self._select_on_plateau(front, observations)
        if selected is None:
            raise RuntimeError("architecture search produced no valid finalist")
        selected_metrics = observations[selected.candidate_id].metrics
        missing_required_metrics = [
            metric
            for metric in self.required_metrics
            if metric not in selected_metrics
        ]
        return {
            "version": "nlp-efficient-architecture-search-v1",
            "status": "probe_complete",
            "selection_split": "validation_or_heldout_validation_only",
            "test_read": False,
            "token_budgets": list(self.token_budgets),
            "candidate_count": len(self.candidates),
            "rounds": rounds,
            "pareto_front": [candidate.candidate_id for candidate in front],
            "selected_candidate_id": selected.candidate_id,
            "selected_candidate": selected.as_dict(),
            "selected_evaluation": observations[selected.candidate_id].as_dict(),
            "required_metrics": list(self.required_metrics),
            "promotion_eligible": not missing_required_metrics,
            "promotion_blockers": missing_required_metrics,
            "observations": {
                candidate_id: evaluation.as_dict()
                for candidate_id, evaluation in observations.items()
            },
            "candidate_catalog": [
                by_id[candidate_id].as_dict()
                for candidate_id in sorted(observations)
            ],
            "evidence_boundary": (
                "Validation-only architecture screen. Promotion requires a "
                "frozen external-aligned verifier and matched confirmation; "
                "this report alone is not a capability or SOTA claim."
            ),
        }

    def _rank_key(
        self,
        candidate: ArchitectureCandidate,
        evaluation: CandidateEvaluation,
    ) -> tuple[float, int, int, float]:
        return (
            -evaluation.quality_score,
            candidate.parameters,
            candidate.estimated_inference_flops,
            (
                evaluation.measured_latency_ms
                if evaluation.measured_latency_ms is not None
                else math.inf
            ),
        )

    def _select_on_plateau(
        self,
        front: Sequence[ArchitectureCandidate],
        observations: Mapping[str, CandidateEvaluation],
    ) -> ArchitectureCandidate | None:
        if not front:
            return None
        best_quality = max(
            observations[candidate.candidate_id].quality_score
            for candidate in front
        )
        eligible = [
            candidate
            for candidate in front
            if best_quality
            - observations[candidate.candidate_id].quality_score
            <= self.quality_plateau
        ]
        return min(
            eligible,
            key=lambda candidate: self._rank_key(
                candidate,
                observations[candidate.candidate_id],
            )[1:],
        )


def freeze_selected_architecture(
    result: Mapping[str, object],
    *,
    config_path: Path,
    report_path: Path,
) -> dict[str, object]:
    """Write a reproducible candidate config and the full search ledger."""

    candidate = result.get("selected_candidate")
    if not isinstance(candidate, Mapping):
        raise TypeError("search result has no selected candidate")
    config = candidate.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("selected candidate has no model config")
    frozen = {
        "version": "nlp-selected-architecture-v1",
        "status": "search_selected_validation_only",
        "candidate_id": candidate["candidate_id"],
        "parameters": candidate["parameters"],
        "estimated_training_flops_per_token": candidate[
            "estimated_training_flops_per_token"
        ],
        "estimated_inference_flops": candidate[
            "estimated_inference_flops"
        ],
        "model": dict(config),
        "selection_report": str(report_path),
        "evidence_boundary": result["evidence_boundary"],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(frozen, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(dict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    return frozen


__all__ = [
    "ArchitectureCandidate",
    "ArchitectureCandidateFactory",
    "ArchitectureEvaluator",
    "CandidateEvaluation",
    "EfficientArchitectureSearch",
    "SearchSpace",
    "estimate_inference_flops",
    "freeze_selected_architecture",
    "pareto_front",
]
