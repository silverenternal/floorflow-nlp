"""Coding/engineering target and claim-ladder contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterTier:
    tier_id: str
    parameters: int
    corpus_tokens: int

    @property
    def ratio(self) -> float:
        return self.corpus_tokens / self.parameters


def _positive_int(row: Mapping[str, Any], field: str, errors: list[str]) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{field} must be a positive integer")
        return 0
    return value


def parameter_ladder(payload: Any, errors: list[str]) -> tuple[ParameterTier, ...]:
    if not isinstance(payload, list) or not payload:
        errors.append("parameter_ladder must be a non-empty list")
        return ()
    tiers: list[ParameterTier] = []
    ids: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            errors.append(f"parameter_ladder[{index}] must be an object")
            continue
        tier_id = str(row.get("id", ""))
        if not tier_id or tier_id in ids:
            errors.append(f"parameter_ladder[{index}] has a missing or duplicate id")
        ids.add(tier_id)
        parameters = _positive_int(row, "parameters", errors)
        tokens = _positive_int(row, "corpus_tokens", errors)
        if parameters and tokens != parameters * 20:
            errors.append(f"{tier_id} violates corpus_tokens = 20 * parameters")
        tiers.append(ParameterTier(tier_id, parameters, tokens))
    if [tier.parameters for tier in tiers] != sorted(tier.parameters for tier in tiers):
        errors.append("parameter_ladder must be ordered by increasing parameters")
    return tuple(tiers)


def _tracks(payload: Any, errors: list[str]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, list) or not payload:
        errors.append("evaluation_tracks must be a non-empty list")
        return ()
    tracks = tuple(row for row in payload if isinstance(row, Mapping))
    if len(tracks) != len(payload):
        errors.append("every evaluation track must be an object")
    ids = {str(track.get("id", "")) for track in tracks}
    required = {
        "python_code_generation",
        "multilingual_code",
        "ast_structure",
        "repository_patch",
        "engineering_acceleration",
        "language_retention",
    }
    if ids != required:
        errors.append(f"evaluation tracks must be exactly {sorted(required)}")
    weights = [track.get("weight") for track in tracks]
    if not all(isinstance(weight, (int, float)) for weight in weights):
        errors.append("every evaluation track needs a numeric weight")
    elif abs(sum(weights) - 1.0) > 1e-9:
        errors.append("evaluation track weights must sum to 1")
    return tracks


def validate_target_plan(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all coding-target contract violations."""

    errors: list[str] = []
    parameter_ladder(payload.get("parameter_ladder"), errors)
    tracks = _tracks(payload.get("evaluation_tracks"), errors)
    multilingual = next(
        (track for track in tracks if track.get("id") == "multilingual_code"),
        {},
    )
    benchmark_names = " ".join(str(value) for value in multilingual.get("benchmarks", []))
    for language in ("Python", "Rust", "C++"):
        if language not in benchmark_names:
            errors.append(f"multilingual_code is missing {language}")
    ast_track = next((track for track in tracks if track.get("id") == "ast_structure"), {})
    ast_names = " ".join(str(value) for value in ast_track.get("benchmarks", []))
    for required in ("AST node", "AST round-trip", "Tree edit"):
        if required not in ast_names:
            errors.append(f"ast_structure is missing {required}")

    engineering = next(
        (track for track in tracks if track.get("id") == "engineering_acceleration"),
        {},
    )
    engineering_names = " ".join(str(value) for value in engineering.get("benchmarks", []))
    for required in ("Triton", "Linux"):
        if required not in engineering_names:
            errors.append(f"engineering_acceleration is missing {required}")

    policy = payload.get("benchmark_policy")
    if not isinstance(policy, Mapping):
        errors.append("benchmark_policy must be an object")
    else:
        if policy.get("availability_audit_before_freeze") is not True:
            errors.append("benchmark availability audit must precede freezing")
        budget = policy.get("generation_budget")
        if not isinstance(budget, Mapping):
            errors.append("generation_budget must be an object")
        elif budget.get("best_of_n_allowed_in_primary") is not False:
            errors.append("best-of-N cannot enter the primary endpoint")

    guards = payload.get("hard_guards")
    if not isinstance(guards, Mapping):
        errors.append("hard_guards must be an object")
    else:
        if guards.get("strict_json_parse_validity") != 1.0:
            errors.append("strict JSON parse validity must equal 1.0")
        if guards.get("schema_compliance") != 1.0:
            errors.append("JSON schema compliance must equal 1.0")
        if guards.get("posthoc_repair_count") != 0 or guards.get("retry_count") != 0:
            errors.append("JSON repair and retry counts must be zero")
        if guards.get("native_valid_by_construction") is not True:
            errors.append("JSON validity must be native to model structure")
        if "blocks all C1-C4" not in str(guards.get("failure_action", "")):
            errors.append("JSON failure must block all C1-C4 claims")

    ladder = payload.get("promotion_ladder")
    if not isinstance(ladder, list) or [row.get("id") for row in ladder] != ["C0", "C1", "C2", "C3", "C4"]:
        errors.append("promotion_ladder must contain C0 through C4 in order")
    if "pending" not in str(payload.get("sota_band", {}).get("status", "")):
        errors.append("SOTA-band thresholds must remain pending until the baseline audit")
    anti_scaling = payload.get("anti_scaling_contract")
    if not isinstance(anti_scaling, Mapping):
        errors.append("anti_scaling_contract must be an object")
    else:
        claim_cap = anti_scaling.get("primary_claim_parameter_cap")
        absolute_cap = anti_scaling.get("absolute_experiment_parameter_cap")
        if not isinstance(claim_cap, int) or claim_cap > 23_000_000:
            errors.append("primary claim cap must not exceed 23M parameters")
        if not isinstance(absolute_cap, int) or absolute_cap > 46_000_000:
            errors.append("absolute experiment cap must not exceed 46M parameters")
        if anti_scaling.get("net_parameter_growth_allowed") is not False:
            errors.append("net parameter growth must be disabled for the primary route")
        if "reference only" not in str(anti_scaling.get("large_model_comparison_policy", "")).lower():
            errors.append("large-model comparisons must be reference only")
        efficiency = anti_scaling.get("primary_efficiency_metrics")
        if not isinstance(efficiency, list) or len(efficiency) < 4:
            errors.append("per-parameter, per-FLOP, per-token, and wall-clock efficiency are required")
        forbidden = " ".join(anti_scaling.get("forbidden_strategies", [])).lower()
        for phrase in ("generic transformer layers", "ensembling", "test-time expert creation"):
            if phrase not in forbidden:
                errors.append(f"anti-scaling contract is missing {phrase}")
    return tuple(errors)


def target_tiers(payload: Mapping[str, Any]) -> tuple[ParameterTier, ...]:
    errors = validate_target_plan(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return parameter_ladder(payload["parameter_ladder"], [])


__all__ = [
    "ParameterTier",
    "parameter_ladder",
    "target_tiers",
    "validate_target_plan",
]
