"""Architecture and staged-training contracts for the compact SOTA route."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPECTED_CHANNELS = ("syntax", "symbol", "discourse", "execution", "engineering")
EXPECTED_ATTENTION = {
    "semantic_channel_attention",
    "typed_cross_attention",
    "mita_sparse_routing",
    "slot_mixing",
}
EXPECTED_SCOPES = ("S0-base", "S1-channels", "S2-cross-attention", "S3-cfm", "S4-joint", "S5-factory")


def validate_architecture_plan(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every architecture, data-use, and staged-training violation."""

    errors: list[str] = []
    ambition = str(payload.get("ambition", "")).lower()
    if "sota-level" not in ambition or "without gpt-class parameter stacking" not in ambition:
        errors.append("ambition must preserve compact-class SOTA and reject GPT-class stacking")
    core = payload.get("shared_core")
    if not isinstance(core, Mapping):
        errors.append("shared_core must be an object")
    else:
        cap = core.get("parameter_cap")
        if not isinstance(cap, int) or cap > 23_000_000:
            errors.append("shared-core cap must not exceed 23M")
        if core.get("net_growth_allowed") is not False:
            errors.append("net growth must be disabled")
        if "token decoder" not in str(core.get("token_readout", "")).lower():
            errors.append("shared core must declare the token decoder interface")

    structured = payload.get("structured_output")
    if not isinstance(structured, Mapping):
        errors.append("structured_output must be an object")
    else:
        if structured.get("native_valid_by_construction") is not True:
            errors.append("structured output must be valid by construction")
        if structured.get("hard_validity_gate") is not True:
            errors.append("structured output needs a hard validity gate")
        if structured.get("posthoc_repair") is not False or structured.get("retry") is not False:
            errors.append("structured output cannot use post-hoc repair or retry")

    channels = payload.get("typed_channels")
    if not isinstance(channels, list):
        errors.append("typed_channels must be a list")
        channels = []
    channel_ids = tuple(row.get("id") for row in channels if isinstance(row, Mapping))
    if channel_ids != EXPECTED_CHANNELS:
        errors.append(f"typed channels must be exactly {EXPECTED_CHANNELS}")
    for index, channel in enumerate(channels):
        if not isinstance(channel, Mapping):
            errors.append(f"typed_channels[{index}] must be an object")
            continue
        for field in ("role", "training_targets", "cross_attention_relations"):
            if not channel.get(field):
                errors.append(f"channel {channel.get('id')} lacks {field}")

    attention = payload.get("attention")
    if not isinstance(attention, Mapping) or set(attention) != EXPECTED_ATTENTION:
        errors.append(f"attention must contain exactly {sorted(EXPECTED_ATTENTION)}")
    else:
        for name, mechanism in attention.items():
            if not isinstance(mechanism, Mapping) or not mechanism.get("source") or not mechanism.get("borrow"):
                errors.append(f"{name} must document its source and borrowed design")

    cfm = payload.get("cfm")
    if not isinstance(cfm, Mapping):
        errors.append("cfm must be an object")
    else:
        if "two-stage ast-first" not in str(cfm.get("role", "")).lower():
            errors.append("CFM role must be two-stage AST-first flow")
        stage_a = cfm.get("stage_a_ast")
        stage_b = cfm.get("stage_b_result")
        if not isinstance(stage_a, Mapping) or not isinstance(stage_b, Mapping):
            errors.append("CFM must define stage_a_ast and stage_b_result")
        else:
            hierarchy = stage_a.get("hierarchy")
            if not isinstance(hierarchy, list) or len(hierarchy) < 3:
                errors.append("Stage A must define AST root, subtree, and leaf hierarchy")
            objectives = " ".join(stage_a.get("objectives", [])).lower()
            for objective in ("node type", "edge type", "grammar validity", "parseability"):
                if objective not in objectives:
                    errors.append(f"Stage-A AST CFM is missing {objective}")
            if "stage-a generated ast" not in str(stage_b.get("condition", "")).lower():
                errors.append("Stage B must condition on generated AST")
            curriculum = " ".join(stage_b.get("training_curriculum", [])).lower()
            if "teacher ast" not in curriculum or "generated ast only" not in curriculum:
                errors.append("Stage B needs teacher-to-generated AST curriculum")
            b_objectives = " ".join(stage_b.get("objectives", [])).lower()
            for objective in ("reconstruction", "verifier", "consistency"):
                if objective not in b_objectives:
                    errors.append(f"Stage-B result CFM is missing {objective}")
        inference = str(cfm.get("inference", "")).lower()
        if "stage a generates ast first" not in inference or "target ast is never available" not in inference:
            errors.append("CFM inference must generate AST first and prohibit target AST")
        if "one-stage control" not in str(cfm.get("ablation_rule", "")).lower():
            errors.append("two-stage CFM needs a matched one-stage control")

    scopes = payload.get("training_scopes")
    if not isinstance(scopes, list):
        errors.append("training_scopes must be a list")
    else:
        scope_ids = tuple(scope.get("id") for scope in scopes if isinstance(scope, Mapping))
        if scope_ids != EXPECTED_SCOPES:
            errors.append(f"training scopes must be exactly {EXPECTED_SCOPES}")
        for scope in scopes:
            if not isinstance(scope, Mapping) or not scope.get("train") or not isinstance(scope.get("freeze"), list):
                errors.append("every training scope needs train and freeze lists")

    data = payload.get("data_usage")
    required_data = {"general_language", "code", "model_architecture", "engineering_acceleration"}
    if not isinstance(data, Mapping) or set(data) != required_data:
        errors.append(f"data_usage must cover {sorted(required_data)}")
    code_languages = " ".join(data.get("code", {}).get("languages", [])).lower() if isinstance(data, Mapping) else ""
    for language in ("python", "rust", "cpp"):
        if language not in code_languages:
            errors.append(f"code data usage is missing {language}")

    arms = payload.get("experimental_arms")
    arm_ids = tuple(row.get("id") for row in arms if isinstance(row, Mapping)) if isinstance(arms, list) else ()
    if arm_ids != ("A0", "A1", "A2", "A3", "A4", "A5"):
        errors.append("experimental arms must contain A0 through A5")
    gates = " ".join(payload.get("promotion_gates", [])).lower()
    for phrase in (
        "two-stage ast-first cfm",
        "stage-a generated ast",
        "100% strict json validity",
        "cross-attention-only",
        "23m",
        "static",
    ):
        if phrase not in gates:
            errors.append(f"promotion gates are missing {phrase}")
    return tuple(errors)


def validated_channels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors = validate_architecture_plan(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return tuple(row["id"] for row in payload["typed_channels"])


__all__ = [
    "EXPECTED_ATTENTION",
    "EXPECTED_CHANNELS",
    "EXPECTED_SCOPES",
    "validate_architecture_plan",
    "validated_channels",
]
