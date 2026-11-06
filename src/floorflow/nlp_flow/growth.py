"""Contracts for neuron-as-submodel and bounded learned neurogenesis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EVIDENCE_FIELDS = (
    "id",
    "title",
    "authors",
    "venue",
    "publication_date",
    "doi",
    "source_type",
    "verified_claim",
    "engineering_implication",
    "scope_limit",
)
EXPECTED_ARMS = ("G0", "G1", "G2", "G3", "G4", "G5")


def _nonempty_string(row: Mapping[str, Any], field: str, errors: list[str], prefix: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}.{field} must be a non-empty string")
        return ""
    return value


def validate_growth_plan(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all evidence and growth-mechanism contract violations."""

    errors: list[str] = []
    _nonempty_string(payload, "operational_hypothesis", errors, "plan")
    _nonempty_string(payload, "provisional_claim", errors, "plan")
    rejected = payload.get("rejected_interpretations")
    if not isinstance(rejected, list) or len(rejected) < 3:
        errors.append("rejected_interpretations must contain at least three limits")

    evidence = payload.get("evidence_ledger")
    if not isinstance(evidence, list) or len(evidence) < 5:
        errors.append("evidence_ledger must contain at least five records")
        evidence = []
    peer_reviewed = 0
    for index, row in enumerate(evidence):
        if not isinstance(row, Mapping):
            errors.append(f"evidence_ledger[{index}] must be an object")
            continue
        prefix = f"evidence_ledger[{index}]"
        for field in EVIDENCE_FIELDS:
            if field in {"pmid"}:
                continue
            _nonempty_string(row, field, errors, prefix)
        doi = str(row.get("doi", ""))
        if not doi.startswith("10."):
            errors.append(f"{prefix}.doi is invalid")
        source_type = str(row.get("source_type", ""))
        if source_type.startswith("peer_reviewed"):
            peer_reviewed += 1
        if not str(row.get("scope_limit", "")).strip():
            errors.append(f"{prefix} must state the source scope limit")
    if peer_reviewed < 3:
        errors.append("evidence_ledger needs at least three peer-reviewed sources")

    module = payload.get("neuron_module")
    if not isinstance(module, Mapping):
        errors.append("neuron_module must be an object")
    else:
        if not isinstance(module.get("state"), list) or len(module["state"]) < 3:
            errors.append("neuron_module.state must declare local state")
        if module.get("initialization") != "zero-gated residual child attached to a declared parent":
            errors.append("new neuron modules must start as zero-gated residual children")

    factory = payload.get("expert_factory")
    if not isinstance(factory, Mapping):
        errors.append("expert_factory must be an object")
    else:
        if not isinstance(factory.get("factory_inputs"), list) or len(factory["factory_inputs"]) < 4:
            errors.append("expert_factory needs at least four context inputs")
        outputs = factory.get("factory_outputs")
        if not isinstance(outputs, list) or not {
            "expert blueprint",
            "low-rank weight factors",
            "zero-initialized birth gate",
        }.issubset(set(outputs)):
            errors.append("expert_factory must generate a blueprint, weight factors, and zero gate")
        if "hypernetwork" not in str(factory.get("minimal_generation_method", "")).lower():
            errors.append("expert_factory must synthesize weights rather than select a fixed pool")
        distinctions = " ".join(factory.get("counts_as_generated_expert_only_if", [])).lower()
        for phrase in ("current parent", "different contexts", "fixed pretrained pool", "parameter budget"):
            if phrase not in distinctions:
                errors.append(f"generated-expert definition is missing {phrase}")

    moe = payload.get("moe_boundary")
    if not isinstance(moe, Mapping):
        errors.append("moe_boundary must be an object")
    else:
        if moe.get("fixed_expert_pool") != "G5 baseline":
            errors.append("fixed expert pools must be the G5 control")
        if "non-constant" not in str(moe.get("disentanglement_requirement", "")).lower():
            errors.append("MoE disentanglement must require non-constant generated experts")

    ladder = payload.get("expert_synthesis_ladder")
    if not isinstance(ladder, list) or [row.get("id") for row in ladder if isinstance(row, Mapping)] != [
        "S0",
        "S1",
        "S2",
        "S3",
    ]:
        errors.append("expert_synthesis_ladder must contain S0 through S3 in order")
    else:
        by_level = {row["id"]: row for row in ladder}
        if by_level["S0"].get("generated") is not False:
            errors.append("S0 fixed-pool routing cannot count as generated")
        if by_level["S2"].get("core_self_generation_claim") is not True:
            errors.append("S2 must be the minimum core self-generation claim")

    meta = payload.get("meta_growth_learning")
    if not isinstance(meta, Mapping):
        errors.append("meta_growth_learning must be an object")
    else:
        episode = meta.get("episode")
        if not isinstance(episode, list) or len(episode) < 6:
            errors.append("meta-growth episodes must include attempt, diagnosis, synthesis, adaptation, and cost")
        steps = meta.get("adaptation_steps_per_episode")
        if not isinstance(steps, int) or not 1 <= steps <= 32:
            errors.append("adaptation_steps_per_episode must be in [1, 32]")
        if meta.get("final_training_phase") != "Freeze the factory and the final expert set before primary evaluation.":
            errors.append("the expert factory and final expert set must freeze before primary evaluation")

    controller = payload.get("growth_controller")
    if not isinstance(controller, Mapping):
        errors.append("growth_controller must be an object")
    else:
        rounds = controller.get("max_growth_rounds")
        if not isinstance(rounds, int) or not 1 <= rounds <= 3:
            errors.append("max_growth_rounds must be in [1, 3]")
        if controller.get("freeze_after_growth") is not True:
            errors.append("the final network must freeze after growth")
        actions = set(controller.get("actions", []))
        if actions != {"birth", "route", "merge", "retire"}:
            errors.append("growth actions must cover birth, route, merge, and retire")

    arms = payload.get("ablation_arms")
    if not isinstance(arms, list):
        errors.append("ablation_arms must be a list")
    else:
        arm_ids = tuple(row.get("id") for row in arms if isinstance(row, Mapping))
        if arm_ids != EXPECTED_ARMS:
            errors.append(f"ablation arms must be exactly {EXPECTED_ARMS}")

    budget = payload.get("budget")
    if not isinstance(budget, Mapping):
        errors.append("budget must be an object")
    else:
        if budget.get("mode") != "fixed_tier_reallocation":
            errors.append("growth must use fixed-tier reallocation")
        if budget.get("net_parameter_growth_allowed") is not False:
            errors.append("net parameter growth must be disabled")
        claim_cap = budget.get("primary_claim_parameter_cap")
        absolute_cap = budget.get("absolute_experiment_parameter_cap")
        if not isinstance(claim_cap, int) or claim_cap > 23_000_000:
            errors.append("primary growth claim cap must not exceed 23M")
        if not isinstance(absolute_cap, int) or absolute_cap > 46_000_000:
            errors.append("absolute growth experiment cap must not exceed 46M")
        if budget.get("generated_parameters_count_toward_total") is not True:
            errors.append("generated parameters must count toward the model budget")
        if budget.get("corpus_token_parameter_ratio") != 20:
            errors.append("growth must preserve the 20:1 corpus/parameter ratio")
        if budget.get("inference_after_freeze") != "static final network":
            errors.append("primary inference must use a static final network")
        if "forbidden" not in str(budget.get("absolute_growth_mode", "")).lower():
            errors.append("absolute net growth must be forbidden for the primary claim")
        if not isinstance(budget.get("capacity_reallocation"), list) or len(budget["capacity_reallocation"]) < 3:
            errors.append("growth must define capacity reallocation actions")

    gates = payload.get("promotion_gates")
    if not isinstance(gates, list) or len(gates) < 7:
        errors.append("promotion_gates must contain at least seven checks")
    else:
        gate_text = " ".join(str(gate) for gate in gates).lower()
        for phrase in (
            "random births",
            "static width/depth",
            "fixed-expert moe",
            "net trainable parameters do not increase",
            "20:1",
            "no dynamic births",
        ):
            if phrase not in gate_text:
                errors.append(f"promotion_gates is missing {phrase}")
    return tuple(errors)


__all__ = ["EXPECTED_ARMS", "validate_growth_plan"]
