from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.research.architecture.growth.growth import validate_growth_plan

ROOT = Path(__file__).resolve().parents[4]


def load_growth_plan() -> dict:
    return json.loads((ROOT / "project/configs/models_training/learning/growth/nlp_neuron_growth.json").read_text())


def test_growth_plan_has_verified_scope_limited_evidence() -> None:
    payload = load_growth_plan()

    assert validate_growth_plan(payload) == ()
    assert sum(
        row["source_type"].startswith("peer_reviewed")
        for row in payload["evidence_ledger"]
    ) >= 3


def test_growth_plan_requires_all_matched_controls() -> None:
    payload = load_growth_plan()
    payload["ablation_arms"].pop()

    errors = validate_growth_plan(payload)

    assert any("ablation arms" in error for error in errors)


def test_growth_plan_rejects_unbounded_growth_rounds() -> None:
    payload = load_growth_plan()
    payload["growth_controller"]["max_growth_rounds"] = 10

    errors = validate_growth_plan(payload)

    assert "max_growth_rounds must be in [1, 3]" in errors


def test_growth_plan_rejects_hidden_parameter_cost() -> None:
    payload = load_growth_plan()
    payload["budget"]["generated_parameters_count_toward_total"] = False

    errors = validate_growth_plan(payload)

    assert "generated parameters must count toward the model budget" in errors


def test_growth_plan_rejects_parameter_stacking_under_growth_name() -> None:
    payload = load_growth_plan()
    payload["budget"]["net_parameter_growth_allowed"] = True
    payload["budget"]["primary_claim_parameter_cap"] = 100_000_000

    errors = validate_growth_plan(payload)

    assert "net parameter growth must be disabled" in errors
    assert "primary growth claim cap must not exceed 23M" in errors


def test_growth_plan_rejects_fixed_pool_masquerading_as_synthesis() -> None:
    payload = load_growth_plan()
    payload["moe_boundary"]["fixed_expert_pool"] = "G4 learned branch"

    errors = validate_growth_plan(payload)

    assert "fixed expert pools must be the G5 control" in errors


def test_growth_plan_requires_context_dependent_expert_weights() -> None:
    payload = load_growth_plan()
    payload["expert_factory"]["counts_as_generated_expert_only_if"] = [
        "the expert has a nonzero gate"
    ]

    errors = validate_growth_plan(payload)

    assert any("generated-expert definition is missing current parent" in error for error in errors)


def test_growth_plan_requires_s2_as_core_self_generation_gate() -> None:
    payload = load_growth_plan()
    payload["expert_synthesis_ladder"][2].pop("core_self_generation_claim")

    errors = validate_growth_plan(payload)

    assert "S2 must be the minimum core self-generation claim" in errors
