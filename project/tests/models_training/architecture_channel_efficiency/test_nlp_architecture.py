from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.research.architecture.search.architecture import (
    EXPECTED_ATTENTION,
    EXPECTED_CHANNELS,
    validate_architecture_plan,
    validated_channels,
)

ROOT = Path(__file__).resolve().parents[4]


def load_architecture() -> dict:
    return json.loads((ROOT / "project/configs/models_training/architecture/definitions/nlp_architecture.json").read_text())


def test_architecture_contract_covers_channels_attention_cfm_and_scopes() -> None:
    payload = load_architecture()

    assert validate_architecture_plan(payload) == ()
    assert validated_channels(payload) == EXPECTED_CHANNELS
    assert set(payload["attention"]) == EXPECTED_ATTENTION


def test_architecture_rejects_missing_staged_cross_attention() -> None:
    payload = load_architecture()
    payload["attention"].pop("typed_cross_attention")

    errors = validate_architecture_plan(payload)

    assert any("attention must contain exactly" in error for error in errors)


def test_architecture_requires_planned_mixed_corpus() -> None:
    payload = load_architecture()
    payload["data_usage"]["code"]["languages"] = ["python"]

    errors = validate_architecture_plan(payload)

    assert "code data usage is missing rust" in errors
    assert "code data usage is missing cpp" in errors


def test_architecture_rejects_large_parameter_stacking() -> None:
    payload = load_architecture()
    payload["shared_core"]["parameter_cap"] = 100_000_000

    errors = validate_architecture_plan(payload)

    assert "shared-core cap must not exceed 23M" in errors


def test_architecture_rejects_target_ast_at_inference() -> None:
    payload = load_architecture()
    payload["cfm"]["inference"] = "Use target AST and the token decoder."

    errors = validate_architecture_plan(payload)

    assert any("generate AST first" in error for error in errors)


def test_architecture_requires_two_stage_control() -> None:
    payload = load_architecture()
    payload["cfm"].pop("ablation_rule")

    errors = validate_architecture_plan(payload)

    assert any("one-stage control" in error for error in errors)


def test_architecture_requires_native_strict_json() -> None:
    payload = load_architecture()
    payload["structured_output"]["native_valid_by_construction"] = False
    payload["structured_output"]["posthoc_repair"] = True

    errors = validate_architecture_plan(payload)

    assert "structured output must be valid by construction" in errors
    assert "structured output cannot use post-hoc repair or retry" in errors
