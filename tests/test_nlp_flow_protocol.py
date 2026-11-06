from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.protocol import (
    EXPECTED_STAGES,
    EXPECTED_TYPED_CHANNELS,
    protocol_from_payload,
    validate_protocol_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def load_payload() -> dict:
    return json.loads((ROOT / "configs/nlp_cross_domain_v1.json").read_text())


def test_language_protocol_is_target_free_and_matched() -> None:
    protocol = protocol_from_payload(load_payload())

    assert protocol.target_free
    assert protocol.stage_sequence == EXPECTED_STAGES
    assert [level.level_id for level in protocol.hierarchy] == ["L0", "L1", "L2", "L3"]
    assert protocol.typed_channels == EXPECTED_TYPED_CHANNELS


def test_rust_generated_contract_matches_frozen_config() -> None:
    payload = load_payload()
    generated_levels = [
        {
            "id": level.level_id,
            "name": level.name,
            "role": level.role,
            "max_state_tokens": level.max_state_tokens,
        }
        for level in protocol_from_payload(payload).hierarchy
    ]

    assert generated_levels == payload["hierarchy"]
    assert set(EXPECTED_TYPED_CHANNELS) == set(payload["typed_channels"])


def test_language_protocol_rejects_target_leakage() -> None:
    payload = load_payload()
    payload["inference_contract"]["required_inputs"].append("target_texts")

    errors = validate_protocol_payload(payload)

    assert any("overlap" in error for error in errors)


def test_language_protocol_rejects_unmatched_arm() -> None:
    payload = load_payload()
    payload["arms"]["candidate"]["tokenizer"] = "different-tokenizer"

    errors = validate_protocol_payload(payload)

    assert "matched arm field differs: tokenizer" in errors


def test_language_protocol_rejects_large_model_stacking() -> None:
    payload = load_payload()
    payload["budget_contract"]["primary_claim_parameter_cap"] = 100_000_000
    payload["budget_contract"]["net_parameter_growth_allowed"] = True

    errors = validate_protocol_payload(payload)

    assert "primary claim parameter cap must not exceed 23M" in errors
    assert "net parameter growth must be disabled" in errors


def test_nlp_todo_declares_every_protocol_phase() -> None:
    todo = json.loads((ROOT / "nlptodo.json").read_text())

    assert tuple(todo["phases"]) == EXPECTED_STAGES
    assert {task["phase"] for task in todo["tasks"]} <= set(EXPECTED_STAGES)
    assert {task["phase"] for task in todo["tasks"]} >= {
        "NLP-PHASE-MODEL",
        "NLP-PHASE-SCREEN",
        "NLP-PHASE-SPECIALIZE",
        "NLP-PHASE-EVALUATE",
        "NLP-PHASE-CERTIFY",
    }


@pytest.mark.parametrize(
    "field",
    [
        "held_out_token_nll",
        "constraint_success",
        "grammar_validity",
        "decoder_nfe",
    ],
)
def test_metric_ledger_covers_primary_contracts(field: str) -> None:
    todo = json.loads((ROOT / "nlptodo.json").read_text())
    values = {
        metric
        for group in todo["metric_ledger"].values()
        for metric in group
    }

    assert field in values
