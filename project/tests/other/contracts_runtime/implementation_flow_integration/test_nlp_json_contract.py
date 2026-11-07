from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.generation.outputs.json.json_contract import validate_json_contract

ROOT = Path(__file__).resolve().parents[5]


def load_contract() -> dict:
    return json.loads((ROOT / "project/configs/other/contracts_runtime/contract_freeze_implementation/nlp_json_contract.json").read_text())


def test_json_contract_is_native_hard_validity() -> None:
    payload = load_contract()

    assert validate_json_contract(payload) == ()
    assert payload["native_generation"]["posthoc_repair"] is False
    assert payload["validity_contract"]["strict_json_parse_validity"] == 1.0


def test_json_contract_rejects_training_only_guarantee() -> None:
    payload = load_contract()
    payload["validity_contract"]["training_only_guarantee_possible"] = True

    errors = validate_json_contract(payload)

    assert "unconstrained training alone cannot guarantee 100% JSON validity" in errors


def test_json_contract_rejects_posthoc_repair() -> None:
    payload = load_contract()
    payload["native_generation"]["posthoc_repair"] = True

    errors = validate_json_contract(payload)

    assert "post-hoc JSON repair is forbidden" in errors


def test_json_contract_requires_zero_retry_and_full_test_validity() -> None:
    payload = load_contract()
    payload["hard_gates"]["test_strict_json_validity"] = 0.999
    payload["hard_gates"]["retry_count"] = 1

    errors = validate_json_contract(payload)

    assert "test JSON validity must equal 1.0" in errors
    assert "repair and retry counts must be zero" in errors
