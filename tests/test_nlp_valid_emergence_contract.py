from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.emergence_contract import (
    valid_emergence_reward,
    validate_mechanism_queue,
)


def _validity():
    return {
        "ast_valid": True,
        "native_schema_valid": True,
        "application_valid": True,
        "execution_valid": True,
    }


def test_only_valid_supported_holdout_compositions_receive_novelty_credit() -> None:
    assert valid_emergence_reward(0.8, supported=True, holdout=True, validity=_validity()) == 0.8
    assert valid_emergence_reward(0.8, supported=False, holdout=True, validity=_validity()) == 0.0
    assert valid_emergence_reward(0.8, supported=True, holdout=False, validity=_validity()) == 0.0
    invalid = _validity()
    invalid["native_schema_valid"] = False
    assert valid_emergence_reward(0.8, supported=True, holdout=True, validity=invalid) == 0.0


def test_mechanism_queue_has_fixed_random_pool_and_generated_controls_without_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs/nlp_mechanism_ablation_v11.json").read_text())
    validate_mechanism_queue(payload)
    assert payload["execution"]["run_enabled"] is False
    assert payload["execution"]["gpu_enabled"] is False


def test_invalid_emergence_values_and_incomplete_mechanism_queue_fail() -> None:
    with pytest.raises(ValueError, match="finite"):
        valid_emergence_reward(float("nan"), supported=True, holdout=True, validity=_validity())
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs/nlp_mechanism_ablation_v11.json").read_text())
    payload["controls"].pop()
    with pytest.raises(ValueError, match="each fixed and generated"):
        validate_mechanism_queue(payload)
