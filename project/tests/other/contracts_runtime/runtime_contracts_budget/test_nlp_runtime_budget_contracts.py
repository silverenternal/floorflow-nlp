from __future__ import annotations

import json
from pathlib import Path

import pytest

from newera.nlp_flow.components.platform.contracts.runtime.runtime_budget_contract import (
    BudgetCaps,
    BudgetRequest,
    KnobSource,
    check_budget,
    validate_knob_ledger,
)


def _caps():
    return BudgetCaps(256, 65536, 8, 4, 32, 8)


def test_node_source_depth_lane_nfe_and_expert_caps_are_hard_and_reported() -> None:
    request = BudgetRequest(257, 65537, 9, 5, 33, 9, KnobSource.LEARNED)
    result = check_budget(request, _caps())
    assert not result["feasible"]
    assert {item["field"] for item in result["violations"]} == {
        "nodes", "source_bytes", "depth", "lanes", "nfe", "generated_experts",
    }


def test_feasible_budget_and_stopping_source_are_explicit() -> None:
    request = BudgetRequest(80, 4000, 4, 2, 8, 1, KnobSource.LEARNED)
    result = check_budget(request, _caps())
    assert result["feasible"]
    assert result["stopping_source"] == "learned"


def test_runtime_knobs_have_traceable_provenance_without_zero_tuning_claim() -> None:
    root = Path(__file__).resolve().parents[5]
    payload = json.loads((root / "project/configs/other/contracts_runtime/knob_ledger_runtime/nlp_runtime_knob_ledger.json").read_text())
    validate_knob_ledger(payload)
    assert payload["zero_tuning_certificate"] is False
    assert payload["per_task_answer_overrides"] is False


def test_validation_selected_knob_requires_frozen_validation_receipt() -> None:
    root = Path(__file__).resolve().parents[5]
    payload = json.loads((root / "project/configs/other/contracts_runtime/knob_ledger_runtime/nlp_runtime_knob_ledger.json").read_text())
    payload["knobs"]["runtime.trigger_threshold"] = {
        "value": 0.5,
        "source": "validation_selected",
        "evidence": "selection",
    }
    with pytest.raises(ValueError, match="selection receipt"):
        validate_knob_ledger(payload)
