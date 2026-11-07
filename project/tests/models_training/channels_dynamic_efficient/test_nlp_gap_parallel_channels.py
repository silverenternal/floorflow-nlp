from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _validate_contract(payload: dict) -> tuple[str, ...]:
    path = ROOT / "project/scripts/audit/models/objectives/audit_nlp_gap_parallel_channels.py"
    spec = importlib.util.spec_from_file_location("gap_contract_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load gap contract audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_contract(payload)


def _payload() -> dict:
    return json.loads(
        (ROOT / "project/configs/models_training/learning/objectives/nlp_gap_parallel_training.json").read_text(
            encoding="utf-8"
        )
    )


def test_gap_parallel_training_contract_passes() -> None:
    assert _validate_contract(_payload()) == ()


def test_contract_rejects_unsplit_open_capability_claims() -> None:
    payload = copy.deepcopy(_payload())
    payload["tracks"]["open_capability_exploratory"]["claim_eligible"] = True

    errors = _validate_contract(payload)

    assert any("ineligible" in error for error in errors)


def test_contract_rejects_stage_order_regression() -> None:
    payload = copy.deepcopy(_payload())
    payload["stages"][0]["parallelizable"] = False

    errors = _validate_contract(payload)

    assert any("parallelizable" in error for error in errors)


def test_contract_rejects_test_leakage_or_repair() -> None:
    payload = copy.deepcopy(_payload())
    payload["forbidden"].remove("JSON repair")

    errors = _validate_contract(payload)

    assert any("JSON repair" in error for error in errors)
