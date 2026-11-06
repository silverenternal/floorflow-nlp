from __future__ import annotations

import json
from pathlib import Path

from floorflow.nlp_flow.source_registry import validate_source_registry

ROOT = Path(__file__).resolve().parents[1]


def load_registry() -> dict:
    return json.loads((ROOT / "configs/nlp_source_registry_v1.json").read_text())


def test_source_registry_covers_code_model_and_engineering() -> None:
    assert validate_source_registry(load_registry()) == ()


def test_source_registry_requires_revision_and_license() -> None:
    payload = load_registry()
    payload["sources"][0]["revision"] = ""
    payload["sources"][1]["license"] = ""

    errors = validate_source_registry(payload)

    assert any("revision must be non-empty" in error for error in errors)
    assert any("license must be non-empty" in error for error in errors)


def test_conditional_source_cannot_allow_redistribution() -> None:
    payload = load_registry()
    linux = next(row for row in payload["sources"] if row["id"] == "linux_documentation")
    linux["redistribution"] = "permitted"

    errors = validate_source_registry(payload)

    assert any("restricts redistribution" in error for error in errors)
