from __future__ import annotations

import pytest

from newera.nlp_flow.components.generation.outputs.contracts.output_contract import (
    OUTPUT_CONTRACTS,
    validate_output_contract,
)
from newera.nlp_flow.components.generation.trees.typed.typed_tree import TypedTreeValidity


def test_json_requires_structure_and_schema_but_not_execution() -> None:
    decision = validate_output_contract(
        "json",
        TypedTreeValidity(
            structural=True,
            schema=True,
            executable=False,
        ),
    )

    assert decision.valid
    assert decision.contract.required_dimensions == ("structural", "schema")
    assert not OUTPUT_CONTRACTS["json"].executable


def test_patch_requires_application_and_code_requires_execution() -> None:
    schema_only = TypedTreeValidity(
        structural=True,
        schema=True,
        executable=False,
    )

    assert not validate_output_contract("patch", schema_only).valid
    assert not validate_output_contract("code", schema_only).valid
    with pytest.raises(ValueError, match="executable"):
        validate_output_contract("patch", schema_only).require_valid()


def test_structured_json_cannot_hide_a_schema_failure() -> None:
    decision = validate_output_contract(
        "json",
        TypedTreeValidity(
            structural=True,
            schema=False,
            executable=True,
        ),
    )

    assert not decision.valid
    assert decision.missing == ("schema",)
