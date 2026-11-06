from __future__ import annotations

import pytest
import torch

from floorflow.nlp_flow.structured_generation import TypedTreeGateway


def test_gateway_routes_ast_and_json_through_one_result_contract() -> None:
    ast = TypedTreeGateway.from_ast(
        "def add(a, b):\n    return a + b\n",
        "python",
    )
    data = TypedTreeGateway.from_json('{"status":"ok","count":2}')

    assert ast.mode == "ast"
    assert data.mode == "json"
    assert ast.tree.domain == "ast"
    assert data.tree.domain == "json"
    assert ast.validity.structural
    assert data.validity.structural
    assert data.text == '{"status":"ok","count":2}'


def test_gateway_uses_planner_mode_logits_and_explicit_validity() -> None:
    mode = TypedTreeGateway.select_mode(
        torch.tensor([[0.0, 0.0, 4.0, 1.0]])
    )
    result = TypedTreeGateway.from_json(
        {"files": [], "summary": "ok"},
        domain="patch",
    )
    result = TypedTreeGateway.with_validity(
        result,
        schema=True,
        executable=True,
    )

    assert mode == "json"
    assert result.mode == "patch"
    assert result.validity.all_valid


def test_gateway_uses_domain_specific_json_contract() -> None:
    result = TypedTreeGateway.from_json(
        '{"status":"ok"}',
        schema_valid=True,
    )

    decision = TypedTreeGateway.contract_decision(result)

    assert decision.valid
    assert decision.contract.required_dimensions == ("structural", "schema")
    assert TypedTreeGateway.require_contract(result) is result


def test_gateway_rejects_patch_without_application_evidence() -> None:
    result = TypedTreeGateway.from_json(
        {"files": []},
        domain="patch",
        schema_valid=True,
    )

    with pytest.raises(ValueError, match="executable"):
        TypedTreeGateway.require_contract(result)
