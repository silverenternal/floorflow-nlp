from __future__ import annotations

import pytest
import torch

from floorflow.nlp_flow.model.competence import (
    CompetenceCell,
    UnifiedCompetenceBank,
    UnifiedCompetenceConfig,
)


def _bank() -> UnifiedCompetenceBank:
    return UnifiedCompetenceBank(
        UnifiedCompetenceConfig(
            hidden_size=32,
            competence_names=(
                "json",
                "ast",
                "python",
                "rust",
                "engineering",
                "language",
            ),
            cell_rank=8,
            top_k=3,
            planner_dim=16,
            generated_rank=8,
            generated_basis_count=3,
            max_depth_steps=4,
            parallel_lanes=2,
        )
    )


def test_competence_cells_and_bank_start_as_identity() -> None:
    hidden = torch.randn(2, 6, 32)
    cell = CompetenceCell(32, 8).eval()
    with torch.no_grad():
        cell_output = cell(hidden)
        bank_output = _bank().eval()(hidden)

    assert torch.equal(cell_output, torch.zeros_like(hidden))
    assert torch.equal(bank_output.hidden_states, hidden)
    assert bank_output.planner.depth_logits.shape == (2, 6)
    assert bank_output.planner.birth_logits.shape == (2, 6)
    assert bank_output.diagnostics["route_mode"] == "unified"


def test_unified_planner_routes_and_generates_all_control_signals() -> None:
    bank = _bank()
    hidden = torch.randn(2, 7, 32)
    output = bank(hidden)

    assert set(output.competence_states) == set(bank.cells)
    assert output.planner.competence_logits.shape == (2, 7, 6)
    assert output.planner.lane_logits.shape == (2, 7, 2)
    assert output.planner.mode_logits.shape == (2, 7, 4)
    assert output.planner.budget_logits.shape == (2, 7)
    assert torch.isfinite(output.planner.birth_logits).all()
    assert 0.0 <= output.diagnostics["mean_compute_budget"].item() <= 1.0
    assert output.diagnostics["inference_generated_weights"].item() == 1.0
    assert output.diagnostics["generated_weight_elements"].item() > 0.0
    assert output.diagnostics["competence_token_calls"] < (
        output.diagnostics["dense_competence_token_calls"]
    )


def test_generated_expert_can_be_frozen_independently() -> None:
    bank = _bank()
    bank.freeze_generated_expert()

    assert all(
        not parameter.requires_grad
        for parameter in bank.generated_expert.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in bank.cells.parameters()
    )


def test_unified_bank_can_materialize_e_for_static_inference() -> None:
    bank = _bank()
    hidden = torch.randn(2, 5, 32)
    materialized = bank.materialize_generated_expert(hidden)
    output = bank.apply_materialized_generated_expert(hidden, materialized)

    assert output.shape == hidden.shape
    assert materialized.left.shape[0] == 2
    assert materialized.rank.shape == (2,)


def test_generated_e_slot_can_be_disabled_without_changing_shared_cells() -> None:
    bank = UnifiedCompetenceBank(
        UnifiedCompetenceConfig(
            hidden_size=32,
            competence_names=("json", "ast", "python"),
            cell_rank=8,
            top_k=2,
            planner_dim=16,
            generated_rank=8,
            generated_basis_count=3,
            generated_expert_mode="disabled",
        )
    ).eval()
    hidden = torch.randn(2, 5, 32)
    output = bank(hidden)

    assert torch.equal(output.hidden_states, hidden)
    assert bank.generated_expert is None
    assert output.diagnostics["generated_expert_mode"] == "disabled"
    assert output.diagnostics["generated_weight_elements"].item() == 0.0
    assert output.diagnostics["generated_basis_resident_parameters"].item() == 0.0


def test_disabled_e_removes_resident_factory_parameters() -> None:
    online = _bank()
    disabled = UnifiedCompetenceBank(
        UnifiedCompetenceConfig(
            hidden_size=32,
            competence_names=("json", "ast", "python"),
            cell_rank=8,
            top_k=2,
            planner_dim=16,
            generated_rank=8,
            generated_basis_count=3,
            generated_expert_mode="disabled",
        )
    )

    online_parameters = sum(
        parameter.numel() for parameter in online.parameters()
    )
    disabled_parameters = sum(
        parameter.numel() for parameter in disabled.parameters()
    )
    assert disabled_parameters < online_parameters


def test_runtime_competence_rejects_disabled_generated_e() -> None:
    from floorflow.nlp_flow.model.config import NLPModelConfig

    with pytest.raises(ValueError, match="online generated experts"):
        NLPModelConfig(
            vocab_size=128,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            parameter_cap=2_000_000,
            enable_unified_competence=True,
            unified_top_k=2,
            unified_planner_dim=8,
            unified_cell_rank=4,
            unified_generated_rank=4,
            unified_generated_basis_count=2,
            unified_generated_expert_mode="disabled",
            enable_runtime_competence=True,
        )
