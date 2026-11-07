from __future__ import annotations

import pytest
import torch
from torch import nn

from newera.nlp_flow.components.platform.contracts.runtime.compute_ledger import (
    build_compute_ledger,
    unique_resident_parameter_elements,
)


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4, bias=False)
        self.right = nn.Linear(4, 4, bias=False)
        self.right.weight = self.left.weight


def test_tied_storage_counts_once_and_generated_basis_is_resident() -> None:
    model = _TiedModel()
    assert unique_resident_parameter_elements(model) == 16
    model.register_parameter("left_basis", nn.Parameter(torch.zeros(2, 3)))
    assert unique_resident_parameter_elements(model) == 22


def test_compute_ledger_separates_actual_work_from_dense_reference_and_costs() -> None:
    model = _TiedModel()
    diagnostics = {
        "competence_token_calls": torch.tensor(5),
        "dense_competence_token_calls": torch.tensor(20),
        "generated_weight_elements": torch.tensor(32),
        "generated_apply_flops_proxy": torch.tensor(128),
    }
    ledger = build_compute_ledger(
        model,
        diagnostics,
        control_id="hierarchical",
        router_flops=100,
        attention_flops=200,
        serial_depth=3,
        lane_calls=4,
        nfe=2,
        latency_ms=1.5,
        peak_memory_bytes=4096,
    )
    assert ledger.resident_parameter_elements == 16
    assert ledger.materialized_expert_parameter_elements == 32
    assert ledger.selected_cell_token_calls == 5
    assert ledger.dense_cell_token_calls == 20
    assert ledger.total_accounted_flops == 428
    assert ledger.as_dict()["control_id"] == "hierarchical"


def test_compute_ledger_fails_when_executed_counters_are_missing_or_impossible() -> None:
    model = _TiedModel()
    with pytest.raises(ValueError, match="omit executed"):
        build_compute_ledger(
            model, {}, control_id="flat", router_flops=0, attention_flops=0,
            serial_depth=1, lane_calls=0, nfe=1,
        )
    with pytest.raises(ValueError, match="exceed their dense"):
        build_compute_ledger(
            model,
            {"competence_token_calls": 2, "dense_competence_token_calls": 1},
            control_id="flat", router_flops=0, attention_flops=0,
            serial_depth=1, lane_calls=0, nfe=1,
        )


def test_fixed_and_generated_control_ledgers_expose_their_different_executed_work() -> None:
    model = _TiedModel()
    dense_calls = 18
    fixed = build_compute_ledger(
        model,
        {"competence_token_calls": 0, "dense_competence_token_calls": dense_calls},
        control_id="fixed_pool_off",
        router_flops=12,
        attention_flops=30,
        serial_depth=1,
        lane_calls=2,
        nfe=1,
    )
    generated = build_compute_ledger(
        model,
        {
            "competence_token_calls": 6,
            "dense_competence_token_calls": dense_calls,
            "generated_weight_elements": 24,
            "generated_apply_flops_proxy": 48,
        },
        control_id="generated_e_on",
        router_flops=20,
        attention_flops=30,
        serial_depth=2,
        lane_calls=4,
        nfe=3,
    )
    assert fixed.selected_cell_token_calls == 0
    assert generated.selected_cell_token_calls == 6
    assert generated.materialized_expert_parameter_elements == 24
    assert generated.serial_depth > fixed.serial_depth
    assert generated.nfe > fixed.nfe
