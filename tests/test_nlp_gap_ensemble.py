from __future__ import annotations

import pytest
import torch

from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.gap_ensemble import (
    GAP_NAMES,
    GapEnsemble,
    GapEnsembleConfig,
)


def _config(*, gaps: tuple[str, ...] = GAP_NAMES) -> GapEnsembleConfig:
    return GapEnsembleConfig(
        hidden_size=32,
        enabled_gaps=gaps,
        adapter_rank=4,
        fusion_dim=16,
        heads=4,
    )


def test_zero_gated_aggregator_is_identity_at_initialization() -> None:
    model = GapEnsemble(_config()).eval()
    hidden = torch.randn(2, 7, 32)

    with torch.no_grad():
        output = model(hidden)

    assert torch.equal(output.hidden_states, hidden)
    assert set(output.gap_states) == set(GAP_NAMES)
    assert output.routing_weights.shape == (2, 7, len(GAP_NAMES))
    assert torch.allclose(
        output.routing_weights.sum(dim=-1),
        torch.ones(2, 7),
    )
    assert all(torch.isfinite(value).all() for value in output.gap_logits.values())


def test_single_gap_loss_reaches_only_its_adapter_and_head() -> None:
    model = GapEnsemble(_config())
    hidden = torch.randn(2, 5, 32)
    output = model(hidden)
    targets = torch.randint(0, 2, output.gap_logits["python_execution"].shape)

    loss = model.supervised_loss(output, "python_execution", targets.float())
    loss.backward()

    assert model.adapters["python_execution"].down.weight.grad is not None
    assert model.loss_heads["python_execution"].weight.grad is not None
    for name in GAP_NAMES:
        if name == "python_execution":
            continue
        assert model.adapters[name].down.weight.grad is None
        assert model.loss_heads[name].weight.grad is None
    assert model.aggregator.query.weight.grad is None


def test_aggregator_receives_cross_gap_gradient() -> None:
    model = GapEnsemble(_config())
    with torch.no_grad():
        model.aggregator.gate.fill_(1.0)
    hidden = torch.randn(2, 5, 32)
    output = model(hidden)

    output.hidden_states.square().mean().backward()

    assert model.aggregator.gate.grad is not None
    assert model.aggregator.query.weight.grad is not None
    assert model.aggregator.key.weight.grad is not None
    assert model.aggregator.out.weight.grad is not None
    assert all(
        model.adapters[name].down.weight.grad is not None
        for name in GAP_NAMES
    )


def test_disabled_and_inactive_gaps_have_no_compute_or_parameters() -> None:
    model = GapEnsemble(
        _config(gaps=("json_validity", "ast_structure"))
    )
    assert set(model.adapters) == {"json_validity", "ast_structure"}
    assert set(model.loss_heads) == {"json_validity", "ast_structure"}

    output = model(
        torch.randn(1, 6, 32),
        active_gaps=("ast_structure",),
    )
    assert set(output.gap_states) == {"ast_structure"}
    assert set(output.gap_logits) == {"ast_structure"}
    assert output.routing_weights.shape[-1] == 1

    with pytest.raises(ValueError, match="unavailable"):
        model(
            torch.randn(1, 6, 32),
            active_gaps=("python_execution",),
        )


def test_frozen_router_control_is_identity_without_aggregation() -> None:
    model = GapEnsemble(_config())
    hidden = torch.randn(2, 6, 32)
    with torch.no_grad():
        output = model(hidden, router_enabled=False)

    assert torch.equal(output.hidden_states, hidden)
    assert torch.allclose(
        output.routing_weights,
        torch.full((2, 6, len(GAP_NAMES)), 1.0 / len(GAP_NAMES)),
    )
    assert output.diagnostics["router_enabled"].item() == 0.0


def test_gap_loss_mask_and_parameter_budget() -> None:
    model = GapEnsemble(GapEnsembleConfig(hidden_size=160))
    hidden = torch.randn(2, 9, 160)
    output = model(hidden, active_gaps=("json_validity", "ast_structure"))
    logits = output.gap_logits["json_validity"]
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, 1:4] = True

    loss = model.supervised_loss(
        output,
        "json_validity",
        targets,
        mask,
    )
    assert torch.isfinite(loss)
    assert sum(model.parameter_breakdown().values()) < 150_000

    with pytest.raises(ValueError, match="no active"):
        model.supervised_loss(
            output,
            "json_validity",
            targets,
            torch.zeros_like(mask),
        )


def test_compact_model_surfaces_gap_diagnostics_with_identity_initialization() -> None:
    config = NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        parameter_cap=1_000_000,
        enable_gap_ensemble=True,
        gap_adapter_rank=4,
        gap_fusion_dim=16,
        gap_heads=4,
    )
    model = CompactNLPDecoder(config).eval()
    with torch.no_grad():
        output = model(torch.randint(0, 128, (2, 8)))

    assert output.gap_diagnostics is not None
    assert output.gap_logits is not None
    assert output.gap_states is not None
    assert set(output.gap_logits) == set(GAP_NAMES)
    assert output.gap_diagnostics["aggregator_gate"].item() == 0.0
    assert output.token_logits.shape == (2, 8, 128)
