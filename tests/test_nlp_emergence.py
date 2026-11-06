from __future__ import annotations

import torch

from floorflow.nlp_flow.model.emergence import (
    EmergenceConfig,
    EmergentSemanticObjective,
)


def test_emergence_objective_is_sparse_compositional_and_finite() -> None:
    torch.manual_seed(13)
    objective = EmergentSemanticObjective(
        EmergenceConfig(
            hidden_size=16,
            basis_count=12,
            active_basis=3,
            novelty_margin=0.1,
        )
    )
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    loss = objective(hidden)
    loss.total.backward()

    assert hidden.grad is not None
    assert torch.isfinite(loss.total)
    assert 0.0 <= float(loss.support_recall) <= 1.0
    assert objective.encoder.weight.grad is not None
    assert objective.composer.weight.grad is not None
    assert objective.decoder.weight.grad is not None
    assert 0.0 <= float(loss.emergence_score.detach()) <= 1.0
    assert "emergence_score" in loss.as_dict()


def test_emergence_masks_ignore_padding() -> None:
    objective = EmergentSemanticObjective(
        EmergenceConfig(hidden_size=8, basis_count=8, active_basis=2)
    ).eval()
    hidden = torch.randn(1, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        masked = objective(hidden, mask)
        crowded = objective(
            torch.cat((hidden[:, :4], torch.randn(1, 2, 8) * 100.0), dim=1),
            mask,
        )

    assert torch.allclose(masked.total, crowded.total)


def test_support_memory_is_bounded_and_drives_novelty_distance() -> None:
    objective = EmergentSemanticObjective(
        EmergenceConfig(
            hidden_size=8,
            basis_count=8,
            active_basis=2,
            support_capacity=4,
        )
    ).train()
    hidden = torch.randn(3, 8, 8)
    with torch.no_grad():
        first = objective(hidden)
        assert int(objective.support_count) == 4
        assert int(objective.support_valid.sum()) == 4
        second = objective(hidden)

    assert torch.isfinite(first.total)
    assert torch.isfinite(second.total)
    assert first.novel_distance.ndim == 0


def test_emergence_score_favors_valid_compositions_under_fixed_head() -> None:
    objective = EmergentSemanticObjective(
        EmergenceConfig(
            hidden_size=8,
            basis_count=8,
            active_basis=2,
            support_capacity=16,
            reward_weight=0.1,
                novelty_margin=0.0,
        )
    ).eval()
    hidden = torch.randn(2, 8, 8)
    with torch.no_grad():
        objective.validity_head.weight.zero_()
        objective.validity_head.bias.copy_(torch.tensor([10.0, 10.0]))
        objective.support_vectors[0] = -torch.nn.functional.normalize(
            hidden.mean(dim=(0, 1)),
            dim=-1,
        )
        objective.support_valid[0] = True
        objective.support_count.fill_(1)
    mask = torch.ones(2, 8, dtype=torch.bool)
    valid = objective(
        hidden,
        mask,
        surface_targets=torch.ones_like(mask),
        execution_targets=torch.ones_like(mask, dtype=torch.long),
    )
    invalid = objective(
        hidden,
        mask,
        surface_targets=torch.zeros_like(mask),
        execution_targets=torch.zeros_like(mask, dtype=torch.long),
    )

    assert valid.validity_gate > invalid.validity_gate
    assert valid.emergence_score > invalid.emergence_score
