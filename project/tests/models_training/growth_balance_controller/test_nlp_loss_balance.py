from __future__ import annotations

import torch

from newera.nlp_flow.components.research.training.objectives.loss_balance import NLP_OBJECTIVES, NLPJointLossBalancer


def test_learned_balancer_covers_every_joint_objective() -> None:
    balancer = NLPJointLossBalancer(NLP_OBJECTIVES)
    losses = {
        name: torch.tensor(float(index + 1), requires_grad=True)
        for index, name in enumerate(NLP_OBJECTIVES)
    }

    total, weights = balancer(losses)
    total.backward()

    assert set(weights) == set(NLP_OBJECTIVES)
    assert torch.isfinite(total)
    assert balancer.log_variance.grad is not None
    assert balancer.log_variance.grad.abs().sum() > 0


def test_learned_balancer_rejects_missing_or_extra_objectives() -> None:
    balancer = NLPJointLossBalancer(("ast", "validity"))

    try:
        balancer({"ast": torch.tensor(1.0)})
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing objective was accepted")
