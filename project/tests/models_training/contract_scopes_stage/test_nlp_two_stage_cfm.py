from __future__ import annotations

import torch

from newera.nlp_flow.engineering import audit_parameter_budget, count_unique_trainable_parameters
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig, TwoStageASTCFM
from newera.nlp_flow.model.components.training.objectives.cfm import flow_matching_stage_loss


def test_two_stage_cfm_predict_shapes_and_gradients() -> None:
    cfm = TwoStageASTCFM(32, rank=8)
    context = torch.randn(2, 6, 32)
    ast_state = torch.randn(2, 6, 32)
    result_state = torch.randn(2, 6, 32)
    time = torch.rand(2)

    output = cfm.predict(
        ast_state,
        result_state,
        time,
        time,
        context,
    )
    loss = output.ast_velocity.square().mean() + output.result_velocity.square().mean()
    loss.backward()

    assert output.ast_velocity.shape == (2, 6, 32)
    assert cfm.stage_a.down.weight.grad is not None
    assert cfm.stage_b.down.weight.grad is not None


def test_two_stage_sampling_uses_generated_ast_not_target() -> None:
    cfm = TwoStageASTCFM(16, rank=4).eval()
    context = torch.randn(2, 5, 16)
    ast_noise = torch.zeros_like(context)
    result_noise = torch.zeros_like(context)

    output = cfm.sample(
        context,
        steps_a=2,
        steps_b=2,
        ast_noise=ast_noise,
        result_noise=result_noise,
    )

    assert output.ast_state.shape == context.shape
    assert output.result_state.shape == context.shape
    assert not torch.equal(output.ast_state, ast_noise)
    assert not torch.equal(output.result_state, result_noise)


def test_flow_matching_loss_masks_and_backpropagates() -> None:
    velocity = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn(2, 4, 8)
    noise = torch.randn(2, 4, 8)
    mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)[None, :, None]

    loss = flow_matching_stage_loss(velocity, target, noise, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert velocity.grad is not None


def test_full_p2m_candidate_fits_budget_with_cfm() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            enable_typed_channels=True,
            enable_typed_fusion=True,
            enable_two_stage_cfm=True,
            enable_json_action_head=True,
        )
    )
    parameters = count_unique_trainable_parameters(model)
    report = audit_parameter_budget(parameters, 46_000_000)

    assert report.passed
