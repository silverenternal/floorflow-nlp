from __future__ import annotations

import inspect

import pytest
import torch

from floorflow.nlp_flow.result_cfm import (
    GeneratedASTCurriculum,
    GeneratedASTResultCFM,
    ResultCFMConfig,
)


def _inputs():
    torch.manual_seed(5)
    prompt = torch.randn(2, 16)
    generated_ast = torch.randn(2, 7, 16)
    ast_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, False, False, False, False],
        ]
    )
    return prompt, generated_ast, ast_mask


def test_stage_b_all_objectives_receive_gradients() -> None:
    model = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=6))
    prompt, generated_ast, ast_mask = _inputs()
    target = torch.randn(2, 5, 16)
    loss, output = model.training_loss(
        target,
        prompt,
        generated_ast,
        ast_mask,
        verifier_target=torch.tensor([1.0, 0.0]),
        time=torch.tensor([0.2, 0.8]),
        noise=torch.randn_like(target),
    )
    loss.total.backward()

    assert output.endpoint.shape == target.shape
    for name, head in (
        ("vector_field", model.vector_field.state),
        ("conditioner", model.conditioner.project[0]),
        ("verifier", model.verifier),
        ("consistency", model.consistency),
        ("uncertainty", model.uncertainty),
    ):
        assert head.weight.grad is not None, name
    for value in loss.as_dict().values():
        assert torch.isfinite(value)


def test_primary_sample_has_no_target_or_teacher_ast_argument() -> None:
    model = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=6)).eval()
    prompt, generated_ast, ast_mask = _inputs()
    output = model.sample(
        prompt,
        generated_ast,
        ast_mask,
        result_length=5,
        steps=4,
        noise=torch.randn(2, 5, 16),
    )

    parameters = inspect.signature(model.sample).parameters
    assert "generated_ast" in parameters
    assert not any("target" in name for name in parameters)
    assert not any("teacher" in name for name in parameters)
    assert output.endpoint.shape == (2, 5, 16)


def test_curriculum_ends_with_generated_ast_only() -> None:
    curriculum = GeneratedASTCurriculum(100)
    assert curriculum.at(0).phase == "teacher"
    assert curriculum.at(50).phase == "mixed"
    assert curriculum.at(99) == curriculum.at(99)
    assert curriculum.at(99).teacher_weight == 0.0
    assert curriculum.at(99).generated_weight == 1.0

    teacher = torch.zeros(1, 3, 8)
    generated = torch.ones(1, 3, 8)
    blended, schedule = curriculum.blend(99, teacher, generated)
    assert schedule.phase == "generated_only"
    assert torch.equal(blended, generated)


def test_stage_b_rejects_low_quality_generated_ast_when_enforced() -> None:
    model = GeneratedASTResultCFM(
        ResultCFMConfig(
            hidden_size=16,
            rank=6,
            enforce_ast_quality=True,
            minimum_ast_quality=0.8,
        )
    ).eval()
    prompt = torch.randn(1, 16)
    generated_ast = torch.randn(1, 5, 16)
    ast_mask = torch.ones(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="quality gate"):
        model.sample(
            prompt,
            generated_ast,
            ast_mask,
            torch.tensor([0.79]),
            result_length=4,
            steps=2,
        )
    output = model.sample(
        prompt,
        generated_ast,
        ast_mask,
        torch.tensor([1.0]),
        result_length=4,
        steps=2,
    )
    assert torch.isfinite(output.endpoint).all()


def test_stage_b_conditioner_is_sensitive_to_generated_ast_topology() -> None:
    torch.manual_seed(47)
    model = GeneratedASTResultCFM(
        ResultCFMConfig(hidden_size=16, rank=6)
    ).eval()
    prompt, generated_ast, ast_mask = _inputs()
    chain_parents = torch.tensor(
        [
            [-1, 0, 1, 2, 3, -2, -2],
            [-1, 0, 1, -2, -2, -2, -2],
        ]
    )
    flat_parents = torch.tensor(
        [
            [-1, 0, 0, 0, 0, -2, -2],
            [-1, 0, 0, -2, -2, -2, -2],
        ]
    )
    levels = ast_mask.long() * 2
    noise = torch.randn(2, 5, 16)

    chain = model.sample(
        prompt,
        generated_ast,
        ast_mask,
        parent_indices=chain_parents,
        node_levels=levels,
        result_length=5,
        steps=2,
        noise=noise,
    )
    flat = model.sample(
        prompt,
        generated_ast,
        ast_mask,
        parent_indices=flat_parents,
        node_levels=levels,
        result_length=5,
        steps=2,
        noise=noise,
    )

    assert not torch.equal(chain.endpoint, flat.endpoint)


def test_stage_b_selects_its_own_nfe_from_predicted_uncertainty() -> None:
    model = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=6)).eval()
    prompt, generated_ast, ast_mask = _inputs()
    output = model.sample(
        prompt,
        generated_ast,
        ast_mask,
        result_length=5,
        noise=torch.randn(2, 5, 16),
    )

    assert 1 <= output.executed_nfe <= 32
