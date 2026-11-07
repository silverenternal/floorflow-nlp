from __future__ import annotations

import inspect

import pytest
import torch

from newera.nlp_flow.components.ast.flow.core.ast_cfm import ASTCodecVocabulary, ASTStageAFlow, encode_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast
from newera.nlp_flow.components.ast.flow.runtime.result_cfm import GeneratedASTResultCFM, ResultCFMConfig
from newera.nlp_flow.components.generation.execution.stages.two_stage_control import (
    OneStageResultCFMControl,
    TwoStageASTFirstPipeline,
    match_control_width,
)


def _pipeline() -> tuple[
    TwoStageASTFirstPipeline,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    graphs = [
        parse_ast("def add(a, b):\n    return a + b\n", "python"),
        parse_ast("fn add(a: i32, b: i32) -> i32 { a + b }\n", "rust"),
    ]
    vocabulary = ASTCodecVocabulary.from_graphs(graphs)
    batch = encode_ast_graph(graphs[0], vocabulary, max_nodes=32, template_id=1)
    stage_a = ASTStageAFlow(
        vocabulary,
        hidden_size=16,
        num_heads=4,
        max_nodes=32,
    )
    stage_b = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=6))
    return (
        TwoStageASTFirstPipeline(stage_a, stage_b),
        batch.level_masks,
        batch.node_mask,
        batch.parent_indices,
        batch.levels,
    )


def test_pipeline_uses_generated_ast_and_matches_total_nfe() -> None:
    pipeline, level_masks, node_mask, parents, levels = _pipeline()
    prompt = torch.randn(1, 16)
    ast_noise = torch.randn(1, 32, 16)
    result_noise = torch.randn(1, 5, 16)
    output = pipeline.sample_fixture_latent(
        prompt,
        node_mask[None],
        level_masks={name: value[None] for name, value in level_masks.items()},
        parent_indices=parents[None],
        node_levels=levels[None],
        result_length=5,
        steps_a=3,
        steps_b=5,
        ast_noise=ast_noise,
        result_noise=result_noise,
    )

    assert output.total_nfe == 8
    assert output.result_output.endpoint.shape == result_noise.shape
    parameters = inspect.signature(pipeline.sample_target_free).parameters
    assert not any("target" in name for name in parameters)
    assert not any("endpoint" in name or "latent" in name for name in parameters)


def test_one_stage_control_has_no_ast_or_target_input() -> None:
    control = OneStageResultCFMControl(
        ResultCFMConfig(hidden_size=16, rank=6),
        condition_width=16,
    )
    output = control.sample(
        torch.randn(2, 16),
        result_length=5,
        steps=8,
        noise=torch.randn(2, 5, 16),
    )

    parameters = inspect.signature(control.sample).parameters
    assert not any("ast" in name for name in parameters)
    assert not any("target" in name for name in parameters)
    assert output.endpoint.shape == (2, 5, 16)


def test_parameter_matching_selects_closest_control_width() -> None:
    pipeline, _level_masks, _node_mask, _parents, _levels = _pipeline()
    control, match = match_control_width(
        pipeline,
        candidate_widths=tuple(range(16, 1_025, 8)),
        condition_slots=8,
    )

    assert control.condition.slots == 8
    assert match.control_parameters > 0
    assert match.relative_difference < 0.02


def test_pipeline_blocks_stage_b_when_generated_ast_quality_is_low() -> None:
    pipeline, level_masks, node_mask, parents, levels = _pipeline()
    pipeline.stage_b = GeneratedASTResultCFM(
        ResultCFMConfig(
            hidden_size=16,
            rank=6,
            enforce_ast_quality=True,
        )
    )

    with pytest.raises(ValueError, match="quality gate"):
        pipeline.sample_fixture_latent(
            torch.randn(1, 16),
            node_mask[None],
            level_masks={
                name: value[None] for name, value in level_masks.items()
            },
            parent_indices=parents[None],
            node_levels=levels[None],
            result_length=5,
            steps_a=2,
            steps_b=2,
            ast_noise=torch.randn(1, 32, 16),
            result_noise=torch.randn(1, 5, 16),
            ast_quality=torch.tensor([0.1]),
        )


def test_verified_inference_requires_receipt_request_and_has_no_latent_shortcuts() -> None:
    pipeline, *_rest = _pipeline()
    parameters = inspect.signature(pipeline.sample_verified).parameters

    assert {"language", "request", "parsed_ast_encoder"} <= set(parameters)
    assert not {"node_mask", "ast_handoff", "parent_indices", "node_levels"} & set(parameters)
    assert not any("endpoint" in name or "latent" in name for name in parameters)


def test_final_target_free_pipeline_has_no_target_or_node_mask_input() -> None:
    pipeline, *_rest = _pipeline()
    parameters = inspect.signature(pipeline.sample_target_free).parameters

    assert "language" in parameters
    assert "node_mask" not in parameters
    assert not any("target" in name for name in parameters)
    assert not any("teacher" in name for name in parameters)
    token_parameters = inspect.signature(
        pipeline.sample_target_free_from_tokens
    ).parameters
    assert "input_ids" in token_parameters
    assert not any("target" in name for name in token_parameters)
