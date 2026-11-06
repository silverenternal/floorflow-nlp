from __future__ import annotations

import torch

from floorflow.nlp_flow.result_cfm import GeneratedASTResultCFM, ResultCFMConfig
from floorflow.nlp_flow.stage_b_execution import (
    NoResultFlowControl,
    ResultExecutionHead,
    execute_program,
)


def test_executable_programs_are_deterministic() -> None:
    assert execute_program(0, 5, 3) == 8
    assert execute_program(1, 5, 3) == 2
    assert execute_program(2, 5, 3) == 15
    assert execute_program(3, 5, 3) == 5


def test_stage_b_execution_head_trains_and_control_has_no_ast_input() -> None:
    torch.manual_seed(9)
    model = GeneratedASTResultCFM(ResultCFMConfig(hidden_size=16, rank=6))
    head = ResultExecutionHead(16)
    control = NoResultFlowControl(16)
    prompt = torch.randn(4, 16)
    generated_ast = torch.randn(4, 7, 16)
    mask = torch.ones(4, 7, dtype=torch.bool)
    target = torch.randn(4, 5, 16)
    loss, output = model.training_loss(
        target,
        prompt,
        generated_ast,
        mask,
    )
    logits = head(output.endpoint)
    action_loss = torch.nn.functional.cross_entropy(
        logits,
        torch.arange(4),
    )
    (loss.total + action_loss).backward()

    assert head.proj[1].weight.grad is not None
    assert control(prompt).shape == (4, 4)
    assert "ast" not in str(control.forward.__annotations__)
