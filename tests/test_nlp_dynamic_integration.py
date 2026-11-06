from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def _config(max_steps: int) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        parameter_cap=2_000_000,
        enable_dynamic_stack=True,
        dynamic_max_steps=max_steps,
        dynamic_min_steps=1,
        dynamic_parallel_lanes=2,
        dynamic_parallel_top_k=1,
        dynamic_lane_rank=4,
    )


def test_compact_decoder_reuses_dynamic_stack_parameters_and_reports_compute() -> None:
    shallow = CompactNLPDecoder(_config(2)).eval()
    deep = CompactNLPDecoder(_config(8)).eval()
    assert len(shallow.blocks) == len(deep.blocks) == 0
    assert shallow.dynamic_stack is not None
    assert deep.dynamic_stack is not None
    assert count_unique_trainable_parameters(shallow) == (
        count_unique_trainable_parameters(deep)
    )

    input_ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        left = shallow(input_ids)
        right = deep(input_ids)

    assert left.token_logits.shape == right.token_logits.shape == (2, 12, 128)
    assert left.dynamic_diagnostics is not None
    assert left.dynamic_diagnostics["executed_steps"] <= 2
    assert right.dynamic_diagnostics is not None
    assert right.dynamic_diagnostics["executed_steps"] <= 8
    assert left.dynamic_diagnostics["serial_token_steps"] > 0
    assert left.dynamic_diagnostics["parallel_token_lane_calls"] == (
        left.dynamic_diagnostics["serial_token_steps"]
    )


def test_dynamic_depth_objective_backpropagates_through_halting() -> None:
    model = CompactNLPDecoder(_config(4)).train()
    input_ids = torch.randint(0, 128, (2, 8))
    output = model(input_ids)
    assert output.dynamic_diagnostics is not None

    task_loss = F.cross_entropy(
        output.token_logits[:, :-1].reshape(-1, 128),
        input_ids[:, 1:].reshape(-1),
    )
    depth_loss = (
        output.dynamic_diagnostics["ponder_cost"] - 2.0
    ).square()
    (task_loss + 0.1 * depth_loss).backward()

    assert model.dynamic_stack is not None
    halting_gradient = model.dynamic_stack.halting_head[-1].weight.grad
    assert halting_gradient is not None
    assert torch.isfinite(halting_gradient).all()
