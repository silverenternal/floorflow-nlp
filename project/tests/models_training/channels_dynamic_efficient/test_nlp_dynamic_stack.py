from __future__ import annotations

import pytest
import torch

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model.components.architecture.state.dynamic_stack import (
    DynamicStackConfig,
    DynamicStackDecoder,
    DynamicStackEncoder,
)


def _config(
    max_steps: int = 4,
    *,
    min_steps: int = 1,
    top_k: int = 1,
) -> DynamicStackConfig:
    return DynamicStackConfig(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        max_serial_steps=max_steps,
        min_serial_steps=min_steps,
        parallel_lanes=2,
        lane_rank=4,
        parallel_top_k=top_k,
    )


def test_serial_depth_adds_compute_without_adding_parameters() -> None:
    shallow = DynamicStackDecoder(_config(2))
    deep = DynamicStackDecoder(_config(8))

    assert count_unique_trainable_parameters(shallow) == count_unique_trainable_parameters(deep)


def test_decoder_is_causal_and_returns_dynamic_depth_diagnostics() -> None:
    model = DynamicStackDecoder(_config()).eval()
    first = torch.randn(1, 6, 32)
    second = first.clone()
    second[:, 3:] += 10.0

    with torch.no_grad():
        left = model(first)
        right = model(second)

    assert torch.allclose(
        left.hidden_states[:, :3],
        right.hidden_states[:, :3],
        atol=1.0e-6,
    )
    assert left.mean_depth.ndim == 0
    assert left.max_depth.ndim == 0
    assert torch.allclose(
        left.lane_weights.sum(dim=-1),
        torch.ones_like(left.lane_weights[..., 0]),
    )
    assert left.serial_token_steps.item() > 0
    assert left.parallel_token_lane_calls.item() == left.serial_token_steps.item()
    assert left.executed_steps >= 1
    assert left.dense_serial_token_calls >= (
        left.serial_token_steps.item()
    )


def test_encoder_has_bidirectional_context() -> None:
    model = DynamicStackEncoder(_config()).eval()
    first = torch.randn(1, 6, 32)
    second = first.clone()
    second[:, 3:] += 10.0

    with torch.no_grad():
        left = model(first)
        right = model(second)

    assert not torch.allclose(
        left.hidden_states[:, :3],
        right.hidden_states[:, :3],
    )


def test_top_k_lane_compute_is_routed_and_reported() -> None:
    model = DynamicStackDecoder(
        _config(max_steps=3, min_steps=3, top_k=2)
    ).eval()
    hidden = torch.randn(2, 5, 32)

    with torch.no_grad():
        output = model(hidden)

    expected = output.serial_token_steps * 2
    assert output.executed_steps == 3
    assert output.parallel_token_lane_calls == expected
    assert output.selected_lane_counts.sum() == expected
    assert torch.all(output.selected_lane_counts > 0)


def test_only_selected_lane_adapter_receives_dense_compute_gradient() -> None:
    model = DynamicStackDecoder(_config(max_steps=2, min_steps=1))
    with torch.no_grad():
        model.halting_head[-1].bias.fill_(20.0)
        model.lane_router.weight.zero_()
        model.lane_router.bias.copy_(torch.tensor([10.0, -10.0]))
    hidden = torch.randn(1, 4, 32, requires_grad=True)

    output = model(hidden)
    output.hidden_states.square().mean().backward()

    assert model.lanes[0].down.weight.grad is not None
    assert model.lanes[0].up.weight.grad is not None
    assert model.lanes[1].down.weight.grad is None
    assert model.lanes[1].up.weight.grad is None
    assert output.selected_lane_counts.tolist() == [4, 0]


def test_halting_stops_serial_compute_after_minimum_depth() -> None:
    model = DynamicStackDecoder(_config(max_steps=4, min_steps=1)).eval()
    with torch.no_grad():
        model.halting_head[-1].bias.fill_(20.0)
        output = model(torch.randn(2, 5, 32))

    assert output.executed_steps == 1
    assert torch.equal(output.token_depth, torch.ones_like(output.token_depth))
    assert torch.all(output.halting_probabilities[..., 1:] == 0.0)


def test_expected_depth_objective_reaches_the_halting_head() -> None:
    model = DynamicStackDecoder(_config(max_steps=4, min_steps=1))
    output = model(torch.randn(2, 5, 32))

    objective = (output.ponder_cost - 2.0).square()
    objective.backward()

    assert model.halting_head[-1].weight.grad is not None
    assert torch.isfinite(model.halting_head[-1].weight.grad).all()
    assert model.halting_head[-1].weight.grad.abs().sum() > 0.0


def test_minimum_depth_prevents_early_exit() -> None:
    model = DynamicStackDecoder(_config(max_steps=4, min_steps=3)).eval()
    with torch.no_grad():
        model.halting_head[-1].bias.fill_(20.0)
        output = model(torch.randn(1, 4, 32))

    assert output.executed_steps == 3
    assert torch.equal(output.token_depth, torch.full_like(output.token_depth, 3.0))
    assert output.max_depth.item() == 3.0
    assert torch.all(output.halting_probabilities[..., :2] == 0.0)


def test_external_unified_planner_can_bias_halting() -> None:
    model = DynamicStackDecoder(_config(max_steps=4, min_steps=1)).eval()
    hidden = torch.randn(2, 5, 32)

    with torch.no_grad():
        early = model(hidden, halt_bias=torch.full((2, 5), 20.0))
        late = model(hidden, halt_bias=torch.full((2, 5), -20.0))

    assert early.executed_steps == 1
    assert late.executed_steps == 4


def test_external_unified_planner_can_route_parallel_lanes() -> None:
    model = DynamicStackDecoder(_config(max_steps=2, min_steps=2)).eval()
    hidden = torch.randn(2, 5, 32)
    lane_bias = torch.tensor([20.0, -20.0])[None, None].expand(2, 5, -1)

    with torch.no_grad():
        output = model(hidden, lane_bias=lane_bias)

    assert output.selected_lane_counts[0] > 0
    assert output.selected_lane_counts[1] == 0


def test_expected_depth_counts_compute_at_the_final_step() -> None:
    model = DynamicStackDecoder(_config(max_steps=4, min_steps=1)).eval()
    with torch.no_grad():
        model.halting_head[-1].bias.fill_(-20.0)
        output = model(torch.randn(2, 5, 32))

    assert output.executed_steps == 4
    assert torch.isclose(output.ponder_cost, torch.tensor(4.0), atol=1.0e-5)


def test_padding_is_invariant_and_excluded_from_depth() -> None:
    model = DynamicStackDecoder(_config(max_steps=3)).eval()
    first = torch.randn(1, 6, 32)
    second = first.clone()
    second[:, 3:] = 1000.0
    mask = torch.tensor([[True, True, True, False, False, False]])

    with torch.no_grad():
        left = model(first, attention_mask=mask)
        right = model(second, attention_mask=mask)

    assert torch.allclose(
        left.hidden_states[:, :3],
        right.hidden_states[:, :3],
        atol=1.0e-6,
    )
    assert torch.all(left.token_depth[:, 3:] == 0.0)
    assert torch.isclose(left.mean_depth, right.mean_depth)


def test_fixed_depth_choices_change_compute_not_parameters() -> None:
    shallow = DynamicStackEncoder(
        _config(max_steps=4, min_steps=4)
    ).eval()
    deep = DynamicStackEncoder(
        _config(max_steps=4, min_steps=4)
    ).eval()
    deep.load_state_dict(shallow.state_dict())
    hidden = torch.randn(2, 7, 32)

    with torch.no_grad():
        output_2 = shallow(hidden, max_steps=2)
        output_4 = deep(hidden, max_steps=4)

    assert count_unique_trainable_parameters(shallow) == (
        count_unique_trainable_parameters(deep)
    )
    assert output_2.executed_steps == 2
    assert output_4.executed_steps == 4
    assert output_4.serial_token_steps == 2 * output_2.serial_token_steps


def test_config_rejects_invalid_dynamic_routing() -> None:
    with pytest.raises(ValueError, match="min_serial_steps"):
        DynamicStackConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            max_serial_steps=2,
            min_serial_steps=3,
        )
    with pytest.raises(ValueError, match="parallel_top_k"):
        DynamicStackConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            parallel_lanes=2,
            parallel_top_k=3,
        )
    with pytest.raises(ValueError, match="configured serial depth"):
        model = DynamicStackDecoder(_config(max_steps=3))
        model(torch.randn(1, 4, 32), max_steps=4)
