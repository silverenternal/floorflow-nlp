"""Parameter-shared serial/parallel dynamic-depth encoder and decoder stacks.

The shared transformer cell supplies the serial depth. Lightweight low-rank
adapters supply conditional parallel lanes, but only receive computation for
tokens selected by the router. This keeps physical parameters independent of
the maximum serial depth while making the actual routed work observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from newera.nlp_flow.model.components.architecture.foundation.blocks import CompactDecoderBlock


@dataclass(frozen=True)
class DynamicStackConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int
    max_serial_steps: int = 4
    parallel_lanes: int = 2
    lane_rank: int = 8
    parallel_top_k: int = 1
    min_serial_steps: int = 1
    dropout: float = 0.0
    norm_eps: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_heads",
            "num_kv_heads",
            "intermediate_size",
            "max_seq_len",
            "max_serial_steps",
            "parallel_lanes",
            "lane_rank",
            "parallel_top_k",
            "min_serial_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.min_serial_steps > self.max_serial_steps:
            raise ValueError("min_serial_steps cannot exceed max_serial_steps")
        if self.parallel_top_k > self.parallel_lanes:
            raise ValueError("parallel_top_k cannot exceed parallel_lanes")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class DynamicStackOutput:
    hidden_states: torch.Tensor
    mean_depth: torch.Tensor
    max_depth: torch.Tensor
    token_depth: torch.Tensor
    ponder_cost: torch.Tensor
    lane_weights: torch.Tensor
    lane_balance_loss: torch.Tensor
    selected_lane_counts: torch.Tensor
    halting_probabilities: torch.Tensor
    serial_token_steps: torch.Tensor
    parallel_token_lane_calls: torch.Tensor
    executed_steps: int
    dense_serial_token_calls: int


class _LaneAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.scale.to(hidden.dtype) * self.up(
            F.silu(self.down(hidden))
        )


class DynamicSharedStack(nn.Module):
    """Reuse one transformer cell across serial depth and parallel lanes."""

    def __init__(self, config: DynamicStackConfig, *, causal: bool) -> None:
        super().__init__()
        self.config = config
        self.causal = bool(causal)

        class _BlockConfig:
            hidden_size = config.hidden_size
            num_heads = config.num_heads
            num_kv_heads = config.num_kv_heads
            head_dim = config.hidden_size // config.num_heads
            kv_size = config.num_kv_heads * head_dim
            intermediate_size = config.intermediate_size
            max_seq_len = config.max_seq_len
            rope_base = 10_000.0
            norm_eps = config.norm_eps
            dropout = config.dropout

        self.cell = CompactDecoderBlock(_BlockConfig(), causal=causal)
        self.lanes = nn.ModuleList(
            [
                _LaneAdapter(config.hidden_size, config.lane_rank)
                for _ in range(config.parallel_lanes)
            ]
        )
        self.lane_router = nn.Linear(config.hidden_size, config.parallel_lanes)
        self.halting_head = nn.Sequential(
            nn.RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )

    def _depth_feature(self, step: int, hidden: torch.Tensor) -> torch.Tensor:
        half = self.config.hidden_size // 2
        frequencies = torch.exp(
            -torch.arange(half, device=hidden.device, dtype=torch.float32)
            * (9.210340371976184 / max(1, half - 1))
        )
        phase = float(step + 1) * frequencies
        values = torch.cat((phase.sin(), phase.cos()))
        if values.numel() < self.config.hidden_size:
            values = F.pad(values, (0, self.config.hidden_size - values.numel()))
        return values.to(hidden.dtype)[None, None]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        halt_bias: torch.Tensor | None = None,
        lane_bias: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        *,
        max_steps: int | None = None,
    ) -> DynamicStackOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [batch, sequence, hidden]")
        batch, sequence, _hidden = hidden_states.shape
        steps = self.config.max_serial_steps if max_steps is None else int(max_steps)
        if not 1 <= steps <= self.config.max_serial_steps:
            raise ValueError("max_steps is outside the configured serial depth")
        mask = (
            torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        if mask.shape != (batch, sequence):
            raise ValueError("attention_mask must match hidden states")
        if halt_bias is not None:
            halt_bias = halt_bias.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if halt_bias.shape != (batch, sequence):
                raise ValueError("halt_bias must match hidden token states")
        if lane_bias is not None:
            lane_bias = lane_bias.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if lane_bias.shape != (
                batch,
                sequence,
                self.config.parallel_lanes,
            ):
                raise ValueError("lane_bias must match batch, sequence, and lanes")

        state = hidden_states
        active = mask.clone().bool()
        continuous_active = mask.to(hidden_states.dtype)
        cumulative_halt = torch.zeros(batch, sequence, device=hidden_states.device)
        expected_depth = torch.zeros_like(cumulative_halt)
        token_depth = torch.zeros_like(cumulative_halt)
        route_probability_sum = torch.zeros(
            batch,
            sequence,
            self.config.parallel_lanes,
            device=hidden_states.device,
        )
        halting_probabilities = torch.zeros(
            batch,
            sequence,
            self.config.max_serial_steps,
            device=hidden_states.device,
        )
        selected_lane_counts = torch.zeros(
            self.config.parallel_lanes,
            dtype=torch.long,
            device=hidden_states.device,
        )
        executed_steps = 0
        for step in range(steps):
            if not active.any():
                break
            depth_feature = self._depth_feature(step, state)
            proposal = self.cell(
                state + depth_feature,
                attention_mask=mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
            )

            router_logits = self.lane_router(state + depth_feature)
            if lane_bias is not None:
                router_logits = router_logits + lane_bias
            router_probabilities = torch.softmax(router_logits, dim=-1)
            top_values, top_indices = torch.topk(
                router_probabilities,
                k=self.config.parallel_top_k,
                dim=-1,
                sorted=True,
            )
            top_gates = top_values / top_values.sum(dim=-1, keepdim=True)
            route_probability_sum = (
                route_probability_sum
                + router_probabilities * active[..., None].to(router_probabilities.dtype)
            )

            lane_delta = torch.zeros_like(proposal)
            flat_lane_delta = lane_delta.reshape(batch * sequence, -1)
            for lane_index, lane in enumerate(self.lanes):
                selected = (top_indices == lane_index).any(dim=-1)
                selected = selected & active
                lane_token_count = selected.sum()
                selected_lane_counts[lane_index] += lane_token_count
                if lane_token_count == 0:
                    continue
                batch_indices, sequence_indices = selected.nonzero(as_tuple=True)
                flat_indices = batch_indices * sequence + sequence_indices
                gate = (
                    top_gates * (top_indices == lane_index)
                ).sum(dim=-1)[batch_indices, sequence_indices]
                update = lane(
                    state[batch_indices, sequence_indices]
                )
                flat_lane_delta = torch.index_add(
                    flat_lane_delta,
                    0,
                    flat_indices,
                    update * gate[..., None],
                )
            proposal = proposal + flat_lane_delta.reshape_as(proposal)

            halt_probability = torch.sigmoid(
                self.halting_head(proposal).squeeze(-1)
                + (0.0 if halt_bias is None else halt_bias)
            )
            if step + 1 < self.config.min_serial_steps:
                halt_probability = torch.zeros_like(halt_probability)
            elif step + 1 == steps:
                halt_probability = torch.ones_like(halt_probability)
            halting_probabilities[..., step] = (
                halt_probability * active.to(halt_probability.dtype)
            )
            blended = (1.0 - halt_probability[..., None]) * state + (
                halt_probability[..., None] * proposal
            )
            active_float = active.to(blended.dtype)
            state = state + active_float[..., None] * (blended - state)
            cumulative_halt = (
                cumulative_halt + active_float * halt_probability
            )
            expected_depth = expected_depth + continuous_active
            continuous_active = continuous_active * (1.0 - halt_probability)
            token_depth = token_depth + active.float()
            can_halt = step + 1 >= self.config.min_serial_steps
            active = active & ((not can_halt) | cumulative_halt.lt(1.0))
            executed_steps = step + 1

        average_lane_weights = (
            route_probability_sum / token_depth.clamp_min(1.0)[..., None]
        )
        lane_usage = average_lane_weights.sum(dim=(0, 1))
        lane_usage = lane_usage / lane_usage.sum().clamp_min(1.0)
        lane_balance_loss = lane_usage.square().sum() * (
            self.config.parallel_lanes
        ) - 1.0
        valid_tokens = mask.sum().clamp_min(1)
        mean_depth = token_depth.sum() / valid_tokens
        max_depth = token_depth.masked_fill(~mask, 0.0).amax()
        expected_mean_depth = expected_depth.sum() / valid_tokens
        serial_token_steps = token_depth.sum()
        parallel_token_lane_calls = selected_lane_counts.sum()
        return DynamicStackOutput(
            hidden_states=state,
            mean_depth=mean_depth,
            max_depth=max_depth,
            token_depth=token_depth,
            ponder_cost=expected_mean_depth,
            lane_weights=average_lane_weights,
            lane_balance_loss=lane_balance_loss,
            selected_lane_counts=selected_lane_counts,
            halting_probabilities=halting_probabilities,
            serial_token_steps=serial_token_steps,
            parallel_token_lane_calls=parallel_token_lane_calls,
            executed_steps=executed_steps,
            dense_serial_token_calls=executed_steps * batch * sequence,
        )


class DynamicStackDecoder(DynamicSharedStack):
    def __init__(self, config: DynamicStackConfig) -> None:
        super().__init__(config, causal=True)


class DynamicStackEncoder(DynamicSharedStack):
    def __init__(self, config: DynamicStackConfig) -> None:
        super().__init__(config, causal=False)


__all__ = [
    "DynamicSharedStack",
    "DynamicStackConfig",
    "DynamicStackDecoder",
    "DynamicStackEncoder",
    "DynamicStackOutput",
]
