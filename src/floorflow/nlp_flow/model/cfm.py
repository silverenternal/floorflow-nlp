"""Two-stage AST-first conditional flow-matching heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(1, half - 1)
        )
        phases = time[:, None] * frequencies[None]
        embedding = torch.cat([phases.sin(), phases.cos()], dim=-1)
        if self.dimension % 2:
            embedding = torch.cat([embedding, torch.zeros_like(time[:, None])], dim=-1)
        return embedding


class LowRankVectorField(nn.Module):
    """Low-rank conditional velocity field for one CFM stage."""

    def __init__(self, hidden_size: int, rank: int = 32) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.time = nn.Sequential(
            TimeEmbedding(rank),
            nn.Linear(rank, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )
        self.down = nn.Linear(hidden_size, rank)
        self.context = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size)
        self.norm = nn.RMSNorm(hidden_size)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim not in {2, 3} or state.shape[-1] != self.hidden_size:
            raise ValueError("state must be [batch, (sequence,) hidden_size]")
        if context.ndim == 2:
            if context.shape != (state.shape[0], self.hidden_size):
                raise ValueError("context must be [batch, hidden_size]")
            context_hidden = self.context(context)
            if state.ndim == 3:
                context_hidden = context_hidden[:, None]
        elif context.ndim == 3:
            if state.ndim != 3:
                raise ValueError("rank-three context requires a rank-three state")
            if (
                context.shape[0] != state.shape[0]
                or context.shape[-1] != self.hidden_size
            ):
                raise ValueError("context must share batch and hidden size")
            context_hidden = self.context(context)
        else:
            raise ValueError("context must be rank two or three")
        if time.shape != (state.shape[0],):
            raise ValueError("time must be [batch]")
        time_hidden = self.time(time)
        if state.ndim == 3:
            time_hidden = time_hidden[:, None]
        hidden = (
            self.down(state)
            + context_hidden
            + time_hidden
        )
        velocity = self.up(torch.nn.functional.silu(hidden))
        return self.norm(velocity + state)


@dataclass(frozen=True)
class TwoStageFlowOutput:
    ast_state: torch.Tensor
    result_state: torch.Tensor
    ast_velocity: torch.Tensor
    result_velocity: torch.Tensor


class TwoStageASTCFM(nn.Module):
    """Stage A AST flow followed by Stage B generated-AST result flow."""

    def __init__(self, hidden_size: int, rank: int = 32) -> None:
        super().__init__()
        self.stage_a = LowRankVectorField(hidden_size, rank)
        self.stage_b = LowRankVectorField(hidden_size, rank)

    def predict(
        self,
        ast_state: torch.Tensor,
        result_state: torch.Tensor,
        time_a: torch.Tensor,
        time_b: torch.Tensor,
        context: torch.Tensor,
    ) -> TwoStageFlowOutput:
        ast_velocity = self.stage_a(ast_state, time_a, context)
        result_context = context + ast_state
        result_velocity = self.stage_b(result_state, time_b, result_context)
        return TwoStageFlowOutput(
            ast_state=ast_state,
            result_state=result_state,
            ast_velocity=ast_velocity,
            result_velocity=result_velocity,
        )

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        *,
        steps_a: int = 4,
        steps_b: int = 4,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
    ) -> TwoStageFlowOutput:
        ast_state = (
            torch.randn_like(context)
            if ast_noise is None
            else ast_noise.to(context)
        )
        for step in range(max(1, int(steps_a))):
            time = torch.full(
                (context.shape[0],),
                step / max(1, int(steps_a)),
                device=context.device,
                dtype=context.dtype,
            )
            ast_state = ast_state + self.stage_a(ast_state, time, context) / max(1, int(steps_a))
        result_state = (
            torch.randn_like(context)
            if result_noise is None
            else result_noise.to(context)
        )
        for step in range(max(1, int(steps_b))):
            time = torch.full(
                (context.shape[0],),
                step / max(1, int(steps_b)),
                device=context.device,
                dtype=context.dtype,
            )
            result_state = (
                result_state
                + self.stage_b(result_state, time, context + ast_state)
                / max(1, int(steps_b))
            )
        return TwoStageFlowOutput(
            ast_state=ast_state,
            result_state=result_state,
            ast_velocity=torch.zeros_like(ast_state),
            result_velocity=torch.zeros_like(result_state),
        )


def flow_matching_stage_loss(
    velocity: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if velocity.shape != target.shape or velocity.shape != noise.shape:
        raise ValueError("velocity, target, and noise shapes must match")
    error = (velocity - (target - noise)).square()
    if mask is not None:
        try:
            expanded = torch.broadcast_to(mask, velocity.shape)
        except RuntimeError as error_info:
            raise ValueError("flow mask must broadcast to velocity") from error_info
        error = error * expanded.float()
        return error.sum() / expanded.float().sum().clamp_min(1.0)
    return error.mean()


__all__ = [
    "LowRankVectorField",
    "TimeEmbedding",
    "TwoStageASTCFM",
    "TwoStageFlowOutput",
    "flow_matching_stage_loss",
]
