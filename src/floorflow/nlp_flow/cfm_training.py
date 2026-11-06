"""Training-time utilities for faster conditional-flow convergence."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class CFMTrainingConfig:
    timestep_sampler: Literal["uniform", "logit_normal"] = "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 0.8
    timestep_epsilon: float = 1.0e-3
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.timestep_sampler not in {"uniform", "logit_normal"}:
            raise ValueError("unsupported timestep sampler")
        if self.logit_normal_std <= 0.0:
            raise ValueError("logit-normal standard deviation must be positive")
        if not 0.0 < self.timestep_epsilon < 0.5:
            raise ValueError("timestep epsilon must be in (0, 0.5)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")


def sample_flow_times(
    batch_size: int,
    config: CFMTrainingConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.timestep_sampler == "uniform":
        return torch.rand(
            batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    normal = torch.randn(
        batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return torch.sigmoid(
        config.logit_normal_mean + config.logit_normal_std * normal
    ).clamp(
        config.timestep_epsilon,
        1.0 - config.timestep_epsilon,
    )


def interpolate_flow(
    noise: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    if noise.shape != target.shape:
        raise ValueError("noise and target must have the same shape")
    if time.shape != (target.shape[0],):
        raise ValueError("time must be [batch]")
    view = (1.0 - time).reshape(
        (time.shape[0],) + (1,) * (target.ndim - 1)
    )
    return view * noise + (1.0 - view) * target


def flow_velocity_target(noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if noise.shape != target.shape:
        raise ValueError("noise and target must have the same shape")
    return target - noise


class ModelEMA:
    """Small deterministic EMA wrapper for fast checkpoint evaluation."""

    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        buffer_names = {name for name, _buffer in model.named_buffers()}
        if set(model_state) != set(ema_state):
            raise ValueError("EMA and model state keys differ")
        for name, value in model_state.items():
            ema_value = ema_state[name]
            if name not in buffer_names and value.is_floating_point():
                ema_value.lerp_(value.detach(), 1.0 - self.decay)
            else:
                ema_value.copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()


def clip_and_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    max_grad_norm: float,
) -> float:
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return float(norm.detach())


def warmup_cosine_scale(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int,
    final_scale: float = 0.05,
) -> float:
    if total_steps <= 0 or warmup_steps < 0:
        raise ValueError("invalid schedule dimensions")
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError("final_scale must be in [0, 1]")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return final_scale + (1.0 - final_scale) * 0.5 * (
        1.0 + math.cos(math.pi * min(1.0, progress))
    )


__all__ = [
    "CFMTrainingConfig",
    "ModelEMA",
    "clip_and_step",
    "flow_velocity_target",
    "interpolate_flow",
    "sample_flow_times",
    "warmup_cosine_scale",
]
