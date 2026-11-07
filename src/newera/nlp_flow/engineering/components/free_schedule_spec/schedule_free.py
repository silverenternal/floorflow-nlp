"""Schedule-free AdamW for horizon-independent NLP training.

Adapted from Meta's Apache-2.0 reference implementation:
https://github.com/facebookresearch/schedule_free
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """AdamW without a fixed training-horizon learning-rate schedule."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
    ) -> None:
        if lr <= 0.0 or eps <= 0.0:
            raise ValueError("lr and eps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        defaults = {
            "lr": float(lr),
            "betas": betas,
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "warmup_steps": int(warmup_steps),
            "r": float(r),
            "weight_lr_power": float(weight_lr_power),
            "k": 0,
            "train_mode": False,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "scheduled_lr": 0.0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            for parameter in group["params"]:
                state = self.state[parameter]
                if "y" in state:
                    parameter.copy_(state["y"])
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            for parameter in group["params"]:
                state = self.state[parameter]
                if "x" in state:
                    parameter.copy_(state["x"])
            group["train_mode"] = False

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError("ScheduleFreeAdamW.step requires train mode")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            step = group["k"]
            warmup_steps = group["warmup_steps"]
            scheduled = (
                (step + 1) / warmup_steps
                if step < warmup_steps
                else 1.0
            )
            lr = group["lr"] * scheduled
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])
            weight = (
                (step + 1) ** group["r"]
                * lr_max ** group["weight_lr_power"]
            )
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum
            average = weight / weight_sum if weight_sum else 0.0
            bias_correction2 = 1.0 - beta2 ** (step + 1)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "z" not in state:
                    state["z"] = parameter.detach().clone()
                    state["x"] = parameter.detach().clone()
                    state["y"] = parameter.detach().clone()
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                z = state["z"]
                x = state["x"]
                y = state["y"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad
                if decay:
                    z.sub_(y, alpha=lr * decay)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )
                denominator = (
                    exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                )
                z.addcdiv_(gradient, denominator, value=-lr)
                x.mul_(1.0 - average).add_(z, alpha=average)
                y.copy_(x.mul(beta1).add_(z, alpha=1.0 - beta1))
                parameter.copy_(y)
            group["k"] = step + 1
        return loss


__all__ = ["ScheduleFreeAdamW"]
