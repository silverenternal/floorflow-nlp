"""Small training loop primitives shared by capability experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from newera.capabilities.contracts import Capability, TaskAdapter


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    optimizer_steps: int
    samples_seen: int
    mean_loss: float
    elapsed_seconds: float
    gradient_coverage: dict[str, Any]


def sample_indices(population_size: int, sample_count: int, generator: torch.Generator):
    """Yield reproducible sample-with-replacement indices from a CPU generator."""
    if population_size <= 0 or sample_count < 0:
        raise ValueError("population_size must be positive and sample_count non-negative")
    for _ in range(sample_count):
        yield int(torch.randint(population_size, (1,), generator=generator).item())


def execute_task(
    capability: Capability[Any, Any],
    adapter: TaskAdapter[Any, Any, Any, Any],
    task_input: Any,
    *,
    return_raw: bool = False,
) -> Any:
    """Apply the task boundary without embedding task-answer logic in the model."""
    capability_output = capability.execute(adapter.encode(task_input))
    task_output = adapter.decode(capability_output)
    return (capability_output, task_output) if return_raw else task_output


def train_with_accumulation(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    loss_for: Callable[[nn.Module, Any, int], Tensor],
    accumulation_steps: int,
    start_step: int = 0,
    progress: Callable[[int, int, float], None] | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> TrainingSummary:
    """Optimize mean per-sample loss over batches that cannot be vectorized.

    ``batches`` must yield one sample at a time. A final partial accumulation
    window is scaled by its true size, so it is not underweighted.
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    window_count = 0
    optimizer_steps = 0
    samples_seen = 0
    gradient_coverage: dict[str, Any] = {}
    total_hint = getattr(batches, "total_samples", None)

    def flush() -> None:
        nonlocal window_count, optimizer_steps
        if window_count == 0:
            return
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(window_count)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        window_count = 0
        if on_step is not None:
            on_step(start_step + optimizer_steps, samples_seen)

    for sample_index, batch in enumerate(batches):
        loss = loss_for(model, batch, start_step + optimizer_steps)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        if sample_index == 0:
            parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            connected = [(name, parameter.grad) for name, parameter in parameters if parameter.grad is not None]
            finite = [name for name, gradient in connected if torch.isfinite(gradient).all().item()]
            nonzero = [name for name, gradient in connected if torch.count_nonzero(gradient).item() > 0]
            gradient_coverage = {
                "parameter_tensor_count": len(parameters),
                "gradient_connected_count": len(connected),
                "finite_gradient_count": len(finite),
                "nonzero_gradient_count": len(nonzero),
                "disconnected_parameters": [name for name, _ in parameters if name not in {item[0] for item in connected}],
                "nonfinite_gradient_parameters": [name for name, gradient in connected if name not in finite],
            }
        losses.append(float(loss.detach().item()))
        window_count += 1
        samples_seen += 1
        if window_count == accumulation_steps:
            flush()
            if progress is not None:
                progress(samples_seen, total_hint or 0, losses[-1])
    flush()
    if progress is not None and samples_seen and samples_seen % accumulation_steps:
        progress(samples_seen, total_hint or samples_seen, losses[-1])
    return TrainingSummary(
        optimizer_steps=optimizer_steps,
        samples_seen=samples_seen,
        mean_loss=sum(losses) / len(losses) if losses else 0.0,
        elapsed_seconds=time.perf_counter() - started,
        gradient_coverage=gradient_coverage,
    )
