"""Atomic resumable checkpoints for small capability training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.engineering.components.checkpoint_reproducibility_runtime.checkpoint import (
    capture_rng_state,
    restore_rng_state,
)
from newera.nlp_flow.engineering.components.free_schedule_spec.storage import save_torch_atomic

CHECKPOINT_SCHEMA = "newera-capability-checkpoint-v1"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    sample_generator: torch.Generator | None = None,
    caller_state: Any = None,
    metadata: dict[str, Any] | None = None,
    rng_device: str | torch.device = "cpu",
) -> None:
    if step < 0 or samples_seen < 0:
        raise ValueError("step and samples_seen must be non-negative")
    destination = Path(path)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "samples_seen": int(samples_seen),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(rng_device),
        "sample_generator": None if sample_generator is None else sample_generator.get_state(),
        "caller_state": caller_state,
        "metadata": metadata or {},
    }
    save_torch_atomic(destination, payload)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sample_generator: torch.Generator | None = None,
    map_location: str | torch.device = "cpu",
    expected_metadata: dict[str, Any] | None = None,
    rng_device: str | torch.device | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema')}")
    metadata = payload.get("metadata", {})
    for key, value in (expected_metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint metadata mismatch for {key}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    restore_rng_state(rng, map_location if rng_device is None else rng_device)
    sampler_state = payload.get("sample_generator")
    if (sampler_state is None) != (sample_generator is None):
        raise ValueError("checkpoint sample generator configuration does not match")
    if sample_generator is not None:
        sample_generator.set_state(sampler_state.detach().cpu().to(dtype=torch.uint8))
    return {
        "step": int(payload["step"]),
        "samples_seen": int(payload["samples_seen"]),
        "caller_state": payload.get("caller_state"),
        "metadata": metadata,
    }
