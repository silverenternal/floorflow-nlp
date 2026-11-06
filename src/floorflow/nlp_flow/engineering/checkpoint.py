"""Resumable nlp_flow checkpoint format built from shared atomic/RNG helpers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from floorflow.nlp_flow.engineering.storage import (
    cuda_rng_state,
    restore_cuda_rng_state,
    save_torch_atomic,
)

CHECKPOINT_VERSION = "nlp-checkpoint-v1"


def capture_rng_state(device: str | torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_rng_state(device),
    }


def restore_rng_state(state: Mapping[str, Any], device: str | torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("checkpoint torch RNG state must be a tensor")
    torch.set_rng_state(torch_state.detach().cpu().to(dtype=torch.uint8))
    restore_cuda_rng_state(state.get("cuda"), device)


def save_nlp_checkpoint(
    path: str | Path,
    *,
    step: int,
    samples_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    spec_sha256: str,
    data_manifest_sha256: str,
    tokenizer_sha256: str,
    config: Mapping[str, Any],
    device: str | torch.device,
    ema: Mapping[str, torch.Tensor] | None = None,
) -> None:
    if step < 0 or samples_seen < 0:
        raise ValueError("step and samples_seen must be non-negative")
    save_torch_atomic(
        path,
        {
            "version": CHECKPOINT_VERSION,
            "step": int(step),
            "samples_seen": int(samples_seen),
            "model": model.state_dict(),
            "ema": None if ema is None else dict(ema),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "rng": capture_rng_state(device),
            "provenance": {
                "spec_sha256": spec_sha256,
                "data_manifest_sha256": data_manifest_sha256,
                "tokenizer_sha256": tokenizer_sha256,
            },
            "config": dict(config),
        },
    )


def load_nlp_checkpoint(
    path: str | Path,
    *,
    expected_spec_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("checkpoint provenance must be a mapping")
    for field in ("spec_sha256", "data_manifest_sha256", "tokenizer_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint provenance lacks {field}")
    if expected_spec_sha256 is not None and provenance["spec_sha256"] != expected_spec_sha256:
        raise ValueError("checkpoint experiment spec does not match the active run")
    return payload


def restore_nlp_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, int]:
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"], device)
    return {
        "step": int(payload["step"]),
        "samples_seen": int(payload["samples_seen"]),
    }


__all__ = [
    "CHECKPOINT_VERSION",
    "capture_rng_state",
    "load_nlp_checkpoint",
    "restore_nlp_checkpoint",
    "restore_rng_state",
    "save_nlp_checkpoint",
]
