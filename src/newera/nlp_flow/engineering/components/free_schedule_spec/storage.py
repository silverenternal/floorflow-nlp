"""Atomic persistence and CUDA RNG helpers for NLP runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def cuda_rng_state(device: str | torch.device) -> torch.Tensor | None:
    resolved = torch.device(device)
    return None if resolved.type != "cuda" else torch.cuda.get_rng_state(resolved)


def restore_cuda_rng_state(
    state: torch.Tensor | None, device: str | torch.device
) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and state is not None:
        torch.cuda.set_rng_state(state.cpu(), resolved)


def save_torch_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "cuda_rng_state", "restore_cuda_rng_state", "save_torch_atomic", "write_json_atomic"
]
