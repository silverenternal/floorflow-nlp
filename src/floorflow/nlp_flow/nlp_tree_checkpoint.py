"""Deterministic checkpoint state for bounded hierarchical tree training."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

CHECKPOINT_VERSION = "nlp-tree-checkpoint-v1"


def _parameter_trainability(model: nn.Module) -> dict[str, bool]:
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def capture_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    sampler: torch.Generator,
    identity: Mapping[str, str],
    counters: Mapping[str, int],
    active_scopes: tuple[str, ...],
    effective_batch_size: int,
) -> dict[str, Any]:
    """Capture all mutable state needed to reproduce the next bounded update."""
    if not identity or any(not value for value in identity.values()):
        raise ValueError("checkpoint identity hashes must be present and non-empty")
    if any(value < 0 for value in counters.values()):
        raise ValueError("compute counters cannot be negative")
    if effective_batch_size < 1:
        raise ValueError("effective batch size must be positive")
    return {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_training_mode": tuple(
            bool(group.get("train_mode", False)) for group in optimizer.param_groups
        ),
        "model_training_mode": model.training,
        "parameter_trainability": _parameter_trainability(model),
        "sampler_rng": sampler.get_state(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        "identity": dict(identity),
        "counters": dict(counters),
        "active_scopes": tuple(active_scopes),
        "effective_batch_size": effective_batch_size,
    }


def restore_training_checkpoint(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    sampler: torch.Generator,
    expected_identity: Mapping[str, str],
    expected_effective_batch_size: int,
) -> dict[str, Any]:
    """Restore state and reject changed data, parser, model or tokenizer identity."""
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported hierarchical tree checkpoint version")
    if dict(checkpoint.get("identity", {})) != dict(expected_identity):
        raise ValueError("checkpoint data/parser/model identity changed")
    if int(checkpoint.get("effective_batch_size", -1)) != expected_effective_batch_size:
        raise ValueError("checkpoint effective batch size changed")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    trainability = checkpoint["parameter_trainability"]
    parameters = dict(model.named_parameters())
    if set(parameters) != set(trainability):
        raise ValueError("checkpoint parameter scopes do not match this model")
    for name, parameter in parameters.items():
        parameter.requires_grad_(bool(trainability[name]))
    sampler.set_state(checkpoint["sampler_rng"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    cuda_rng = checkpoint.get("cuda_rng", ())
    if cuda_rng:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_rng)
    model.train(bool(checkpoint["model_training_mode"]))
    expected_optimizer_modes = tuple(checkpoint["optimizer_training_mode"])
    actual_optimizer_modes = tuple(
        bool(group.get("train_mode", False)) for group in optimizer.param_groups
    )
    if actual_optimizer_modes != expected_optimizer_modes:
        raise ValueError("optimizer training mode did not restore exactly")
    return {
        "counters": dict(checkpoint["counters"]),
        "active_scopes": tuple(checkpoint["active_scopes"]),
        "identity": dict(checkpoint["identity"]),
        "effective_batch_size": int(checkpoint["effective_batch_size"]),
    }


__all__ = ["CHECKPOINT_VERSION", "capture_training_checkpoint", "restore_training_checkpoint"]
