"""Deterministic execution controls for NLP experiments."""

from __future__ import annotations

import os

import torch


def enable_deterministic_execution() -> None:
    """Request deterministic CUDA kernels and disable benchmark-dependent choices."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


__all__ = ["enable_deterministic_execution"]
