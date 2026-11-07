"""Fail-closed tensor checks with asynchronous CUDA assertions."""

from __future__ import annotations

import torch


def assert_true_async(
    condition: torch.Tensor,
    message: str,
) -> None:
    if condition.numel() != 1:
        raise ValueError("assertion condition must be scalar")
    if condition.is_cuda:
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise ValueError(message)


__all__ = ["assert_true_async"]
