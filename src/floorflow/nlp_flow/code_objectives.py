"""Fill-in-the-middle, diff, and bounded verifier-feedback objectives."""

from __future__ import annotations

import difflib
import hashlib
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FIMExample:
    input_ids: torch.Tensor
    inference_input_ids: torch.Tensor
    labels: torch.Tensor
    prefix_ids: torch.Tensor
    middle_ids: torch.Tensor
    suffix_ids: torch.Tensor

    def inference_view(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inference_input_ids,
            "prefix_ids": self.prefix_ids,
            "suffix_ids": self.suffix_ids,
        }


def make_fim_example(
    token_ids: list[int] | torch.Tensor,
    *,
    seed: int,
    prefix_sentinel: int,
    middle_sentinel: int,
    suffix_sentinel: int,
    minimum_middle_tokens: int = 1,
    ignore_index: int = -100,
) -> FIMExample:
    """Create one reproducible hole while hiding middle tokens from model input."""

    ids = list(token_ids.tolist() if torch.is_tensor(token_ids) else token_ids)
    if len(ids) < minimum_middle_tokens + 2:
        raise ValueError("sequence is too short for prefix, middle, and suffix")
    if len({prefix_sentinel, middle_sentinel, suffix_sentinel}) != 3:
        raise ValueError("FIM sentinels must be distinct")
    rng = random.Random(seed)
    middle_length = rng.randint(minimum_middle_tokens, len(ids) - 2)
    maximum_start = len(ids) - middle_length - 1
    start = rng.randint(1, maximum_start)
    prefix = ids[:start]
    middle = ids[start : start + middle_length]
    suffix = ids[start + middle_length :]
    inference_input = [
        prefix_sentinel,
        *prefix,
        suffix_sentinel,
        *suffix,
        middle_sentinel,
    ]
    model_input = [*inference_input, *middle]
    labels = [ignore_index] * (len(model_input) - len(middle)) + middle
    return FIMExample(
        input_ids=torch.tensor(model_input, dtype=torch.long),
        inference_input_ids=torch.tensor(inference_input, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        prefix_ids=torch.tensor(prefix, dtype=torch.long),
        middle_ids=torch.tensor(middle, dtype=torch.long),
        suffix_ids=torch.tensor(suffix, dtype=torch.long),
    )


@dataclass(frozen=True)
class DiffExample:
    patch_text: str
    original_text: str
    patched_text: str
    changed_line_count: int

    def inference_view(self) -> dict[str, str]:
        return {"original_text": self.original_text}


def make_diff_example(
    original_text: str,
    patched_text: str,
    *,
    context_lines: int = 3,
    filename: str = "target.py",
) -> DiffExample:
    if original_text == patched_text:
        raise ValueError("diff target must change the source")
    if context_lines < 0:
        raise ValueError("context_lines must be non-negative")
    original = original_text.splitlines(keepends=True)
    patched = patched_text.splitlines(keepends=True)
    patch_lines = list(
        difflib.unified_diff(
            original,
            patched,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines,
        )
    )
    changed = sum(
        line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
        for line in patch_lines
    )
    return DiffExample(
        patch_text="".join(patch_lines),
        original_text=original_text,
        patched_text=patched_text,
        changed_line_count=changed,
    )


@dataclass(frozen=True)
class VerifierSignal:
    exit_code: int
    failed_test_count: int
    timed_out: bool
    stdout_tail: str


def encode_verifier_feedback(
    signal: VerifierSignal,
    *,
    buckets: int = 64,
) -> torch.Tensor:
    if buckets <= 1:
        raise ValueError("buckets must be greater than one")
    stdout_hash = hashlib.sha256(signal.stdout_tail.encode()).digest()
    bucket = int.from_bytes(stdout_hash[:4], "big") % buckets
    return torch.tensor(
        [
            signal.exit_code,
            signal.failed_test_count,
            int(signal.timed_out),
            bucket,
        ],
        dtype=torch.long,
    )


class FeedbackBudget:
    """Bound evaluator calls and forbid hidden primary retries."""

    def __init__(self, *, max_attempts: int = 1) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = int(max_attempts)
        self.attempts = 0

    def record_attempt(self) -> None:
        if self.attempts >= self.max_attempts:
            raise RuntimeError("verifier attempt budget exhausted")
        self.attempts += 1

    @property
    def primary_retry_count(self) -> int:
        return max(0, self.attempts - 1)


__all__ = [
    "DiffExample",
    "FIMExample",
    "FeedbackBudget",
    "VerifierSignal",
    "encode_verifier_feedback",
    "make_diff_example",
    "make_fim_example",
]
