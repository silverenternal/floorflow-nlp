"""Generated-AST-only Stage-B execution probe."""

from __future__ import annotations

import torch
from torch import nn

from newera.nlp_flow.components.ast.flow.runtime.result_cfm import GeneratedASTResultCFM

PROGRAM_NAMES = ("add", "subtract", "multiply", "maximum")


def execute_program(program_id: int, left: int, right: int) -> int:
    if program_id == 0:
        return left + right
    if program_id == 1:
        return left - right
    if program_id == 2:
        return left * right
    if program_id == 3:
        return max(left, right)
    raise ValueError("program id outside the executable probe")


class ResultExecutionHead(nn.Module):
    def __init__(self, hidden_size: int, programs: int = 4) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, programs),
        )

    def forward(self, result_state: torch.Tensor) -> torch.Tensor:
        if result_state.ndim != 3:
            raise ValueError("result_state must be [batch, length, hidden]")
        return self.proj(result_state.mean(dim=1))


class NoResultFlowControl(nn.Module):
    """Prompt-only control with no result flow and no generated AST."""

    def __init__(self, hidden_size: int, programs: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, programs)

    def forward(self, prompt_context: torch.Tensor) -> torch.Tensor:
        return self.proj(prompt_context)


__all__ = [
    "PROGRAM_NAMES",
    "GeneratedASTResultCFM",
    "NoResultFlowControl",
    "ResultExecutionHead",
    "execute_program",
]
