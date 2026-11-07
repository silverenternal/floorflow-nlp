"""Exact token-to-parameter audit for nlp_flow models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _unique_trainable(named_parameters) -> list[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    selected = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        selected.append((name, parameter))
    return selected


def count_unique_trainable_parameters(model: torch.nn.Module) -> int:
    """Count unique trainable tensors once, including adapters and heads."""

    return sum(parameter.numel() for _, parameter in _unique_trainable(model.named_parameters()))


def parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    """Report unique trainable parameters by top-level module prefix."""

    groups: dict[str, int] = {}
    for name, parameter in _unique_trainable(model.named_parameters()):
        prefix = name.split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    return dict(sorted(groups.items()))


@dataclass(frozen=True)
class BudgetReport:
    parameters: int
    corpus_tokens: int
    tokens_per_parameter: float
    target_parameters: int
    relative_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.relative_error <= self.tolerance


def audit_parameter_budget(
    parameters: int,
    corpus_tokens: int,
    *,
    tokens_per_parameter: int = 20,
    tolerance: float = 0.02,
) -> BudgetReport:
    if parameters <= 0 or corpus_tokens <= 0:
        raise ValueError("parameters and corpus_tokens must be positive")
    if tokens_per_parameter <= 0:
        raise ValueError("tokens_per_parameter must be positive")
    if not 0.0 <= tolerance <= 0.05:
        raise ValueError("tolerance must be in [0, 0.05]")
    target = corpus_tokens // tokens_per_parameter
    report = BudgetReport(
        parameters=int(parameters),
        corpus_tokens=int(corpus_tokens),
        tokens_per_parameter=float(corpus_tokens / parameters),
        target_parameters=target,
        relative_error=abs(parameters - target) / max(1, target),
        tolerance=float(tolerance),
    )
    if not report.passed:
        raise ValueError(
            f"parameter budget failed: {parameters} parameters vs target {target} "
            f"({report.relative_error:.4%} > {tolerance:.4%})"
        )
    return report


__all__ = [
    "BudgetReport",
    "audit_parameter_budget",
    "count_unique_trainable_parameters",
    "parameter_breakdown",
]
