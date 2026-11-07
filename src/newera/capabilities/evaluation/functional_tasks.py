"""Deterministic synthetic tasks and independent reference functions.

These generators define data only; task-answer logic stays in the oracle and is
never called by a model adapter at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SetSumSample:
    values: Tensor
    target: Tensor


@dataclass(frozen=True)
class GroupedBindingSample:
    groups: Tensor
    roles: Tensor
    values: Tensor
    target: Tensor


@dataclass(frozen=True)
class InterferenceRecallSample:
    keys: Tensor
    values: Tensor
    query_key: Tensor
    target: Tensor


@dataclass(frozen=True)
class ConditionalRouteSample:
    keys: Tensor
    values: Tensor
    selector: Tensor
    target: Tensor


@dataclass(frozen=True)
class RuleCompositionSample:
    first_rule: Tensor
    second_rule: Tensor
    value: Tensor
    target: Tensor


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def make_set_sum(*, size: int, seed: int) -> SetSumSample:
    if size <= 0:
        raise ValueError("size must be positive")
    values = torch.randn(size, generator=_generator(seed))
    return SetSumSample(values=values, target=set_sum_oracle(values))


def set_sum_oracle(values: Tensor) -> Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty vector")
    return values.sum()


def make_grouped_binding(*, group_count: int, seed: int) -> GroupedBindingSample:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    generator = _generator(seed)
    groups = torch.arange(group_count).repeat_interleave(2)
    roles = torch.tensor([0, 1], dtype=torch.long).repeat(group_count)
    values = torch.randn(group_count * 2, generator=generator)
    order = torch.randperm(group_count * 2, generator=generator)
    groups, roles, values = groups[order], roles[order], values[order]
    return GroupedBindingSample(
        groups=groups,
        roles=roles,
        values=values,
        target=grouped_binding_oracle(groups, roles, values),
    )


def grouped_binding_oracle(groups: Tensor, roles: Tensor, values: Tensor) -> Tensor:
    if groups.ndim != 1 or roles.shape != groups.shape or values.shape != groups.shape:
        raise ValueError("groups, roles and values must be aligned vectors")
    result = values.new_zeros(())
    for group in groups.unique(sorted=True):
        a = values[(groups == group) & (roles == 0)]
        b = values[(groups == group) & (roles == 1)]
        if a.numel() != 1 or b.numel() != 1:
            raise ValueError("each group must contain exactly one item for each role")
        result = result + a[0] * b[0]
    if not torch.all((roles == 0) | (roles == 1)):
        raise ValueError("roles must be 0 or 1")
    return result


def make_interference_recall(
    *, length: int, key_count: int, seed: int
) -> InterferenceRecallSample:
    if length <= 0 or key_count <= 0:
        raise ValueError("length and key_count must be positive")
    generator = _generator(seed)
    keys = torch.randint(key_count, (length,), generator=generator)
    values = torch.randn(length, generator=generator)
    query_key = torch.randint(key_count, (), generator=generator)
    # Guarantee a target exists without exposing its location/value to the adapter.
    if not torch.any(keys == query_key):
        keys[-1] = query_key
    return InterferenceRecallSample(
        keys=keys,
        values=values,
        query_key=query_key,
        target=interference_recall_oracle(keys, values, query_key),
    )


def interference_recall_oracle(keys: Tensor, values: Tensor, query_key: Tensor) -> Tensor:
    if keys.ndim != 1 or values.shape != keys.shape or keys.numel() == 0:
        raise ValueError("keys and values must be aligned non-empty vectors")
    matches = torch.where(keys == query_key)[0]
    if matches.numel() == 0:
        raise ValueError("query key must occur in the event sequence")
    return values[matches[-1]]


def make_conditional_route(*, candidate_count: int, seed: int) -> ConditionalRouteSample:
    """Create keyed candidate memory and a visible key query, then shuffle rows."""
    if candidate_count <= 1:
        raise ValueError("candidate_count must be greater than one")
    generator = _generator(seed)
    keys = torch.randperm(candidate_count, generator=generator)
    values = torch.randn(candidate_count, generator=generator)
    selector = torch.randint(candidate_count, (), generator=generator)
    order = torch.randperm(candidate_count, generator=generator)
    keys, values = keys[order], values[order]
    return ConditionalRouteSample(
        keys=keys,
        values=values,
        selector=selector,
        target=interference_recall_oracle(keys, values, selector),
    )


def make_rule_composition(*, domain_size: int, seed: int) -> RuleCompositionSample:
    if domain_size <= 1:
        raise ValueError("domain_size must be greater than one")
    generator = _generator(seed)
    first_rule = torch.randperm(domain_size, generator=generator)
    second_rule = torch.randperm(domain_size, generator=generator)
    value = torch.randint(domain_size, (), generator=generator)
    return RuleCompositionSample(
        first_rule=first_rule,
        second_rule=second_rule,
        value=value,
        target=rule_composition_oracle(first_rule, second_rule, value),
    )


def rule_composition_oracle(first_rule: Tensor, second_rule: Tensor, value: Tensor) -> Tensor:
    if first_rule.ndim != 1 or second_rule.shape != first_rule.shape or first_rule.numel() <= 1:
        raise ValueError("rules must be aligned non-empty lookup vectors")
    domain = torch.arange(first_rule.numel(), device=first_rule.device)
    if not torch.equal(first_rule.sort().values, domain) or not torch.equal(second_rule.sort().values, domain):
        raise ValueError("rules must be permutations of the finite domain")
    if value.numel() != 1 or int(value.item()) < 0 or int(value.item()) >= first_rule.numel():
        raise ValueError("value must be one index within the rule domain")
    return second_rule[first_rule[value.long()]]


__all__ = [
    "ConditionalRouteSample",
    "GroupedBindingSample",
    "InterferenceRecallSample",
    "RuleCompositionSample",
    "SetSumSample",
    "grouped_binding_oracle",
    "interference_recall_oracle",
    "make_conditional_route",
    "make_grouped_binding",
    "make_interference_recall",
    "make_rule_composition",
    "make_set_sum",
    "rule_composition_oracle",
    "set_sum_oracle",
]
