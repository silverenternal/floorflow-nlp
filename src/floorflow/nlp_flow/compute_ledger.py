"""Parameter-residency and executed-work accounting for NLP controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


def unique_resident_parameter_elements(module: nn.Module) -> int:
    """Count backing storage once, including tied parameters and generated bases."""
    storages: dict[tuple[str, int], int] = {}
    for parameter in module.parameters():
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr())
        elements = storage.nbytes() // parameter.element_size()
        storages[key] = max(storages.get(key, 0), elements)
    return sum(storages.values())


@dataclass(frozen=True)
class ComputeLedger:
    resident_parameter_elements: int
    generated_basis_parameter_elements: int
    materialized_expert_parameter_elements: int
    selected_cell_token_calls: int
    dense_cell_token_calls: int
    router_flops: int
    attention_flops: int
    generated_apply_flops_estimate: int
    serial_depth: int
    lane_calls: int
    nfe: int
    latency_ms: float | None
    peak_memory_bytes: int | None
    control_id: str

    def __post_init__(self) -> None:
        integer_fields = (
            self.resident_parameter_elements,
            self.generated_basis_parameter_elements,
            self.materialized_expert_parameter_elements,
            self.selected_cell_token_calls,
            self.dense_cell_token_calls,
            self.router_flops,
            self.attention_flops,
            self.generated_apply_flops_estimate,
            self.serial_depth,
            self.lane_calls,
            self.nfe,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("compute ledger counters cannot be negative")
        if self.selected_cell_token_calls > self.dense_cell_token_calls:
            raise ValueError("selected cell calls exceed their dense reference work")
        if not self.control_id:
            raise ValueError("compute ledger needs a control identity")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency cannot be negative")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak memory cannot be negative")

    @property
    def total_accounted_flops(self) -> int:
        return self.router_flops + self.attention_flops + self.generated_apply_flops_estimate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_compute_ledger(
    model: nn.Module,
    diagnostics: dict[str, Any],
    *,
    control_id: str,
    router_flops: int,
    attention_flops: int,
    serial_depth: int,
    lane_calls: int,
    nfe: int,
    latency_ms: float | None = None,
    peak_memory_bytes: int | None = None,
) -> ComputeLedger:
    required = ("competence_token_calls", "dense_competence_token_calls")
    if any(name not in diagnostics for name in required):
        raise ValueError("runtime diagnostics omit executed and dense-reference cell calls")

    def scalar(name: str) -> int:
        value = diagnostics.get(name, 0)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"runtime diagnostic {name} must be scalar")
            value = value.detach().item()
        integer = int(value)
        if integer != value:
            raise ValueError(f"runtime diagnostic {name} must be an integer")
        return integer

    generated_basis = 0
    for name, parameter in model.named_parameters():
        if name.endswith(("left_basis", "right_basis")):
            generated_basis += parameter.numel()
    return ComputeLedger(
        resident_parameter_elements=unique_resident_parameter_elements(model),
        generated_basis_parameter_elements=generated_basis,
        materialized_expert_parameter_elements=scalar("generated_weight_elements"),
        selected_cell_token_calls=scalar("competence_token_calls"),
        dense_cell_token_calls=scalar("dense_competence_token_calls"),
        router_flops=int(router_flops),
        attention_flops=int(attention_flops),
        generated_apply_flops_estimate=scalar("generated_apply_flops_proxy"),
        serial_depth=int(serial_depth),
        lane_calls=int(lane_calls),
        nfe=int(nfe),
        latency_ms=latency_ms,
        peak_memory_bytes=peak_memory_bytes,
        control_id=control_id,
    )


__all__ = ["ComputeLedger", "build_compute_ledger", "unique_resident_parameter_elements"]
