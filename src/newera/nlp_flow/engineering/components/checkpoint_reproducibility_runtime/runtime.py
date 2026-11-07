"""Thin nlp_flow runtime assembled from shared training controls."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from newera.nlp_flow.engineering.components.acceleration_artifacts_budget.acceleration import RuntimeConfig, configure_torch_runtime
from newera.nlp_flow.engineering.components.checkpoint_reproducibility_runtime.reproducibility import enable_deterministic_execution


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class NLPRuntime:
    device: torch.device
    accelerator: RuntimeConfig
    seed: int
    deterministic: bool

    @property
    def autocast(self):
        return self.accelerator.autocast(self.device)

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self.accelerator.prepare_model(model).to(self.device)

    def dataloader_options(self, workers: int) -> dict[str, object]:
        return self.accelerator.dataloader_options(self.device, workers)

    def manifest(self) -> dict[str, object]:
        return {
            "version": "nlp-runtime-v1",
            "device": str(self.device),
            "seed": self.seed,
            "deterministic": self.deterministic,
            "accelerator": self.accelerator.manifest(),
        }


def build_runtime(
    *,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    compile_model: bool = False,
    seed: int = 19,
    deterministic: bool = False,
    tf32: bool = True,
    triton_kernels: bool = False,
    prefetch_factor: int = 4,
) -> NLPRuntime:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {resolved}")
    accelerator = RuntimeConfig(
        precision=precision,
        compile_model=compile_model,
        prefetch_factor=prefetch_factor,
        tf32=tf32,
        triton_kernels=triton_kernels,
    )
    _seed_everything(int(seed))
    configure_torch_runtime(resolved, tf32=tf32, triton_kernels=triton_kernels)
    if deterministic:
        enable_deterministic_execution()
    return NLPRuntime(
        device=resolved,
        accelerator=accelerator,
        seed=int(seed),
        deterministic=bool(deterministic),
    )


__all__ = ["NLPRuntime", "build_runtime"]
