"""Runtime controls shared by training and evaluation entry points."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cache

import torch

try:
    import triton
except ImportError:  # optional acceleration
    triton = None

def triton_status() -> dict[str, bool | str]:
    return {
        "available": triton is not None,
        "requested": os.environ.get("FLOORFLOW_TRITON_KERNELS", "0") == "1",
        "version": getattr(triton, "__version__", "unavailable"),
    }

PRECISIONS = ("fp32", "fp16", "bf16")


@cache
def _device_capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def _autocast_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"precision {precision!r} does not use autocast")


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated execution settings with conservative CPU fallbacks."""

    precision: str = "fp32"
    compile_model: bool = False
    compile_mode: str = "default"
    channels_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    tf32: bool = True
    triton_kernels: bool = False

    @classmethod
    def from_args(cls, args) -> RuntimeConfig:
        return cls(
            precision=args.precision,
            compile_model=args.compile_model,
            compile_mode=args.compile_mode,
            channels_last=args.channels_last,
            prefetch_factor=args.prefetch_factor,
            tf32=getattr(args, "tf32", True),
            triton_kernels=getattr(args, "triton_kernels", False),
        )

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"unsupported compile mode: {self.compile_mode}")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")

    def autocast(self, device: torch.device):
        if device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        if self.precision == "bf16":
            index = (
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            )
            major, _minor = _device_capability(index)
            if major < 8:
                return nullcontext()
        return torch.autocast(
            device_type=device.type,
            dtype=_autocast_dtype(self.precision),
        )

    @property
    def scaler_enabled(self) -> bool:
        return self.precision == "fp16"

    def make_scaler(self, device: torch.device) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            device.type,
            enabled=self.scaler_enabled and device.type == "cuda",
        )

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if self.compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable")
            functorch_config = getattr(
                getattr(torch, "_functorch", None),
                "config",
                None,
            )
            if functorch_config is not None and hasattr(
                functorch_config,
                "donated_buffer",
            ):
                functorch_config.donated_buffer = False
            inductor_config = getattr(
                getattr(torch, "_inductor", None),
                "config",
                None,
            )
            if inductor_config is not None and hasattr(
                inductor_config,
                "fx_graph_cache",
            ):
                inductor_config.fx_graph_cache = True
            model.compile(mode=self.compile_mode)
        return model

    @contextmanager
    def eager_model(self, model: torch.nn.Module):
        """Temporarily bypass a top-level compiled call for exact evaluation."""

        compiled = getattr(model, "_compiled_call_impl", None)
        if not self.compile_model or compiled is None:
            yield model
            return
        model._compiled_call_impl = None
        try:
            yield model
        finally:
            model._compiled_call_impl = compiled

    def dataloader_options(self, device: torch.device, workers: int) -> dict:
        options = {
            "num_workers": max(0, int(workers)),
            "pin_memory": device.type == "cuda",
        }
        if options["num_workers"] > 0:
            options["persistent_workers"] = bool(self.persistent_workers)
            options["prefetch_factor"] = int(self.prefetch_factor)
        return options

    def manifest(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "compile_model": self.compile_model,
            "compile_mode": self.compile_mode,
            "channels_last": self.channels_last,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "tf32": self.tf32,
            "triton_kernels": self.triton_kernels,
            "triton_runtime": triton_status(),
        }


def configure_torch_runtime(
    device: torch.device,
    *,
    tf32: bool = True,
    triton_kernels: bool = False,
) -> None:
    """Enable high-throughput kernels without changing numerical objectives."""

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    os.environ["FLOORFLOW_TRITON_KERNELS"] = (
        "1" if triton_kernels else "0"
    )
