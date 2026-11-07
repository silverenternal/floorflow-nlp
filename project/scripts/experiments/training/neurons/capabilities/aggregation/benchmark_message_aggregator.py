"""Measure a shared message aggregator's device forward/backward cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
from compare_message_aggregators import SharedAggregationModel, _cost


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    device_name = config["device"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required; CPU fallback is disabled")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU fallback is disabled")
    torch.use_deterministic_algorithms(bool(config.get("deterministic_algorithms", True)))
    torch.manual_seed(int(config["seed"]))
    model = SharedAggregationModel(
        hidden_size=int(config["hidden_size"]),
        aggregator_kind=config["candidate"],
    ).to(device)
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "device": device_name,
        "accelerator_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "Apple MPS"
            if device.type == "mps"
            else platform.processor() or platform.machine()
        ),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_source_sha256": hashlib.sha256(
            Path(__file__).with_name("compare_message_aggregators.py").read_bytes()
        ).hexdigest(),
        "cost": _cost(model, device=device, widths=config["widths"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
