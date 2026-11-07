"""Run a preregistered CPU comparison of stateful and flat neuron models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from newera.capabilities import SequenceSumCapability
from newera.neuron import MLPNeuron, StatefulMLPNeuron

ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH = ROOT / "project/configs/models_training/learning/neuron_comparisons/capabilities/sequence_sum.json"


class StateResetAblation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neuron = StatefulMLPNeuron(1, 8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = None
        for index in range(inputs.shape[1]):
            output = self.neuron.step(inputs[:, index], state=None).value
        if output is None:
            raise ValueError("sequence must not be empty")
        return output


def _data(seed: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.rand(count, 5, generator=generator) * 2.0 - 1.0
    return inputs.unsqueeze(-1), inputs.sum(dim=1, keepdim=True)


def _build(name: str) -> torch.nn.Module:
    if name == "stateful_candidate":
        return SequenceSumCapability(StatefulMLPNeuron(1, 8, 1))
    if name == "parameter_matched_mlp":
        return MLPNeuron(5, 65, 1)
    if name == "compute_matched_mlp":
        return MLPNeuron(5, 333, 1)
    if name == "state_reset_ablation":
        return StateResetAblation()
    raise ValueError(f"unknown comparison model: {name}")


def _predict(model_name: str, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if model_name in {"parameter_matched_mlp", "compute_matched_mlp"}:
        return model(inputs.flatten(start_dim=1)).value  # type: ignore[attr-defined]
    if model_name == "state_reset_ablation":
        return model(inputs)
    return model(inputs)


def _sha256_tensors(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _model_cost(name: str, model: torch.nn.Module) -> tuple[int, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = {
        "stateful_candidate": 2000,
        "parameter_matched_mlp": 390,
        "compute_matched_mlp": 1998,
        "state_reset_ablation": 2000,
    }[name]
    return parameters, macs


def _median_latency_ms(
    name: str,
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> float:
    sample = inputs[:32]
    with torch.no_grad():
        for _ in range(5):
            _predict(name, model, sample)
        timings = []
        for _ in range(31):
            started = time.perf_counter()
            _predict(name, model, sample)
            timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def run(config: dict[str, Any], steps: int | None = None) -> dict[str, Any]:
    torch.set_num_threads(int(config["runs"]["threads"]))
    step_count = int(config["runs"]["steps"] if steps is None else steps)
    if step_count <= 0:
        raise ValueError("steps must be positive")
    names = tuple(config["models"])
    rows = []
    fingerprints = []

    for seed in config["runs"]["seeds"]:
        train_inputs, train_targets = _data(seed + 100, config["task"]["train_examples"])
        validation_inputs, validation_targets = _data(
            seed + 200, config["task"]["validation_examples"]
        )
        fingerprints.append(
            {
                "seed": seed,
                "train_data_sha256": _sha256_tensors(
                    {"inputs": train_inputs, "targets": train_targets}
                ),
                "validation_data_sha256": _sha256_tensors(
                    {"inputs": validation_inputs, "targets": validation_targets}
                ),
            }
        )

        for model_index, name in enumerate(names):
            torch.manual_seed(int(seed) * int(config["runs"]["initialization_seed_offset"]) + model_index)
            model = _build(name)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(config["runs"]["learning_rate"])
            )
            for _ in range(step_count):
                optimizer.zero_grad(set_to_none=True)
                prediction = _predict(name, model, train_inputs)
                loss = torch.nn.functional.mse_loss(prediction, train_targets)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_predictions = _predict(name, model, train_inputs)
                validation_predictions = _predict(name, model, validation_inputs)
                train_mse = torch.nn.functional.mse_loss(
                    train_predictions, train_targets
                ).item()
                validation_mse = torch.nn.functional.mse_loss(
                    validation_predictions, validation_targets
                ).item()
                validation_mae = torch.nn.functional.l1_loss(
                    validation_predictions, validation_targets
                ).item()
            parameters, macs = _model_cost(name, model)
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    "optimizer_steps": step_count,
                    "training_examples": int(config["task"]["train_examples"]),
                    "validation_examples": int(config["task"]["validation_examples"]),
                    "parameters": parameters,
                    "forward_linear_macs_per_example": macs,
                    "training_forward_linear_mac_proxy": macs
                    * int(config["task"]["train_examples"])
                    * step_count,
                    "train_mse": train_mse,
                    "validation_mse": validation_mse,
                    "validation_mae": validation_mae,
                    "median_cpu_inference_ms_per_32_examples": _median_latency_ms(
                        name, model, validation_inputs
                    ),
                    "model_sha256": _sha256_tensors(model.state_dict()),
                }
            )

    summaries = {}
    for name in names:
        model_rows = [row for row in rows if row["model"] == name]
        summaries[name] = {
            "parameters": model_rows[0]["parameters"],
            "forward_linear_macs_per_example": model_rows[0]["forward_linear_macs_per_example"],
            "training_forward_linear_mac_proxy": model_rows[0]["training_forward_linear_mac_proxy"],
            "validation_mse_mean": statistics.mean(row["validation_mse"] for row in model_rows),
            "validation_mse_stdev": statistics.stdev(row["validation_mse"] for row in model_rows),
            "validation_mae_mean": statistics.mean(row["validation_mae"] for row in model_rows),
            "median_cpu_inference_ms_per_32_examples_mean": statistics.mean(
                row["median_cpu_inference_ms_per_32_examples"] for row in model_rows
            ),
        }

    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return {
        "schema_version": "newera-neuron-comparison-result-v1",
        "run_id": config["run_id"],
        "started_at_local_date": "2026-11-06",
        "code_revision": git_revision,
        "working_tree_dirty": dirty,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "config": config,
        "actual_steps_per_model": step_count,
        "data_fingerprints": fingerprints,
        "summaries": summaries,
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.steps), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
