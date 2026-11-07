"""Train and evaluate one shared ModelNeuron on the first functional tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))

from newera.capabilities.evaluation.functional_tasks import (
    make_conditional_route,
    make_grouped_binding,
    make_interference_recall,
    make_set_sum,
)
from newera.capabilities.evaluation.runner import train_with_accumulation
from newera.neuron.composition import ModelNeuron


class SharedFunctionalNeuron(nn.Module):
    """Apply one shared cell over task elements; caller owns each sample state."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.cell = ModelNeuron(input_size=3, hidden_size=hidden_size, message_size=1)

    def encode(self, family: str, sample: Any) -> Tensor:
        if family in {"interference_recall", "conditional_routing"}:
            query = sample.query_key if family == "interference_recall" else sample.selector
            return torch.stack((sample.values, sample.keys.float(), query.expand_as(sample.values).float()), -1)
        elif family == "set_sum":
            rows = torch.stack((sample.values, torch.zeros_like(sample.values), torch.zeros_like(sample.values)), -1)
        elif family == "grouped_binding":
            scale = max(1, int(sample.groups.max().item()))
            rows = torch.stack((sample.values, sample.roles.float(), sample.groups.float() / scale), -1)
        else:
            raise ValueError(f"unsupported family: {family}")
        return rows

    def process_rows(
        self, rows: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        output = None
        for row in rows:
            result = self.cell.step(
                row.reshape(1, 3),
                incoming_messages=row.new_zeros((1, 0, 1)),
                incoming_strengths=row.new_zeros((1, 0)),
                incoming_edge_types=torch.empty((1, 0), dtype=torch.long, device=row.device),
                state=state,
            )
            output, state = result.value, result.state
        assert output is not None
        assert state is not None
        return output.reshape(()), state

    def forward(self, family: str, sample: Any) -> Tensor:
        return self.process_rows(self.encode(family, sample))[0]


def _sample(family: str, seed: int, size: int) -> Any:
    if family == "set_sum":
        return make_set_sum(size=size, seed=seed)
    if family == "grouped_binding":
        return make_grouped_binding(group_count=size, seed=seed)
    if family == "conditional_routing":
        return make_conditional_route(candidate_count=size, seed=seed)
    return make_interference_recall(length=size, key_count=4, seed=seed)


def _evaluate(
    model: SharedFunctionalNeuron,
    family: str,
    sizes: list[int],
    count: int,
    seed: int,
    device: torch.device,
    train_target_mean: float,
) -> dict[str, Any]:
    errors: list[float] = []
    target_magnitudes: list[float] = []
    permutation_deltas: list[float] = []
    zero_errors: list[float] = []
    train_mean_errors: list[float] = []
    replay_deltas: list[float] = []
    for index in range(count):
        sample = _sample(family, seed + index, sizes[index % len(sizes)])
        sample = type(sample)(**{key: value.to(device) if isinstance(value, Tensor) else value for key, value in sample.__dict__.items()})
        prediction = model(family, sample)
        errors.append(abs(float(prediction.item() - sample.target.item())))
        target_magnitudes.append(abs(float(sample.target.item())))
        zero_errors.append(abs(float(sample.target.item())))
        train_mean_errors.append(abs(float(sample.target.item() - train_target_mean)))
        if family in {"set_sum", "grouped_binding", "conditional_routing"}:
            generator = torch.Generator().manual_seed(seed + index + 50_000_000)
            order = torch.randperm(sample.values.numel(), generator=generator).to(device)
            if family == "set_sum":
                permuted = replace(sample, values=sample.values[order])
            elif family == "grouped_binding":
                permuted = replace(
                    sample,
                    groups=sample.groups[order],
                    roles=sample.roles[order],
                    values=sample.values[order],
                )
            else:
                permuted = replace(sample, keys=sample.keys[order], values=sample.values[order])
            permutation_deltas.append(abs(float(model(family, permuted).item() - prediction.item())))
        if family in {"interference_recall", "conditional_routing"}:
            rows = model.encode(family, sample)
            split = max(1, rows.shape[0] // 2)
            _, saved_state = model.process_rows(rows[:split])
            replayed, _ = model.process_rows(rows[split:], saved_state)
            replay_deltas.append(abs(float(replayed.item() - prediction.item())))
    mae = sum(errors) / len(errors)
    mean_target_magnitude = sum(target_magnitudes) / len(target_magnitudes)
    return {
        "sample_count": count,
        "mae": mae,
        "normalized_mae": mae / mean_target_magnitude if mean_target_magnitude > 1e-12 else None,
        "zero_baseline_mae": sum(zero_errors) / len(zero_errors),
        "train_mean_baseline_mae": sum(train_mean_errors) / len(train_mean_errors),
        "oracle_mae": 0.0,
        "permutation_max_abs_delta": max(permutation_deltas) if permutation_deltas else None,
        "chance_exact_accuracy": sum(1 / sizes[index % len(sizes)] for index in range(count)) / count
        if family == "conditional_routing"
        else (0.25 if family == "interference_recall" else None),
        "zero_baseline_exact_accuracy": sum(error <= 0.05 for error in zero_errors) / len(zero_errors)
        if family in {"interference_recall", "conditional_routing"}
        else None,
        "state_replay_max_abs_delta": max(replay_deltas) if replay_deltas else None,
        "exact_recall_accuracy": sum(error <= 0.05 for error in errors) / len(errors)
        if family in {"interference_recall", "conditional_routing"}
        else None,
        "max_absolute_error": max(errors),
    }


def run(
    output: Path,
    *,
    config_path: Path,
    steps_per_family: int | None,
    eval_samples: int | None,
    device_name: str,
) -> dict[str, Any]:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required by this run but is unavailable; fallback is disabled")
    torch.manual_seed(23000)
    device = torch.device(device_name)
    config = json.loads(config_path.read_text())
    model = SharedFunctionalNeuron().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    families = {
        "set_sum": [4, 16, 64],
        "grouped_binding": [2, 4, 8],
        "interference_recall": [8, 16, 32],
        "conditional_routing": [2, 4, 8],
    }
    split_seeds = {name: item["seed"] for name, item in config["data"]["splits"].items()}
    source_files = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "src/newera/capabilities/evaluation/functional_tasks.py",
        ROOT / "src/newera/neuron/composition.py",
    ]
    results: dict[str, Any] = {
        "device": str(device),
        "seed": 23000,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "architecture": "one shared ModelNeuron(input=3, hidden=32, message=1)",
        "config_path": str(config_path.relative_to(ROOT)),
        "split_seeds": split_seeds,
        "sealed_test_used": False,
    }
    for family_index, (family, sizes) in enumerate(families.items()):
        task_config = config["tasks"][family]
        train_count = steps_per_family or task_config["train_samples"]
        validation_count = eval_samples or task_config["validation_samples"]
        results[family] = {
            "training_sample_count": train_count,
            "validation_sample_count": validation_count,
            "sealed_test_sample_count": task_config["sealed_test_samples"],
        }
        train_samples = [
            _sample(family, split_seeds["train"] + family_index * 100_000 + i, sizes[i % len(sizes)])
            for i in range(train_count)
        ]
        train_samples = [type(sample)(**{key: value.to(device) if isinstance(value, Tensor) else value for key, value in sample.__dict__.items()}) for sample in train_samples]

        def loss_for(module: nn.Module, sample: Any, _: int, task_family: str = family) -> Tensor:
            assert isinstance(module, SharedFunctionalNeuron)
            prediction = module(task_family, sample)
            return torch.nn.functional.mse_loss(prediction, sample.target)

        summary = train_with_accumulation(
            model,
            train_samples,
            optimizer=optimizer,
            loss_for=loss_for,
            accumulation_steps=8,
        )
        model.eval()
        with torch.no_grad():
            validation = _evaluate(
                model,
                family,
                sizes,
                validation_count,
                split_seeds["validation"] + family_index * 100_000,
                device,
                sum(float(sample.target.item()) for sample in train_samples) / len(train_samples),
            )
            held_out_size = (
                256 if family == "set_sum" else
                16 if family in {"grouped_binding", "conditional_routing"} else 64
            )
            extrapolation = _evaluate(
                model,
                family,
                [held_out_size],
                validation_count,
                split_seeds["size_extrapolation_validation"] + family_index * 100_000,
                device,
                sum(float(sample.target.item()) for sample in train_samples) / len(train_samples),
            )
        model.train()
        results[family].update({
            "train": {"samples": summary.samples_seen, "mean_loss": summary.mean_loss, "optimizer_steps": summary.optimizer_steps},
            "validation": validation,
            "size_extrapolation_diagnostic": extrapolation,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/learning/neuron_comparisons/functional/functional_capability_suite.json",
    )
    parser.add_argument("--steps-per-family", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    run(
        args.output,
        config_path=args.config,
        steps_per_family=args.steps_per_family,
        eval_samples=args.eval_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
