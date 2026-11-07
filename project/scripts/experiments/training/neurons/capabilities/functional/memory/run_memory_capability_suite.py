"""Measure memory and temporal-order behavior of the current shared GRU cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[8]
sys.path.insert(0, str(ROOT / "src"))

from newera.capabilities.evaluation.memory_tasks import (
    MemorySequenceSample,
    make_delayed_recall,
    make_interference_recovery,
    make_order_recognition,
)
from newera.neuron.composition import ModelNeuron

FAMILIES = ("delayed_recall", "order_recognition", "interference_recovery")


def _make(family: str, length: int, seed: int) -> MemorySequenceSample:
    factories = {
        "delayed_recall": make_delayed_recall,
        "order_recognition": make_order_recognition,
        "interference_recovery": make_interference_recovery,
    }
    return factories[family](length=length, seed=seed)


class StatefulMemoryModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        cell_kind: str = "gru",
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.cell_kind = cell_kind
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        if cell_kind in {"gru", "gru_multi_output"}:
            self.neuron = ModelNeuron(
                input_size=8,
                hidden_size=hidden_size,
                message_size=output_channels,
            )
        elif cell_kind == "lstm":
            self.state_update = torch.nn.LSTMCell(input_size=8, hidden_size=75)
            self.readout = torch.nn.Linear(75, 1)
            self.hidden_size = 75
        else:
            raise ValueError(f"unsupported recurrent cell: {cell_kind}")

    def run(
        self,
        events: Tensor,
        *,
        state: Tensor | None = None,
        reset_each_step: bool = False,
        drop_at: Tensor | None = None,
        start_step: int = 0,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        batch_size = events.shape[0]
        if self.cell_kind == "lstm":
            return self._run_lstm(
                events,
                state=state,
                reset_each_step=reset_each_step,
                drop_at=drop_at,
                start_step=start_step,
            )
        empty_messages = events.new_empty((batch_size, 0, self.output_channels))
        empty_strengths = events.new_empty((batch_size, 0))
        empty_types = torch.empty((batch_size, 0), dtype=torch.long, device=events.device)
        output: Tensor | None = None
        for offset in range(events.shape[1]):
            step_index = start_step + offset
            if reset_each_step:
                state = None
            elif drop_at is not None:
                drop_rows = drop_at == step_index
                if torch.any(drop_rows):
                    if state is None:
                        state = events.new_zeros((batch_size, self.neuron.hidden_size))
                    state = state.masked_fill(drop_rows[:, None], 0.0)
            result = self.neuron.step(
                events[:, offset],
                incoming_messages=empty_messages,
                incoming_strengths=empty_strengths,
                incoming_edge_types=empty_types,
                state=state,
            )
            output, state = result.value[:, 0], result.state
        assert output is not None and state is not None
        if self.output_channels > 1:
            return result.value, state
        return output, state

    def _run_lstm(
        self,
        events: Tensor,
        *,
        state: Tensor | tuple[Tensor, Tensor] | None,
        reset_each_step: bool,
        drop_at: Tensor | None,
        start_step: int,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size = events.shape[0]
        output: Tensor | None = None
        if state is None:
            hidden = events.new_zeros((batch_size, self.hidden_size))
            cell = events.new_zeros((batch_size, self.hidden_size))
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError("LSTM caller state must be a (hidden, cell) tuple")
            hidden, cell = state
        for offset in range(events.shape[1]):
            step_index = start_step + offset
            if reset_each_step:
                hidden = events.new_zeros((batch_size, self.hidden_size))
                cell = events.new_zeros((batch_size, self.hidden_size))
            elif drop_at is not None:
                drop_rows = drop_at == step_index
                hidden = hidden.masked_fill(drop_rows[:, None], 0.0)
                cell = cell.masked_fill(drop_rows[:, None], 0.0)
            hidden, cell = self.state_update(events[:, offset], (hidden, cell))
            output = self.readout(hidden)[:, 0]
        assert output is not None
        return output, (hidden, cell)

    def forward(self, events: Tensor) -> Tensor:
        return self.run(events)[0]

    def decode(self, outputs: Tensor, family: str) -> Tensor:
        if outputs.ndim == 1:
            return outputs
        channels = {
            "delayed_recall": 0,
            "order_recognition": 1,
            "interference_recovery": 2,
        }
        return outputs[:, channels[family]]

    def forward_for_families(self, events: Tensor, families: list[str]) -> Tensor:
        outputs = self.run(events)[0]
        if outputs.ndim == 1:
            return outputs
        channels = torch.tensor(
            [
                {"delayed_recall": 0, "order_recognition": 1, "interference_recovery": 2}[family]
                for family in families
            ],
            dtype=torch.long,
            device=events.device,
        )
        return outputs.gather(1, channels[:, None]).squeeze(1)


def _dataset(family: str, seed: int, lengths: list[int], count: int) -> list[MemorySequenceSample]:
    chooser = random.Random(seed)
    return [
        _make(family, lengths[chooser.randrange(len(lengths))], seed + index * 31 + 1)
        for index in range(count)
    ]


def _buckets(samples: list[MemorySequenceSample]) -> dict[int, list[MemorySequenceSample]]:
    buckets: dict[int, list[MemorySequenceSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.events.shape[0]].append(sample)
    return buckets


def _clone_state(state: Tensor | tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
    if isinstance(state, tuple):
        return state[0].clone(), state[1].clone()
    return state.clone()


def _train(
    model: StatefulMemoryModel,
    samples: list[MemorySequenceSample],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    grouped = _buckets(samples)
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for _epoch in range(epochs):
        length_keys = list(grouped)
        order = torch.randperm(len(length_keys), generator=rng).tolist()
        for key_index in order:
            rows = grouped[length_keys[key_index]]
            row_order = torch.randperm(len(rows), generator=rng).tolist()
            for start in range(0, len(row_order), batch_size):
                selected = [rows[i] for i in row_order[start : start + batch_size]]
                events = torch.stack([sample.events for sample in selected]).to(device)
                targets = torch.stack([sample.target for sample in selected]).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_for_families(events, [sample.family for sample in selected])
                loss = torch.nn.functional.mse_loss(prediction, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("memory-task loss became non-finite")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
                optimizer_steps += 1
    if device.type == "mps":
        torch.mps.synchronize()
    return {
        "epochs": epochs,
        "samples_per_epoch": len(samples),
        "optimizer_steps": optimizer_steps,
        "training_seconds": time.perf_counter() - started,
        "mean_batch_mse": statistics.mean(losses),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _evaluate(
    model: StatefulMemoryModel,
    samples: list[MemorySequenceSample],
    *,
    device: torch.device,
    seed: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    model.eval()
    grouped = _buckets(samples)
    errors: list[float] = []
    predictions: list[float] = []
    targets: list[float] = []
    reset_predictions: list[float] = []
    dropped_predictions: list[float] = []
    reset_errors: list[float] = []
    dropped_errors: list[float] = []
    replay_deltas: list[float] = []
    checkpoint_deltas: list[float] = []
    with torch.no_grad():
        sample_index = 0
        for length, rows in sorted(grouped.items()):
            for start in range(0, len(rows), batch_size):
                selected = rows[start : start + batch_size]
                events = torch.stack([sample.events for sample in selected]).to(device)
                labels = torch.stack([sample.target for sample in selected]).to(device)
                family = selected[0].family
                prediction_raw, _final_state = model.run(events)
                reset_raw, _ = model.run(events, reset_each_step=True)
                prediction = model.decode(prediction_raw, family)
                reset_prediction = model.decode(reset_raw, family)
                drops = torch.tensor(
                    [1 + ((seed + sample_index + row * 19) % (length - 2)) for row in range(len(selected))],
                    dtype=torch.long,
                    device=device,
                )
                dropped_raw, _ = model.run(events, drop_at=drops)
                dropped_prediction = model.decode(dropped_raw, family)
                midpoint = length // 2
                first_state = model.run(events[:, :midpoint])[1]
                replay_raw, _ = model.run(
                    events[:, midpoint:], state=first_state, start_step=midpoint
                )
                replay_prediction = model.decode(replay_raw, family)
                replay_deltas.extend((prediction - replay_prediction).abs().cpu().tolist())

                # Model weights and caller-owned recurrent state are restored separately.
                clone = StatefulMemoryModel(
                    model.hidden_size,
                    cell_kind=model.cell_kind,
                    output_channels=model.output_channels,
                ).to(device)
                clone.load_state_dict(model.state_dict())
                restored_state = _clone_state(first_state)
                resumed_raw, _ = clone.run(
                    events[:, midpoint:], state=restored_state, start_step=midpoint
                )
                resumed_prediction = clone.decode(resumed_raw, family)
                checkpoint_deltas.extend((prediction - resumed_prediction).abs().cpu().tolist())

                errors.extend((prediction - labels).abs().cpu().tolist())
                predictions.extend(prediction.cpu().tolist())
                targets.extend(labels.cpu().tolist())
                reset_predictions.extend(reset_prediction.cpu().tolist())
                dropped_predictions.extend(dropped_prediction.cpu().tolist())
                reset_errors.extend((reset_prediction - labels).abs().cpu().tolist())
                dropped_errors.extend((dropped_prediction - labels).abs().cpu().tolist())
                sample_index += len(selected)
    mae = statistics.mean(errors)
    family = samples[0].family
    result: dict[str, Any] = {
        "sample_count": len(samples),
        "mae": mae,
        "mse": statistics.mean(error * error for error in errors),
        "max_absolute_error": max(errors),
        "reset_state_mae": statistics.mean(reset_errors),
        "random_state_drop_mae": statistics.mean(dropped_errors),
        "state_replay_max_abs_delta": max(replay_deltas, default=0.0),
        "checkpoint_restore_max_abs_delta": max(checkpoint_deltas, default=0.0),
    }
    if family == "order_recognition":
        result["exact_accuracy"] = statistics.mean(
            int((prediction >= 0.5) == (target >= 0.5))
            for prediction, target in zip(predictions, targets, strict=True)
        )
        result["reset_state_accuracy"] = statistics.mean(
            int((prediction >= 0.5) == (target >= 0.5))
            for prediction, target in zip(reset_predictions, targets, strict=True)
        )
        result["random_state_drop_accuracy"] = statistics.mean(
            int((prediction >= 0.5) == (target >= 0.5))
            for prediction, target in zip(dropped_predictions, targets, strict=True)
        )
    return result


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    device_name = config["device"]
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this run; fallback is disabled")
    device = torch.device(device_name)
    data = config["data"]
    results: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "device": device_name,
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_weight_contract": "one recurrent cell weight set shared over all time steps and three tasks",
        "state_contract": "caller-owned state is per sequence instance; model weights and event state are reported separately",
        "sealed_test_used": True,
        "seeds": [],
    }
    checkpoint_path = output.parent / "shared_model_state.pt"
    results["shared_checkpoint"] = None
    for seed_index in config["training"]["seeds"]:
        seed = data["seed_base"] + seed_index * 10_000
        train = [
            sample
            for family_index, family in enumerate(FAMILIES)
            for sample in _dataset(
                family,
                seed + family_index * 100_000,
                data["train_lengths"],
                data["train_samples_per_family"],
            )
        ]
        validation = {
            family: _dataset(
                family,
                seed + 200_000 + family_index * 100_000,
                data["train_lengths"],
                data["validation_samples_per_family"],
            )
            for family_index, family in enumerate(FAMILIES)
        }
        test = {
            family: _dataset(
                family,
                seed + 300_000 + family_index * 100_000,
                data["train_lengths"],
                data["sealed_test_samples_per_family"],
            )
            for family_index, family in enumerate(FAMILIES)
        }
        extrapolation = {
            family: _dataset(
                family,
                seed + 400_000 + family_index * 100_000,
                data["extrapolation_lengths"],
                data["extrapolation_samples_per_family"],
            )
            for family_index, family in enumerate(FAMILIES)
        }
        row: dict[str, Any] = {"seed": seed, "candidates": {}}
        for candidate in config["training"].get("candidates", ["gru"]):
            torch.manual_seed(seed)
            model = StatefulMemoryModel(
                config["model"]["hidden_sizes"][candidate],
                cell_kind=candidate,
                output_channels=3 if candidate == "gru_multi_output" else 1,
            ).to(device)
            training = _train(
                model,
                train,
                seed=seed,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                device=device,
            )
            if candidate == "gru_multi_output" and seed_index == config["training"]["seeds"][0]:
                checkpoint = {
                    "schema": "newera-shared-memory-model-state-v1",
                    "candidate": candidate,
                    "seed": seed,
                    "model": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "parameter_count": training["parameter_count"],
                    "output_channels": {
                        "delayed_recall": 0,
                        "order_recognition": 1,
                        "interference_recovery": 2,
                    },
                }
                torch.save(checkpoint, checkpoint_path)
                results["shared_checkpoint"] = {
                    "path": str(checkpoint_path.resolve().relative_to(ROOT)),
                    "seed": seed,
                    "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "parameter_count": training["parameter_count"],
                }
            candidate_result: dict[str, Any] = {"training": training, "families": {}}
            for family in FAMILIES:
                candidate_result["families"][family] = {
                    "validation": _evaluate(
                        model,
                        validation[family],
                        device=device,
                        seed=seed + 1_000 + FAMILIES.index(family),
                    ),
                    "sealed_test": _evaluate(
                        model,
                        test[family],
                        device=device,
                        seed=seed + 2_000 + FAMILIES.index(family),
                    ),
                    "length_extrapolation_validation": _evaluate(
                        model,
                        extrapolation[family],
                        device=device,
                        seed=seed + 3_000 + FAMILIES.index(family),
                    ),
                }
            row["candidates"][candidate] = candidate_result
        results["seeds"].append(row)
        print(f"completed memory seed {seed}", flush=True)

    results["summary"] = {}
    for candidate in config["training"].get("candidates", ["gru"]):
        results["summary"][candidate] = {}
        for family in FAMILIES:
            results["summary"][candidate][family] = {}
            for split in ("validation", "sealed_test", "length_extrapolation_validation"):
                keys = ("mae", "mse", "reset_state_mae", "random_state_drop_mae", "state_replay_max_abs_delta", "checkpoint_restore_max_abs_delta")
                summary = {}
                for key in keys:
                    values = [
                        seed["candidates"][candidate]["families"][family][split][key]
                        for seed in results["seeds"]
                        if key in seed["candidates"][candidate]["families"][family][split]
                    ]
                    if values:
                        summary[f"{key}_mean"] = statistics.mean(values)
                if family == "order_recognition":
                    for key in ("exact_accuracy", "reset_state_accuracy", "random_state_drop_accuracy"):
                        values = [
                            seed["candidates"][candidate]["families"][family][split][key]
                            for seed in results["seeds"]
                            if key in seed["candidates"][candidate]["families"][family][split]
                        ]
                        if values:
                            summary[f"{key}_mean"] = statistics.mean(values)
                results["summary"][candidate][family][split] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"], "seeds": len(result["seeds"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
