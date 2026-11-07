"""Train and checkpoint a tiny synthetic neuron control on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from newera.neuron import MLPNeuron

CHECKPOINT_SCHEMA = 1
MODEL_CONFIG = {"input_size": 2, "hidden_size": 16, "output_size": 1}
LEARNING_RATE = 0.03


def _data(seed: int, *, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(count, 2, generator=generator)
    targets = 0.5 * inputs[:, :1] - 1.2 * inputs[:, 1:2] + 0.4
    return inputs, targets


def _model_sha256(model: MLPNeuron) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_segment(
    *,
    steps: int,
    checkpoint: Path,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    torch.manual_seed(seed)
    model = MLPNeuron(**MODEL_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    completed_steps = 0
    if resume:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if payload.get("seed") != seed or payload.get("model_config") != MODEL_CONFIG:
            raise ValueError("checkpoint configuration does not match this run")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_steps = int(payload["completed_steps"])
        torch.set_rng_state(payload["torch_rng_state"])

    train_inputs, train_targets = _data(seed + 1, count=64)
    validation_inputs, validation_targets = _data(seed + 2, count=32)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_inputs).value
        loss = torch.nn.functional.mse_loss(prediction, train_targets)
        loss.backward()
        optimizer.step()
    completed_steps += steps

    model.eval()
    with torch.no_grad():
        train_mse = torch.nn.functional.mse_loss(
            model(train_inputs).value, train_targets
        ).item()
        validation_mse = torch.nn.functional.mse_loss(
            model(validation_inputs).value, validation_targets
        ).item()
    result = {
        "completed_steps": completed_steps,
        "train_mse": train_mse,
        "validation_mse": validation_mse,
        "model_sha256": _model_sha256(model),
    }
    _save_checkpoint(
        checkpoint,
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "seed": seed,
            "model_config": MODEL_CONFIG,
            "learning_rate": LEARNING_RATE,
            "completed_steps": completed_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True, help="additional CPU training steps")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_segment(
        steps=args.steps,
        checkpoint=args.checkpoint,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
