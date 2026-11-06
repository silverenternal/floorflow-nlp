#!/usr/bin/env python3
"""Compare schedule-free and horizon-scheduled AdamW at matched budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW

ROOT = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _dataset(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261102)
    inputs = torch.randn(2048, 16, generator=generator)
    targets = (
        torch.sin(inputs[:, 0] * 1.7)
        + inputs[:, 1:5].square().sum(dim=1) * 0.2
        - inputs[:, 5] * inputs[:, 6]
    )
    return inputs.to(device), targets.to(device)


def _model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
    ).to(device)


def _train(
    *,
    optimizer_kind: str,
    horizon: int,
    seed: int,
    device: torch.device,
) -> float:
    inputs, targets = _dataset(device)
    model = _model(seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    if optimizer_kind == "schedule_free":
        optimizer = ScheduleFreeAdamW(model.parameters())
        optimizer.train()
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=horizon,
        )
    batch_size = 128
    for _step in range(horizon):
        index = torch.randperm(
            len(inputs),
            generator=generator,
            device=device,
        )[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs[index]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, targets[index])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if optimizer_kind == "schedule_free":
        optimizer.eval()
    with torch.no_grad():
        return float(
            torch.nn.functional.mse_loss(
                model(inputs).squeeze(-1),
                targets,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_zero_hp_optimizer_ablation_v1.json",
    )
    args = parser.parse_args()
    device = _device()
    horizons = (128, 384)
    seeds = (17, 29, 43)
    rows = []
    for horizon in horizons:
        for seed in seeds:
            adam = _train(
                optimizer_kind="scheduled_adamw",
                horizon=horizon,
                seed=seed,
                device=device,
            )
            schedule_free = _train(
                optimizer_kind="schedule_free",
                horizon=horizon,
                seed=seed,
                device=device,
            )
            rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "scheduled_adamw_mse": adam,
                    "schedule_free_mse": schedule_free,
                    "schedule_free_not_worse": schedule_free <= adam,
                }
            )
    horizon_summary = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        horizon_summary[str(horizon)] = {
            "scheduled_adamw_mean_mse": sum(
                row["scheduled_adamw_mse"] for row in selected
            )
            / len(selected),
            "schedule_free_mean_mse": sum(
                row["schedule_free_mse"] for row in selected
            )
            / len(selected),
            "schedule_free_not_worse_all_seeds": all(
                row["schedule_free_not_worse"] for row in selected
            ),
        }
    report = {
        "version": "nlp-zero-hp-optimizer-ablation-v1",
        "status": (
            "pass"
            if all(
                value["schedule_free_not_worse_all_seeds"]
                for value in horizon_summary.values()
            )
            else "fail"
        ),
        "device": str(device),
        "learning_rates": {
            "scheduled_adamw": 1.0e-3,
            "schedule_free": 2.5e-3,
        },
        "horizon_specific_retuning": False,
        "same_initialization_per_seed": True,
        "same_data_order_per_pair": True,
        "horizons": list(horizons),
        "rows": rows,
        "horizon_summary": horizon_summary,
        "evidence_boundary": (
            "Matched optimization-contract probe only; it is not a language-model "
            "quality or capability result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
