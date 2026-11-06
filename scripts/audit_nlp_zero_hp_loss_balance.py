#!/usr/bin/env python3
"""Audit learned balancing across all NLP joint-loss objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW
from floorflow.nlp_flow.loss_balance import NLP_OBJECTIVES, NLPJointLossBalancer

ROOT = Path(__file__).resolve().parents[1]


def _device() -> torch.device:
    if torch.cuda.is_available():
        index = 1 if torch.cuda.device_count() > 1 else 0
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_zero_hp_loss_balance_v1.json",
    )
    args = parser.parse_args()
    device = _device()
    torch.manual_seed(20261102)
    balancer = NLPJointLossBalancer(NLP_OBJECTIVES).to(device)
    optimizer = ScheduleFreeAdamW(balancer.parameters())
    optimizer.train()
    scale = torch.tensor(
        [8.0, 4.0, 2.0, 1.5, 0.7, 0.3, 0.1],
        device=device,
    )
    history = []
    for _step in range(200):
        optimizer.zero_grad(set_to_none=True)
        losses = {
            name: scale[index] * torch.ones((), device=device)
            for index, name in enumerate(NLP_OBJECTIVES)
        }
        total, weights = balancer(losses)
        total.backward()
        optimizer.step()
        history.append(
            {
                "total": float(total.detach()),
                "weights": {
                    name: float(value)
                    for name, value in weights.items()
                },
            }
        )
    optimizer.eval()
    with torch.no_grad():
        losses = {
            name: scale[index] * torch.ones((), device=device)
            for index, name in enumerate(NLP_OBJECTIVES)
        }
        total, weights = balancer(losses)
    first = history[0]["weights"]
    final = {name: float(value) for name, value in weights.items()}
    all_changed = all(
        abs(final[name] - first[name]) > 1.0e-5
        for name in NLP_OBJECTIVES
    )
    report = {
        "version": "nlp-zero-hp-loss-balance-v1",
        "status": "pass" if all_changed and torch.isfinite(total) else "fail",
        "device": str(device),
        "objectives": list(NLP_OBJECTIVES),
        "manual_scalar_weights": False,
        "dynamic_weight_initial": first,
        "dynamic_weight_final": final,
        "loss_scales": {
            name: float(scale[index])
            for index, name in enumerate(NLP_OBJECTIVES)
        },
        "initial_total": history[0]["total"],
        "final_total": float(total),
        "validity_objective_present": "validity" in NLP_OBJECTIVES,
        "history_steps": len(history),
        "evidence_boundary": (
            "Joint-loss balancing contract only; this report does not establish "
            "model validity, emergence, or benchmark gains."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
