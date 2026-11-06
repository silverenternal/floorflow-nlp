#!/usr/bin/env python3
"""Audit benchmark-proxy gradient reachability in the joint objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.model.benchmark_proxy import (
    REGRESSION_PROXY_TARGETS,
    SCALAR_PROXY_TARGETS,
    TOKEN_PROXY_TARGETS,
    BenchmarkProxyConfig,
    BenchmarkProxyObjective,
)

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
        default=ROOT / "outputs/reports/nlp_benchmark_proxy_joint_ablation_v1.json",
    )
    args = parser.parse_args()
    device = _device()
    torch.manual_seed(20261102)
    objective = BenchmarkProxyObjective(
        BenchmarkProxyConfig(hidden_size=32, retrieval_dim=8)
    ).to(device)
    hidden = torch.randn(4, 8, 32, device=device, requires_grad=True)
    pooled = hidden.mean(dim=1)
    scalar_targets = {
        name: torch.randint(0, 2, (4,), device=device)
        for name in SCALAR_PROXY_TARGETS
    }
    token_targets = {
        name: torch.randint(0, 2, (4, 8), device=device)
        for name in TOKEN_PROXY_TARGETS
    }
    regression_targets = {
        name: torch.rand(4, device=device)
        for name in REGRESSION_PROXY_TARGETS
    }
    losses = {
        **objective.scalar_losses(pooled, scalar_targets),
        **objective.regression_losses(pooled, regression_targets),
        **objective.token_losses(hidden, token_targets),
        "L_test_output": objective.test_output_loss(
            pooled,
            torch.randint(0, objective.config.test_output_classes, (4,), device=device),
        ),
        "L_test_output_token_sequence": objective.test_output_token_loss(
            hidden,
            torch.randint(0, objective.config.test_output_classes, (4, 8), device=device),
        ),
        "L_multiple_choice": objective.multiple_choice_loss(
            pooled,
            torch.randn(4, 4, 32, device=device),
            torch.randint(0, 4, (4,), device=device),
        ),
        "L_instruction_constraints": objective.constraint_loss(
            pooled,
            torch.randint(
                0,
                2,
                (4, objective.config.max_constraints),
                device=device,
            ).float(),
            torch.ones(
                4,
                objective.config.max_constraints,
                dtype=torch.bool,
                device=device,
            ),
        ),
        "L_retrieval_contrastive": objective.retrieval_loss(
            pooled,
            torch.randn(4, 5, 32, device=device),
            torch.randint(0, 5, (4,), device=device),
        ),
    }
    total = objective.total_loss(losses)
    total.backward()
    missing = [
        name
        for name, parameter in objective.named_parameters()
        if parameter.grad is None
    ]
    report = {
        "version": "nlp-benchmark-proxy-joint-ablation-v1",
        "status": "pass" if not missing and torch.isfinite(total) else "fail",
        "device": str(device),
        "proxy_objectives": len(losses),
        "loss_names": sorted(losses),
        "loss_values": {
            name: float(value.detach())
            for name, value in losses.items()
        },
        "total_loss": float(total.detach()),
        "all_proxy_parameters_receive_gradient": not missing,
        "missing_gradients": missing,
        "hidden_gradient_present": hidden.grad is not None,
        "external_rank_alignment_measured": False,
        "evidence_boundary": (
            "Joint-loss wiring audit only; no external benchmark score or "
            "proxy-to-external rank-alignment claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
