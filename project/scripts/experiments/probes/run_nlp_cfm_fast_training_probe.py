#!/usr/bin/env python3
"""Compare CFM timestep sampling under identical optimization budgets."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from newera.nlp_flow.components.research.training.objectives.cfm_training import (
    CFMTrainingConfig,
    ModelEMA,
    clip_and_step,
    flow_velocity_target,
    interpolate_flow,
    sample_flow_times,
    warmup_cosine_scale,
)
from newera.nlp_flow.model.components.training.objectives.cfm import LowRankVectorField

ROOT = Path(__file__).resolve().parents[4]


def _conditional_target(context: torch.Tensor) -> torch.Tensor:
    return torch.sin(1.3 * context) + 0.25 * torch.cos(0.7 * context)


def _eval_loss(
    model: LowRankVectorField,
    context: torch.Tensor,
    noise: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        state = interpolate_flow(noise, target, time)
        velocity = model(state, time, context)
        loss = (velocity - flow_velocity_target(noise, target)).square().mean()
    return float(loss)


def _run_arm(
    template: LowRankVectorField,
    config: dict,
    arm: dict,
    device: str,
) -> dict[str, object]:
    model = copy.deepcopy(template).to(device).train()
    ema = ModelEMA(model, decay=float(config["ema_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_scale(
            step,
            int(config["steps"]),
            warmup_steps=int(config["warmup_steps"]),
        ),
    )
    train_config = CFMTrainingConfig(
        timestep_sampler=arm["timestep_sampler"],
        logit_normal_std=float(arm.get("logit_normal_std", 0.8)),
        ema_decay=float(config["ema_decay"]),
        gradient_clip_norm=float(config["gradient_clip_norm"]),
    )
    generator = torch.Generator(device=device).manual_seed(
        int(config["seed"]) + 11
    )
    batch = int(config["batch_size"])
    hidden = int(config["hidden_size"])
    context = torch.randn(
        batch,
        hidden,
        device=device,
        generator=generator,
    )
    target = _conditional_target(context)
    eval_context = torch.randn(
        batch,
        hidden,
        device=device,
        generator=generator,
    )
    eval_target = _conditional_target(eval_context)
    fixed_noise = torch.randn(
        batch,
        hidden,
        device=device,
        generator=generator,
    )
    fixed_time = torch.linspace(0.05, 0.95, batch, device=device)
    history = []
    steps_to_ten_percent = None
    initial_eval = _eval_loss(
        model,
        eval_context,
        fixed_noise,
        eval_target,
        fixed_time,
    )
    for step in range(int(config["steps"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn(
            batch,
            hidden,
            device=device,
            generator=generator,
        )
        time = sample_flow_times(
            batch,
            train_config,
            device=device,
            generator=generator,
        )
        state = interpolate_flow(noise, target, time)
        velocity = model(state, time, context)
        loss = (velocity - flow_velocity_target(noise, target)).square().mean()
        loss.backward()
        gradient_norm = clip_and_step(
            optimizer,
            model,
            max_grad_norm=train_config.gradient_clip_norm,
        )
        scheduler.step()
        ema.update(model)
        if step in set(config["evaluation_steps"]):
            model_eval = _eval_loss(
                model,
                eval_context,
                fixed_noise,
                eval_target,
                fixed_time,
            )
            ema_eval = _eval_loss(
                ema.module,
                eval_context,
                fixed_noise,
                eval_target,
                fixed_time,
            )
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "model_eval_loss": model_eval,
                    "ema_eval_loss": ema_eval,
                    "gradient_norm": gradient_norm,
                }
            )
            if (
                steps_to_ten_percent is None
                and ema_eval <= 0.1 * initial_eval
            ):
                steps_to_ten_percent = step
    return {
        "arm": arm["id"],
        "timestep_sampler": arm["timestep_sampler"],
        "initial_eval_loss": initial_eval,
        "final_ema_eval_loss": history[-1]["ema_eval_loss"],
        "steps_to_10_percent_initial": steps_to_ten_percent,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_cfm_fast_training.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_cfm_fast_training_probe_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    requested_device = str(config.get("device", "cuda:1"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for CFM probe but CUDA is unavailable")
    torch.manual_seed(int(config["seed"]))
    template = LowRankVectorField(
        int(config["hidden_size"]),
        int(config["rank"]),
    )
    arms = [
        _run_arm(template, config, arm, requested_device)
        for arm in config["arms"]
    ]
    finite = all(
        math.isfinite(row["ema_eval_loss"])
        for arm in arms
        for row in arm["history"]
    )
    improved = all(
        arm["final_ema_eval_loss"] < arm["initial_eval_loss"]
        for arm in arms
    )
    ranked = sorted(
        arms,
        key=lambda arm: (
            float("inf")
            if arm["steps_to_10_percent_initial"] is None
            else arm["steps_to_10_percent_initial"],
            arm["final_ema_eval_loss"],
        ),
    )
    report = {
        "version": "nlp-cfm-fast-training-probe-v1",
        "status": "pass" if finite and improved else "fail",
        "device": requested_device,
        "checks": {
            "all_evaluations_finite": finite,
            "both_arms_improve": improved,
            "same_steps_data_parameters": True,
        },
        "fastest_arm": ranked[0]["arm"],
        "arms": arms,
        "evidence_boundary": config["evidence_boundary"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
