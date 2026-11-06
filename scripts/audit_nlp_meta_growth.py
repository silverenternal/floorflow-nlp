#!/usr/bin/env python3
"""Audit birth-utility learning and meta-validation leakage boundaries."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import torch

from floorflow.nlp_flow.meta_growth import (
    BirthUtilityNet,
    MetaGrowthConfig,
    MetaGrowthLoop,
    birth_utility_target,
)

ROOT = Path(__file__).resolve().parents[1]


def _model_and_data() -> tuple[
    BirthUtilityNet,
    torch.Tensor,
    torch.Tensor,
    list[float],
    dict[str, float],
]:
    torch.manual_seed(97)
    generator = torch.Generator().manual_seed(223)
    features = torch.rand(1_024, 7, generator=generator)
    gain = torch.rand(1_024, generator=generator) - 0.2
    random_gain = torch.rand(1_024, generator=generator) * 0.7
    fixed_moe_gain = torch.rand(1_024, generator=generator) * 0.7
    parameter_cost = torch.rand(1_024, generator=generator)
    compute_cost = torch.rand(1_024, generator=generator)
    dead = (torch.rand(1_024, generator=generator) > 0.75).float()
    targets = birth_utility_target(
        post_adaptation_gain=gain,
        random_expert_gain=random_gain,
        fixed_moe_gain=fixed_moe_gain,
        parameter_cost=parameter_cost,
        compute_cost=compute_cost,
        dead_module=dead,
        config=MetaGrowthConfig(),
    )
    model = BirthUtilityNet(hidden_size=40)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012)
    losses = []
    for _step in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = model.utility_loss(features, target_utility=targets)
        loss.total.backward()
        optimizer.step()
        losses.append(float(loss.total.detach()))
    return model, features, targets, losses, {
        "random_gain": float(random_gain.mean()),
        "fixed_moe_gain": float(fixed_moe_gain.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_meta_growth_objective_v1.json",
    )
    parser.add_argument(
        "--leakage-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_meta_growth_leakage_v1.json",
    )
    args = parser.parse_args()
    model, features, targets, losses, baselines = _model_and_data()
    with torch.no_grad():
        predicted, birth_logits = model(features)
    positive = targets > 0.0
    objective_checks = {
        "finite_losses": all(math.isfinite(value) for value in losses),
        "positive_utility_ranked_higher": bool(
            predicted[positive].mean() > predicted[~positive].mean()
        ),
        "birth_logits_learned": bool(birth_logits.std() > 1.0e-4),
        "target_free_forward": not any(
            "target" in name for name in inspect.signature(model.forward).parameters
        ),
    }
    objective = {
        "version": "nlp-meta-growth-objective-v1",
        "status": "pass" if all(objective_checks.values()) else "fail",
        "checks": objective_checks,
        "training": {
            "steps": len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "ranking": {
            "positive_target_mean": float(predicted[positive].mean()),
            "nonpositive_target_mean": float(predicted[~positive].mean()),
            "birth_rate": float((birth_logits >= 0).float().mean()),
            "random_expert_gain_mean": baselines["random_gain"],
            "fixed_moe_gain_mean": baselines["fixed_moe_gain"],
        },
        "evidence_boundary": (
            "CPU synthetic birth-utility probe only; no downstream capability gain is claimed."
        ),
    }

    with torch.no_grad():
        model.utility_head.weight.zero_()
        model.utility_head.bias.fill_(2.0)
        model.birth_head.weight.zero_()
        model.birth_head.bias.fill_(2.0)
    loop = MetaGrowthLoop(model, config=MetaGrowthConfig(max_adaptation_steps=6))
    validation = loop.run_episode(
        torch.ones(1, 7),
        phase="meta_validation",
        observed_gain=0.0,
        requested_adaptation_steps=6,
    )
    train = loop.run_episode(
        torch.ones(1, 7),
        phase="meta_train",
        observed_gain=0.25,
        requested_adaptation_steps=99,
    )
    loop.freeze_factory()
    try:
        loop.run_episode(
            torch.ones(1, 7),
            phase="meta_train",
            observed_gain=1.0,
            requested_adaptation_steps=1,
        )
        post_freeze_blocked = False
    except RuntimeError:
        post_freeze_blocked = True
    leakage_checks = {
        "validation_never_synthesizes": "synthesize" not in validation.events,
        "validation_never_adapts": validation.adaptation_steps == 0,
        "train_is_bounded": train.adaptation_steps == 6,
        "factory_freeze_blocks_births": post_freeze_blocked,
    }
    leakage = {
        "version": "nlp-meta-growth-leakage-v1",
        "status": "pass" if all(leakage_checks.values()) else "fail",
        "checks": leakage_checks,
        "validation_episode": {
            "events": validation.events,
            "committed": validation.committed,
            "adaptation_steps": validation.adaptation_steps,
        },
        "train_episode": {
            "events": train.events,
            "committed": train.committed,
            "adaptation_steps": train.adaptation_steps,
        },
        "factory_frozen": loop.factory_frozen,
        "evidence_boundary": (
            "Episode-boundary audit only; real coding/engineering verification remains required."
        ),
    }
    for path, report in (
        (args.objective_output, objective),
        (args.leakage_output, leakage),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(report, indent=2))
    if objective["status"] != "pass" or leakage["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
