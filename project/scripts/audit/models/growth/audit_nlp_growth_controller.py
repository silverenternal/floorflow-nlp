#!/usr/bin/env python3
"""Audit learned, target-free, bounded neuron-growth decisions."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import torch

from newera.nlp_flow.components.research.architecture.growth.growth_controller import (
    GROWTH_ACTIONS,
    BoundedGrowthManager,
    LearnedGrowthController,
    growth_action_utility,
    random_growth_action,
)

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_growth_controller_probe_v1.json",
    )
    args = parser.parse_args()
    torch.manual_seed(83)
    generator = torch.Generator().manual_seed(211)
    features = torch.rand(1_024, 7, generator=generator)
    utilities = growth_action_utility(features)
    targets = utilities.argmax(dim=-1)
    model = LearnedGrowthController(hidden_size=48)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    losses = []
    for _step in range(350):
        optimizer.zero_grad(set_to_none=True)
        loss = model.policy_loss(
            features,
            action_targets=targets,
            utility_targets=utilities,
        )
        loss.total.backward()
        optimizer.step()
        losses.append(float(loss.total.detach()))

    with torch.no_grad():
        decision = model(features)
        predicted = decision.action_logits.argmax(dim=-1)
        learned_utilities = utilities.gather(1, predicted[:, None]).squeeze(1)
    random_actions = random_growth_action(307, features.shape[0])
    random_utilities = utilities.gather(1, random_actions[:, None]).squeeze(1)
    learned_mean = float(learned_utilities.mean())
    random_mean = float(random_utilities.mean())
    oracle_mean = float(utilities.max(dim=-1).values.mean())
    action_counts = Counter(
        GROWTH_ACTIONS[int(index)] for index in predicted.tolist()
    )

    manager = BoundedGrowthManager(
        max_growth_rounds=3,
        parameter_budget=23_000_000,
        initial_parameters=22_900_000,
    )
    for _round in range(3):
        manager.apply(
            "birth",
            birth_parameters=20_000,
            retired_parameters=20_000,
        )
    try:
        manager.apply("birth", birth_parameters=1, retired_parameters=1)
        fourth_round_rejected = False
    except ValueError:
        fourth_round_rejected = True
    manager.freeze()
    try:
        manager.apply("retire", retired_parameters=1)
        post_freeze_rejected = False
    except RuntimeError:
        post_freeze_rejected = True

    signature = inspect.signature(model.forward).parameters
    checks = {
        "learned_beats_random": learned_mean > random_mean,
        "multiple_actions": len(action_counts) >= 2,
        "target_free": not any("target" in name for name in signature),
        "round_cap": fourth_round_rejected,
        "zero_net_parameter_growth": manager.active_parameters == 22_900_000,
        "post_freeze_rejected": post_freeze_rejected,
        "finite_losses": all(math.isfinite(value) for value in losses),
    }
    report = {
        "version": "nlp-growth-controller-probe-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "training": {
            "steps": len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "policy": {
            "actions": dict(sorted(action_counts.items())),
            "learned_mean_utility": learned_mean,
            "random_mean_utility": random_mean,
            "oracle_mean_utility": oracle_mean,
            "relative_gain_over_random": (
                (learned_mean - random_mean) / max(abs(random_mean), 1.0e-12)
            ),
        },
        "manager": {
            "events": manager.events,
            "active_parameters": manager.active_parameters,
            "growth_rounds": manager.growth_rounds,
            "frozen": manager.frozen,
        },
        "controller_signature": {
            name: str(parameter) for name, parameter in signature.items()
        },
        "evidence_boundary": (
            "CPU synthetic decision probe only; no coding, engineering, or growth-frontier claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
