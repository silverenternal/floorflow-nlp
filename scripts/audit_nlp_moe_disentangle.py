#!/usr/bin/env python3
"""Audit generated-expert versus fixed-pool MoE mechanics without a capability claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.fixed_moe import FixedExpertPoolMoE
from floorflow.nlp_flow.model import ContextConditionedExpertFactory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_fixed_moe_control_v1.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_generated_vs_moe_v1.json",
    )
    parser.add_argument(
        "--claim-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_moe_disentanglement_claim_v1.json",
    )
    args = parser.parse_args()
    torch.manual_seed(131)
    hidden_size = 32
    rank = 8
    expert_count = 4
    generated = ContextConditionedExpertFactory(
        hidden_size,
        max_rank=rank,
        basis_count=expert_count,
    )
    fixed = FixedExpertPoolMoE(
        hidden_size,
        expert_count=expert_count,
        rank=rank,
    )
    with torch.no_grad():
        generated.birth_gate.fill_(0.2)
        fixed.birth_gate.fill_(0.2)
    parent = torch.randn(12, 6, hidden_size)
    context = torch.randn(12, hidden_size)
    generated_output = generated(parent, context)
    fixed_output = fixed(parent, context)
    generated_context_variation = (
        generated_output.right.flatten(1).std(dim=0).mean()
    ).detach()
    fixed_context_variation = (
        fixed_output.effective_right.flatten(1).std(dim=0).mean().detach()
    )
    routing_entropy = -(
        fixed_output.expert_weights
        * fixed_output.expert_weights.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    fixed_parameters = count_unique_trainable_parameters(fixed)
    generated_parameters = count_unique_trainable_parameters(generated)
    checks = {
        "generated_context_variation": bool(generated_context_variation > 0.0),
        "fixed_pool_has_constant_expert_count": fixed.left_basis.shape[0] == expert_count,
        "fixed_pool_routing_is_nonconstant": bool(
            fixed_output.expert_weights.std(dim=0).sum() > 0.0
        ),
        "no_fixed_pool_synthesis_api": not hasattr(fixed, "synthesize"),
        "generated_expert_shape_matches": (
            generated_output.output.shape == fixed_output.hidden_states.shape
        ),
    }
    control = {
        "version": "nlp-fixed-moe-control-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "role": "G5 fixed-expert MoE control",
        "expert_count": expert_count,
        "rank": rank,
        "hidden_size": hidden_size,
        "parameters": fixed_parameters,
        "fixed_expert_basis_count": int(fixed.left_basis.shape[0]),
        "learning_interface": "router-only over fixed expert basis",
        "checks": checks,
        "evidence_boundary": "Control architecture only; no coding/engineering quality claim.",
    }
    comparison = {
        "version": "nlp-generated-vs-moe-v1",
        "status": "pass" if control["status"] == "pass" else "fail",
        "parameters": {
            "generated_factory": generated_parameters,
            "fixed_pool_moe": fixed_parameters,
            "absolute_difference": abs(generated_parameters - fixed_parameters),
        },
        "context_variation": {
            "generated_effective_right_std": float(generated_context_variation),
            "fixed_effective_right_std": float(fixed_context_variation),
        },
        "fixed_routing": {
            "mean_entropy": float(routing_entropy.mean()),
            "minimum_entropy": float(routing_entropy.min()),
            "maximum_expert_weight": float(fixed_output.expert_weights.max()),
            "expert_usage_max_count": int(
                fixed_output.expert_weights.argmax(dim=-1).bincount(
                    minlength=expert_count
                ).max()
            ),
        },
        "behavioral_difference": (
            "Generated factors vary continuously with parent/error context; "
            "the fixed control only reweights a static expert basis."
        ),
        "capability_gain_tested": False,
        "evidence_boundary": (
            "Mechanism distinction only; G4-versus-G5 coding/engineering performance remains untested."
        ),
    }
    claim = {
        "version": "nlp-moe-disentanglement-claim-v1",
        "status": "blocked",
        "architecture_distinction": control["status"],
        "paired_capability_gain": None,
        "matched_final_parameters": False,
        "reason": (
            "No trained G4/G5 runs exist under matched parameters, tokens, and inference compute."
        ),
        "claim_decision": "do_not_claim_generated_expert_superiority",
        "evidence_boundary": "Negative/unresolved evidence is retained.",
    }
    for path, report in (
        (args.control_output, control),
        (args.comparison_output, comparison),
        (args.claim_output, claim),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
    if control["status"] != "pass" or comparison["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
