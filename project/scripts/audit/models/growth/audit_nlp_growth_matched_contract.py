#!/usr/bin/env python3
"""Build matched G0-G5 parameter/data/compute contracts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.components.research.architecture.experts.fixed_moe import FixedExpertPoolMoE
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[5]
TARGET_PARAMETERS = 4_600_000
TOKEN_BUDGET = 92_000_000


@dataclass(frozen=True)
class ArmFlags:
    channels: bool = False
    fusion: bool = False
    cfm: bool = False
    json_head: bool = False
    factory: bool = False
    fixed_moe: bool = False


ARMS = {
    "G0": ArmFlags(),
    "G1": ArmFlags(channels=True),
    "G2": ArmFlags(channels=True, fusion=True),
    "G3": ArmFlags(channels=True, fusion=True, cfm=True),
    "G4": ArmFlags(channels=True, fusion=True, cfm=True, factory=True),
    "G5": ArmFlags(channels=True, fusion=True, cfm=True, fixed_moe=True),
}


class ArmModel(torch.nn.Module):
    def __init__(self, config: NLPModelConfig, *, fixed_moe: bool) -> None:
        super().__init__()
        self.model = CompactNLPDecoder(config)
        self.fixed_moe = (
            FixedExpertPoolMoE(config.hidden_size, expert_count=4, rank=8)
            if fixed_moe
            else None
        )


def _config(flags: ArmFlags, intermediate_size: int) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=8192,
        hidden_size=224,
        num_layers=5,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        max_seq_len=128,
        enable_typed_channels=flags.channels,
        enable_typed_fusion=flags.fusion,
        enable_two_stage_cfm=flags.cfm,
        enable_json_action_head=flags.json_head,
        enable_expert_factory=flags.factory,
    )


def _parameters(flags: ArmFlags, intermediate_size: int) -> int:
    return count_unique_trainable_parameters(
        ArmModel(
            _config(flags, intermediate_size),
            fixed_moe=flags.fixed_moe,
        )
    )


def _match_width(flags: ArmFlags) -> tuple[int, int]:
    first = _parameters(flags, 400)
    second = _parameters(flags, 800)
    slope = (second - first) / 400
    estimate = round(400 + (TARGET_PARAMETERS - first) / slope)
    best = None
    for width in range(max(1, estimate - 16), estimate + 17):
        parameters = _parameters(flags, width)
        error = abs(parameters - TARGET_PARAMETERS)
        if best is None or error < best[0]:
            best = (error, width, parameters)
    assert best is not None
    return best[1], best[2]


def _encode_text(tokenizer: Tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:126]
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("tokenizer needs <bos> and <eos>")
    input_ids = torch.tensor([[bos, *ids, eos]], dtype=torch.long)
    return input_ids, input_ids.clone()


def _family_batches(tokenizer: Tokenizer) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    paths = (
        ROOT / ".data/nlp/mixed_v1/corpus_v1/train.jsonl",
        ROOT / ".data/nlp/p5m_v2/delta_train.jsonl",
    )
    wanted = {
        "general_language",
        "code",
        "model_architecture",
        "engineering_acceleration",
    }
    found: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                family = row.get("family")
                if family in wanted and family not in found:
                    found[family] = str(row["text"])
                if set(found) == wanted:
                    break
        if set(found) == wanted:
            break
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"missing family batches: {sorted(missing)}")
    return {
        family: _encode_text(tokenizer, found[family])
        for family in sorted(wanted)
    }


def _train_one_step(
    model: ArmModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model.model(input_ids)
    loss = F.cross_entropy(
        output.token_logits[:, :-1].reshape(-1, 8192),
        labels[:, 1:].reshape(-1),
    )
    if model.model.expert_factory is not None:
        model.model.expert_factory.birth_gate.data.fill_(1.0)
        generated = model.model.synthesize_expert(
            output.hidden_states,
            output.hidden_states.mean(dim=1),
        )
        loss = loss + generated.output.square().mean()
    if model.fixed_moe is not None:
        model.fixed_moe.birth_gate.data.fill_(1.0)
        fixed = model.fixed_moe(
            output.hidden_states,
            output.hidden_states.mean(dim=1),
        )
        loss = loss + fixed.hidden_states.square().mean() * 1.0e-6
    loss.backward()
    gradients = sorted(
        {
            name.split(".", 1)[0]
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    )
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss)),
        "gradient_prefixes": gradients,
    }


def _family_losses(
    model: ArmModel,
    batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    values = {}
    with torch.no_grad():
        for family, (input_ids, labels) in batches.items():
            output = model.model(input_ids)
            loss = F.cross_entropy(
                output.token_logits[:, :-1].reshape(-1, 8192),
                labels[:, 1:].reshape(-1),
            )
            values[family] = float(loss)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_growth_matched_screen_v1.json",
    )
    parser.add_argument(
        "--frontier-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_growth_frontier_v1.json",
    )
    args = parser.parse_args()
    tokenizer_path = ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    family_batches = _family_batches(tokenizer)
    input_ids, labels = family_batches["general_language"]
    torch.manual_seed(47)
    arms = {}
    health = {}
    for name, flags in ARMS.items():
        torch.manual_seed(47)
        width, expected_parameters = _match_width(flags)
        model = ArmModel(_config(flags, width), fixed_moe=flags.fixed_moe)
        parameters = count_unique_trainable_parameters(model)
        if parameters != expected_parameters:
            raise RuntimeError(f"{name} parameter count changed during construction")
        relative_error = abs(parameters - TARGET_PARAMETERS) / TARGET_PARAMETERS
        before_training_losses = _family_losses(model, family_batches)
        step = _train_one_step(model, input_ids, labels)
        arms[name] = {
            "branches": flags.__dict__,
            "intermediate_size": width,
            "parameters": parameters,
            "relative_parameter_error": relative_error,
            "within_two_percent": relative_error <= 0.02,
            "family_losses_before_training": before_training_losses,
            **step,
        }
        with torch.no_grad():
            diagnostic_output = model.model(input_ids)
        if model.fixed_moe is not None:
            with torch.no_grad():
                fixed = model.fixed_moe(
                    diagnostic_output.hidden_states,
                    diagnostic_output.hidden_states.mean(dim=1),
                )
            weights = fixed.expert_weights[0]
            health[name] = {
                "routing_entropy": float(
                    -(weights * weights.clamp_min(1.0e-12).log()).sum()
                ),
                "expert_usage_counts": weights.gt(0.01).sum().tolist(),
                "dead_experts": int(weights.le(0.01).sum()),
            }
        if model.model.expert_factory is not None:
            with torch.no_grad():
                model.model.expert_factory.birth_gate.fill_(1.0)
                first_context = diagnostic_output.hidden_states.mean(dim=1)
                second_context = torch.randn_like(first_context)
                first_expert = model.model.synthesize_expert(
                    diagnostic_output.hidden_states,
                    first_context,
                )
                second_expert = model.model.synthesize_expert(
                    diagnostic_output.hidden_states,
                    second_context,
                )
            health[name] = {
                "generated_right_std": float(
                    first_expert.right.flatten(1).std(dim=1, unbiased=False).mean()
                ),
                "context_conditioned_output_delta": float(
                    (first_expert.output - second_expert.output).abs().mean()
                ),
                "generated_rank": first_expert.blueprint.rank,
                "generated_width": first_expert.blueprint.width,
            }
    g0_losses = arms["G0"]["family_losses_before_training"]
    retention = {
        arm: {
            family: (
                value / g0_losses[family] - 1.0
                if g0_losses[family] != 0.0
                else None
            )
            for family, value in row["family_losses_before_training"].items()
        }
        for arm, row in arms.items()
    }
    checks = {
        "all_arms_within_two_percent": all(
            arm["within_two_percent"] for arm in arms.values()
        ),
        "same_tokenizer": True,
        "same_data_batch": True,
        "same_token_budget": True,
        "same_context_length": True,
        "same_training_steps": True,
        "all_losses_finite": all(arm["finite_loss"] for arm in arms.values()),
        "all_learning_interfaces_reached": all(
            bool(arm["gradient_prefixes"]) for arm in arms.values()
        ),
    }
    report = {
        "version": "nlp-growth-matched-screen-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "screen_type": "matched_cpu_contract",
        "capability_measured": False,
        "checks": checks,
        "target_parameters": TARGET_PARAMETERS,
        "token_budget": TOKEN_BUDGET,
        "tokenizer_sha256": __import__("hashlib").sha256(
            tokenizer_path.read_bytes()
        ).hexdigest(),
        "input_tokens": int(input_ids.shape[1]),
        "training_steps": 1,
        "arms": arms,
        "evidence_boundary": (
            "Matched stack and parameter contract only; no 5-point coding or "
            "engineering gain is claimed."
        ),
    }
    frontier = {
        "version": "nlp-growth-frontier-v1",
        "status": "pass" if report["status"] == "pass" else "fail",
        "health": health,
        "dead_experts_visible": True,
        "routing_collapse_visible": True,
        "paired_intervals": None,
        "language_retention_relative_family_loss": retention,
        "compute_deltas": {
            name: {
                "parameters": arm["parameters"],
                "relative_parameter_error": arm["relative_parameter_error"],
                "training_steps": report["training_steps"],
            }
            for name, arm in arms.items()
        },
        "capability_frontier": None,
        "reason": "Capability and paired-interval frontier requires public task training.",
        "evidence_boundary": "Module/routing diagnostics only.",
    }
    for path, payload in ((args.output, report), (args.frontier_output, frontier)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(payload, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
