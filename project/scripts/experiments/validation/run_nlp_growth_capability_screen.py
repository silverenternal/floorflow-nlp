#!/usr/bin/env python3
"""Preflight and orchestration for the authorized G0-G5 capability screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.components.research.architecture.experts.fixed_moe import FixedExpertPoolMoE
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Arm(torch.nn.Module):
    def __init__(self, config: NLPModelConfig, *, fixed_moe: bool) -> None:
        super().__init__()
        self.model = CompactNLPDecoder(config)
        self.fixed_moe = (
            FixedExpertPoolMoE(config.hidden_size, expert_count=4, rank=8)
            if fixed_moe
            else None
        )


def _flags(arm: str) -> dict[str, bool]:
    index = int(arm[1:])
    return {
        "channels": index >= 1,
        "fusion": index >= 2,
        "cfm": index >= 3,
        "factory": index == 4,
        "fixed_moe": index == 5,
    }


def _arm_parameters(arm: str, intermediate_size: int) -> int:
    flags = _flags(arm)
    config = NLPModelConfig(
        vocab_size=8192,
        hidden_size=224,
        num_layers=5,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        max_seq_len=512,
        enable_typed_channels=flags["channels"],
        enable_typed_fusion=flags["fusion"],
        enable_two_stage_cfm=flags["cfm"],
        enable_expert_factory=flags["factory"],
    )
    return count_unique_trainable_parameters(
        _Arm(config, fixed_moe=flags["fixed_moe"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "project/configs/models_training/learning/growth/nlp_growth_capability_screen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_growth_capability_preflight_v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    todo = json.loads((ROOT / "nlptodo.json").read_text(encoding="utf-8"))
    gpu_authorized = bool(todo["current_round"].get("gpu_authorized", False))
    required_paths = [
        spec["tokenizer"],
        spec["corpus_manifest"],
        *spec["train_shards"],
        *spec["validation_shards"],
        spec["benchmark_manifest"],
        spec["executor_contract"],
    ]
    missing_paths = [
        path for path in required_paths if not (ROOT / path).exists()
    ]
    corpus = json.loads(
        (ROOT / spec["corpus_manifest"]).read_text(encoding="utf-8")
    )
    token_count = int(corpus["combined_total_tokens"])
    tokenizer_hash = _sha256(ROOT / spec["tokenizer"])
    target = int(spec["target_parameters"])
    tolerance = float(spec["parameter_tolerance"])
    arms = {}
    for name, payload in spec["arms"].items():
        parameters = _arm_parameters(name, int(payload["intermediate_size"]))
        relative_error = abs(parameters - target) / target
        arms[name] = {
            "parameters": parameters,
            "relative_error": relative_error,
            "within_tolerance": relative_error <= tolerance,
        }
    preflight_checks = {
        "all_required_paths_exist": not missing_paths,
        "p5m_tokens_available": token_count >= int(spec["token_budget"]),
        "tokenizer_hash_matches_manifest": (
            tokenizer_hash == corpus["base"]["tokenizer_sha256"]
        ),
        "all_arms_within_parameter_tolerance": all(
            arm["within_tolerance"] for arm in arms.values()
        ),
        "two_seeds_declared": len(set(spec["seeds"])) >= 2,
        "no_best_of_n": spec["inference"]["best_of_n"] is False,
        "temperature_zero": spec["inference"]["temperature"] == 0.0,
    }
    cuda_available = torch.cuda.is_available()
    ready = (
        all(preflight_checks.values())
        and gpu_authorized
        and cuda_available
    )
    report = {
        "version": "nlp-growth-capability-preflight-v1",
        "status": "ready" if ready else "blocked",
        "checks": preflight_checks,
        "gpu": {
            "authorized": gpu_authorized,
            "cuda_available": cuda_available,
            "required_device": "cuda:0",
        },
        "missing_paths": missing_paths,
        "token_count": token_count,
        "token_budget": spec["token_budget"],
        "tokenizer_sha256": tokenizer_hash,
        "arms": arms,
        "seeds": spec["seeds"],
        "planned_runs": len(spec["seeds"]) * len(spec["arms"]),
        "execution_command_template": (
            "uv run python project/scripts/experiments/validation/run_nlp_growth_capability_screen.py --execute"
        ),
        "blocker": (
            None
            if ready
            else "GPU authorization or preflight requirement is not satisfied"
        ),
        "evidence_boundary": spec["evidence_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.execute:
        if report["status"] != "ready":
            raise RuntimeError(report["blocker"])
        raise RuntimeError(
            "authorized execution entrypoint is ready, but the capability trainer "
            "must be implemented in the next package"
        )


if __name__ == "__main__":
    main()
