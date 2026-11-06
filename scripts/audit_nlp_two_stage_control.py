#!/usr/bin/env python3
"""Audit one-stage control fairness and keep the AST-first claim blocked."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_cfm import ASTCodecVocabulary, ASTStageAFlow, encode_ast_graph
from floorflow.nlp_flow.ast_codec import parse_ast
from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.result_cfm import GeneratedASTResultCFM, ResultCFMConfig
from floorflow.nlp_flow.two_stage_control import (
    TwoStageASTFirstPipeline,
    match_control_width,
)

ROOT = Path(__file__).resolve().parents[1]


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_one_stage_cfm_control_v1.json",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_two_stage_vs_one_stage_v1.json",
    )
    parser.add_argument(
        "--claim-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_first_claim_v1.json",
    )
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    torch.manual_seed(151)
    sources = (
        ("python", "def add(a, b):\n    return a + b\n"),
        ("rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n"),
        ("cpp", "int add(int a, int b) { return a + b; }\n"),
    )
    graphs = [parse_ast(source, language) for language, source in sources]
    vocabulary = ASTCodecVocabulary.from_graphs(graphs)
    encoded = encode_ast_graph(
        graphs[0],
        vocabulary,
        max_nodes=32,
        template_id=vocabulary.template_to_id[sources[0][1]],
    )
    stage_a = ASTStageAFlow(
        vocabulary,
        hidden_size=16,
        num_heads=4,
        max_nodes=32,
    ).to(args.device)
    stage_b = GeneratedASTResultCFM(
        ResultCFMConfig(hidden_size=16, rank=6)
    ).to(args.device)
    pipeline = TwoStageASTFirstPipeline(stage_a, stage_b).to(args.device)
    control, match = match_control_width(
        pipeline,
        candidate_widths=tuple(range(16, 1_025, 8)),
        condition_slots=8,
    )
    control = control.to(args.device)
    prompt = torch.randn(1, 16, device=args.device)
    ast_noise = torch.randn(1, 32, 16, device=args.device)
    result_noise = torch.randn(1, 5, 16, device=args.device)
    two_stage = pipeline.sample_fixture_latent(
        prompt,
        encoded.node_mask[None].to(args.device),
        level_masks={
            name: value[None].to(args.device)
            for name, value in encoded.level_masks.items()
        },
        parent_indices=encoded.parent_indices[None].to(args.device),
        node_levels=encoded.levels[None].to(args.device),
        result_length=5,
        steps_a=4,
        steps_b=4,
        ast_noise=ast_noise,
        result_noise=result_noise,
    )
    one_stage = control.sample(
        prompt,
        result_length=5,
        steps=8,
        noise=result_noise,
    )
    two_signature = inspect.signature(pipeline.sample).parameters
    one_signature = inspect.signature(control.sample).parameters
    fairness_checks = {
        "parameters_within_two_percent": match.passed,
        "total_nfe_matched": two_stage.total_nfe == 8,
        "one_stage_has_no_target": not any("target" in name for name in one_signature),
        "one_stage_has_no_ast_input": not any("ast" in name for name in one_signature),
        "two_stage_has_no_target": not any("target" in name for name in two_signature),
        "finite_outputs": bool(
            torch.isfinite(two_stage.result_output.endpoint).all()
            and torch.isfinite(one_stage.endpoint).all()
        ),
    }
    control_report = {
        "version": "nlp-one-stage-cfm-control-v1",
        "status": "pass" if all(fairness_checks.values()) else "fail",
        "checks": fairness_checks,
        "parameters": {
            "two_stage": count_unique_trainable_parameters(pipeline),
            "one_stage": count_unique_trainable_parameters(control),
            "control_width": match.width,
            "relative_difference": match.relative_difference,
            "tolerance": 0.02,
        },
        "inference": {
            "two_stage_total_nfe": two_stage.total_nfe,
            "one_stage_total_nfe": 8,
            "tokens_matched": None,
            "measured_flops_matched": None,
            "reason": "No trained matched runs or token workload exist yet.",
        },
        "evidence_boundary": (
            "Control interface and parameter/NFE matching only; executable quality is untested."
        ),
    }
    comparison_report = {
        "version": "nlp-two-stage-vs-one-stage-v1",
        "status": "blocked",
        "ast_parse_rate": None,
        "two_stage_executable_rate": None,
        "one_stage_executable_rate": None,
        "paired_cases": 0,
        "reason": "A trained matched pair and frozen executable suite are required.",
        "evidence_boundary": "No AST-first advantage is claimed.",
    }
    claim_report = {
        "version": "nlp-ast-first-claim-v1",
        "status": "blocked",
        "control_contract": control_report["status"],
        "generated_ast_only": True,
        "nondegenerate_ast": None,
        "paired_capability_gain": None,
        "claim_decision": "do_not_promote_ast_first_claim",
        "evidence_boundary": "Negative/unresolved evidence retained.",
    }
    for path, report in (
        (args.control_output, control_report),
        (args.comparison_output, comparison_report),
        (args.claim_output, claim_report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
    if control_report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
