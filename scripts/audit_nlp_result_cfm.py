#!/usr/bin/env python3
"""Audit generated-AST-only Stage-B objectives and curriculum."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.result_cfm import (
    GeneratedASTCurriculum,
    GeneratedASTResultCFM,
    ResultCFMConfig,
)

ROOT = Path(__file__).resolve().parents[1]


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _curriculum_report() -> dict[str, object]:
    curriculum = GeneratedASTCurriculum(1_000)
    phases = {"teacher": 0, "mixed": 0, "generated_only": 0}
    teacher_mass = []
    boundaries = {}
    for step in range(curriculum.total_steps):
        schedule = curriculum.at(step)
        if schedule.phase not in boundaries:
            boundaries[schedule.phase] = step
        phases[schedule.phase] += 1
        teacher_mass.append(schedule.teacher_weight)
    last = curriculum.at(curriculum.total_steps - 1)
    nonincreasing = all(
        right <= left + 1.0e-12
        for left, right in itertools.pairwise(teacher_mass)
    )
    report = {
        "version": "nlp-ast-curriculum-v1",
        "status": "pass" if last.phase == "generated_only" and nonincreasing else "fail",
        "total_steps": curriculum.total_steps,
        "phase_steps": phases,
        "first_step_by_phase": boundaries,
        "teacher_weight_nonincreasing": nonincreasing,
        "final_schedule": {
            "phase": last.phase,
            "teacher_weight": last.teacher_weight,
            "generated_weight": last.generated_weight,
        },
        "primary_inference_uses_curriculum": False,
        "evidence_boundary": "Schedule contract only; no teacher-to-generated generalization claim.",
    }
    return report


def _objective_report(device: str) -> dict[str, object]:
    torch.manual_seed(71)
    model = GeneratedASTResultCFM(
        ResultCFMConfig(hidden_size=24, rank=8, primary_nfe=4)
    ).to(device)
    prompt = torch.randn(2, 24, device=device)
    generated_ast = torch.randn(2, 9, 24, device=device)
    ast_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False, False, False],
        ]
    ).to(device)
    target = torch.randn(2, 6, 24, device=device)
    noise = torch.randn_like(target)
    parent_indices = torch.tensor(
        [
            [-1, 0, 1, 2, 0, -2, -2, -2, -2],
            [-1, 0, 0, 1, -2, -2, -2, -2, -2],
        ],
        device=device,
    )
    node_levels = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0, 0, 0, 0],
            [0, 1, 2, 2, 0, 0, 0, 0, 0],
        ],
        device=device,
    )
    loss, _output = model.training_loss(
        target,
        prompt,
        generated_ast,
        ast_mask,
        parent_indices=parent_indices,
        node_levels=node_levels,
        verifier_target=torch.tensor([1.0, 0.0]),
        time=torch.tensor([0.25, 0.75], device=device),
        noise=noise,
    )
    loss.total.backward()
    gradients = {
        "velocity": model.vector_field.state.weight.grad is not None,
        "reconstruction": model.conditioner.project[0].weight.grad is not None,
        "verifier": model.verifier.weight.grad is not None,
        "ast_consistency": model.consistency.weight.grad is not None,
        "uncertainty": model.uncertainty.weight.grad is not None,
        "ast_topology_attention": (
            model.conditioner.attention.in_proj_weight.grad is not None
        ),
        "result_self_attention": (
            model.vector_field.self_attention.in_proj_weight.grad is not None
        ),
    }
    finite = {
        name: bool(torch.isfinite(value))
        for name, value in loss.as_dict().items()
    }

    fixed_noise = torch.randn(1, 6, 24, device=device)
    first = model.sample(
        prompt[:1],
        generated_ast[:1],
        ast_mask[:1],
        parent_indices=parent_indices[:1],
        node_levels=node_levels[:1],
        result_length=6,
        steps=4,
        noise=fixed_noise,
    )
    second = model.sample(
        prompt[:1],
        generated_ast[:1] + 0.5,
        ast_mask[:1],
        parent_indices=parent_indices[:1],
        node_levels=node_levels[:1],
        result_length=6,
        steps=4,
        noise=fixed_noise,
    )
    condition_sensitivity = float((first.endpoint - second.endpoint).abs().mean())
    topology_first = model.sample(
        prompt[:1],
        generated_ast[:1],
        ast_mask[:1],
        parent_indices=parent_indices[:1],
        node_levels=node_levels[:1],
        result_length=6,
        steps=4,
        noise=fixed_noise,
    )
    topology_second = model.sample(
        prompt[:1],
        generated_ast[:1],
        ast_mask[:1],
        parent_indices=torch.tensor(
            [[-1, 0, 0, 0, 0, -2, -2, -2, -2]],
            device=device,
        ),
        node_levels=node_levels[:1],
        result_length=6,
        steps=4,
        noise=fixed_noise,
    )
    topology_sensitivity = float(
        (topology_first.endpoint - topology_second.endpoint).abs().mean()
    )
    signature = inspect.signature(model.sample).parameters
    report = {
        "version": "nlp-result-cfm-objective-audit-v1",
        "status": (
            "pass"
            if (
                all(gradients.values())
                and all(finite.values())
                and condition_sensitivity > 0.0
                and topology_sensitivity > 0.0
            )
            else "fail"
        ),
        "losses": {name: float(value.detach()) for name, value in loss.as_dict().items()},
        "finite": finite,
        "gradients": gradients,
        "generated_ast_condition_sensitivity": condition_sensitivity,
        "generated_ast_topology_sensitivity": topology_sensitivity,
        "primary_sample_parameters": {
            name: str(parameter)
            for name, parameter in signature.items()
        },
        "has_target_argument": any("target" in name for name in signature),
        "has_teacher_argument": any("teacher" in name for name in signature),
        "evidence_boundary": (
            "GPU objective and generated-AST sensitivity contract only; "
            "Stage-B executable-quality gain is untested."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_curriculum_v1.json",
    )
    parser.add_argument(
        "--objective-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_result_cfm_objective_audit_v1.json",
    )
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    reports = (_curriculum_report(), _objective_report(args.device))
    for path, report in zip(
        (args.curriculum_output, args.objective_output),
        reports,
        strict=True,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(report, indent=2))
    if any(report["status"] != "pass" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
