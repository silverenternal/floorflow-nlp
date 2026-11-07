#!/usr/bin/env python3
"""Audit the parallel gap-training and cross-attention aggregation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXPECTED_GAPS = {
    "json_validity",
    "ast_structure",
    "stage_b_cfm",
    "python_execution",
    "rust_execution",
    "cpp_execution",
    "engineering_acceleration",
    "model_architecture",
    "language_retention",
    "compositional_emergence",
}
EXPECTED_STAGES = (
    "stage_1_parallel_gap_training",
    "stage_2_router_calibration",
    "stage_3_joint_low_lr",
)


def validate_contract(payload: dict) -> tuple[str, ...]:
    errors = []
    if payload.get("version") != "nlp-gap-parallel-training-v1":
        errors.append("unexpected contract version")
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        errors.append("tracks must be an object")
        tracks = {}
    compressed = tracks.get("compressed_primary", {})
    open_track = tracks.get("open_capability_exploratory", {})
    if compressed.get("token_parameter_ratio") != 20:
        errors.append("compressed primary must retain the 20:1 data ratio")
    if open_track.get("claim_eligible") is not False:
        errors.append("open-capability track must be ineligible for primary claims")
    if not open_track.get("required_follow_up"):
        errors.append("open-capability track must define compression follow-up")

    gaps = payload.get("gaps")
    if not isinstance(gaps, list):
        errors.append("gaps must be a list")
        gaps = []
    gap_ids = [gap.get("id") for gap in gaps if isinstance(gap, dict)]
    if set(gap_ids) != EXPECTED_GAPS or len(gap_ids) != len(set(gap_ids)):
        errors.append("gaps must exactly match the frozen gap set")
    for gap in gaps:
        if not isinstance(gap, dict):
            errors.append("each gap must be an object")
            continue
        for field in (
            "adapter",
            "loss",
            "data_split",
            "verifier",
            "metrics",
            "holdout_isolation",
        ):
            if not gap.get(field):
                errors.append(f"gap {gap.get('id')} is missing {field}")

    stages = payload.get("stages")
    if not isinstance(stages, list):
        errors.append("stages must be a list")
        stages = []
    stage_ids = tuple(
        stage.get("id") for stage in stages if isinstance(stage, dict)
    )
    if stage_ids != EXPECTED_STAGES:
        errors.append("stage order must be parallel gaps -> router -> joint tuning")
    if len(stages) == 3:
        if stages[0].get("parallelizable") is not True:
            errors.append("stage 1 must be parallelizable")
        if stages[1].get("trainable") != "zero_gated_cross_attention_aggregator_only":
            errors.append("stage 2 must train only the aggregator")
        if stages[2].get("shared_trunk") != "low_learning_rate":
            errors.append("stage 3 must use a low-LR shared trunk")

    forbidden = set(payload.get("forbidden", []))
    for rule in (
        "test leakage into channel training or checkpoint selection",
        "JSON repair",
        "hidden retry",
        "best-of-N primary evaluation",
    ):
        if rule not in forbidden:
            errors.append(f"missing forbidden rule: {rule}")

    controls = payload.get("matched_controls", {})
    if "decomposed_parallel_frozen_router" not in controls.get("arms", []):
        errors.append("frozen-router matched control is required")
    if len(controls.get("matched_fields", [])) < 5:
        errors.append("matched control fields are incomplete")
    return tuple(errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/learning/objectives/nlp_gap_parallel_training.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_gap_router_audit_v1.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_contract(payload)
    report = {
        "version": "nlp-gap-parallel-channel-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "gap_count": len(payload.get("gaps", [])),
        "stage_count": len(payload.get("stages", [])),
        "evidence_boundary": (
            "CPU contract audit only; no channel or aggregate capability evidence."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
