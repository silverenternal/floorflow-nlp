#!/usr/bin/env python3
"""Audit the nlp_flow engineering template without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.engineering import (
    ExperimentSpec,
    audit_parameter_budget,
    build_jsonl_manifest,
)
from floorflow.nlp_flow.engineering.spec import canonical_json

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_engineering_p2m_control_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_json(args.config)
    data = spec.payload["data"]
    manifest = build_jsonl_manifest(
        {
            split: ROOT / data[split]
            for split in ("train", "validation", "test")
        }
    )
    manifest_sha256 = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    model = spec.payload["model"]
    budget = audit_parameter_budget(
        int(model["target_parameters"]),
        int(model["planned_corpus_tokens"]),
    )
    blockers = []
    if model.get("tokenizer") is None:
        blockers.append("tokenizer hash is pending")
    if spec.payload["evaluation"].get("benchmark_manifest") is None:
        blockers.append("benchmark manifest is pending")
    if spec.payload["runtime"].get("gpu_authorized") is not True:
        blockers.append("GPU training is not authorized")
    report = {
        "version": "nlp-engineering-audit-v1",
        "status": "pass",
        "training_ready": not blockers,
        "blockers": blockers,
        "config": str(args.config.relative_to(ROOT)),
        "experiment": spec.manifest(),
        "data_manifest": manifest,
        "data_manifest_sha256": manifest_sha256,
        "budget_projection": {
            "parameters": budget.parameters,
            "corpus_tokens": budget.corpus_tokens,
            "actual_tokens_per_parameter": budget.tokens_per_parameter,
            "tolerance": budget.tolerance,
            "passed": budget.passed,
        },
        "evidence_boundary": "Engineering contract only; no model training or quality result.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
