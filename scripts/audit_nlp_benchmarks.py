#!/usr/bin/env python3
"""Audit benchmark provenance and evaluation-only quarantine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.benchmarks import (
    validate_benchmark_manifest,
    validate_runner_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/nlp_benchmark_manifest_v1.json",
    )
    parser.add_argument(
        "--runners",
        type=Path,
        default=ROOT / "configs/nlp_benchmark_runners_v1.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    runner_payload = json.loads(args.runners.read_text(encoding="utf-8"))
    errors = list(validate_benchmark_manifest(payload))
    errors.extend(validate_runner_manifest(runner_payload, payload))
    report = {
        "version": "nlp-benchmark-registry-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "manifest": str(args.manifest.relative_to(ROOT)),
        "runners": str(args.runners.relative_to(ROOT)),
        "benchmarks": [row["id"] for row in payload["benchmarks"]],
        "training_use": payload["training_use"],
        "quarantine_policy": payload["quarantine_policy"],
        "runners_frozen": not errors,
        "evidence_boundary": "Source and runner registry only; contamination and license execution gates remain pending.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
