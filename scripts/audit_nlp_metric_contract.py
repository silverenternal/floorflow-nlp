#!/usr/bin/env python3
"""Audit deterministic metrics, attacks, and the frozen SOTA comparison set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.benchmark_metrics import attack_report, validate_metric_contract

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "configs/nlp_metric_contract_v1.json",
    )
    parser.add_argument(
        "--sota",
        type=Path,
        default=ROOT / "configs/nlp_sota_band_contract_v1.json",
    )
    parser.add_argument("--attack-output", type=Path, default=None)
    parser.add_argument("--sota-output", type=Path, default=None)
    args = parser.parse_args()
    metric_payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    sota_payload = json.loads(args.sota.read_text(encoding="utf-8"))
    errors = list(validate_metric_contract(metric_payload))
    if sota_payload.get("primary_parameter_cap") != 23_000_000:
        errors.append("SOTA contract primary cap must equal 23M")
    if sota_payload.get("comparison_class", "").find("<= 23M") < 0:
        errors.append("SOTA comparison class must be bounded at 23M")
    rule = sota_payload.get("sota_band_rule", {})
    if rule.get("per_track_absolute_tolerance") is None:
        errors.append("SOTA band tolerance must be frozen")
    if rule.get("large_model_comparisons") != "out-of-class reference only":
        errors.append("large-model comparisons must stay out of class")
    attack = attack_report()
    if attack["status"] != "pass":
        errors.append("metric attack failed")
    attack["errors"] = list(errors)
    attack["status"] = "pass" if not errors else "fail"
    sota_report = {
        "version": "nlp-sota-band-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "contract": str(args.sota.relative_to(ROOT)),
        "comparison_class": sota_payload["comparison_class"],
        "tolerance": rule.get("per_track_absolute_tolerance"),
        "evidence_boundary": sota_payload["evidence_boundary"],
    }
    if args.attack_output is not None:
        args.attack_output.parent.mkdir(parents=True, exist_ok=True)
        args.attack_output.write_text(json.dumps(attack, indent=2) + "\n", encoding="utf-8")
    if args.sota_output is not None:
        args.sota_output.parent.mkdir(parents=True, exist_ok=True)
        args.sota_output.write_text(json.dumps(sota_report, indent=2) + "\n", encoding="utf-8")
    if args.attack_output is None and args.sota_output is None:
        print(json.dumps({"metric_attack": attack, "sota_band": sota_report}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
