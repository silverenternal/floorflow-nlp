#!/usr/bin/env python3
"""Audit benchmark-metric to internal-proxy and joint-loss coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def validate_alignment(
    payload: dict,
    metric_contract: dict,
) -> tuple[str, ...]:
    errors = []
    benchmarks = payload.get("external_benchmarks")
    proxies = payload.get("proxy_map")
    if not isinstance(benchmarks, list) or not benchmarks:
        return ("external_benchmarks must be non-empty",)
    if not isinstance(proxies, list) or not proxies:
        return ("proxy_map must be non-empty",)

    external_metrics = {
        metric
        for benchmark in benchmarks
        for metric in benchmark.get("metrics", [])
    }
    mapped_metrics = {
        proxy.get("external_metric")
        for proxy in proxies
        if isinstance(proxy, dict)
    }
    missing = external_metrics - mapped_metrics
    if missing:
        errors.append(f"external metrics lack proxies: {sorted(missing)}")

    internal_metrics = metric_contract.get("metrics", {})
    for proxy in proxies:
        if not isinstance(proxy, dict):
            errors.append("every proxy entry must be an object")
            continue
        for field in (
            "external_metric",
            "internal_metric",
            "loss",
            "status",
        ):
            if not proxy.get(field):
                errors.append(
                    f"proxy for {proxy.get('external_metric')} lacks {field}"
                )
        internal = proxy.get("internal_metric")
        loss = proxy.get("loss")
        if (
            internal
            and internal not in internal_metrics
            and loss != "governance_only_no_loss"
        ):
            errors.append(f"internal metric is not defined: {internal}")

    required_losses = set(payload.get("required_joint_loss_components", []))
    referenced_losses = {
        proxy.get("loss")
        for proxy in proxies
        if isinstance(proxy, dict)
        and proxy.get("loss") != "governance_only_no_loss"
    }
    absent_links = required_losses - referenced_losses
    if absent_links:
        errors.append(
            f"required joint losses are not linked to metrics: {sorted(absent_links)}"
        )

    checks = payload.get("alignment_checks", {})
    if checks.get("minimum_cases", 0) < 24:
        errors.append("alignment evaluation needs at least 24 benchmark families")
    statistics = set(checks.get("statistics", []))
    for statistic in (
        "spearman_rank_correlation",
        "kendall_tau",
        "pairwise_sign_agreement",
        "coverage",
    ):
        if statistic not in statistics:
            errors.append(f"alignment statistic is missing: {statistic}")
    return tuple(errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/evaluation/proxies/nlp_benchmark_proxy_alignment.json",
    )
    parser.add_argument(
        "--metric-contract",
        type=Path,
        default=ROOT / "project/configs/evaluation/metrics/nlp_metric_contract.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_benchmark_proxy_alignment.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    metric_contract = json.loads(
        args.metric_contract.read_text(encoding="utf-8")
    )
    errors = validate_alignment(payload, metric_contract)
    statuses = {}
    for proxy in payload["proxy_map"]:
        status = str(proxy["status"])
        statuses[status] = statuses.get(status, 0) + 1
    unresolved = [
        {
            "external_metric": proxy["external_metric"],
            "internal_metric": proxy["internal_metric"],
            "loss": proxy["loss"],
            "status": proxy["status"],
        }
        for proxy in payload["proxy_map"]
        if proxy["status"] not in {"implemented", "governance_gate"}
    ]
    report = {
        "version": "nlp-benchmark-proxy-alignment-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": list(errors),
        "external_benchmarks": len(payload["external_benchmarks"]),
        "external_metrics": len(
            {
                metric
                for benchmark in payload["external_benchmarks"]
                for metric in benchmark["metrics"]
            }
        ),
        "proxy_entries": len(payload["proxy_map"]),
        "proxy_status_counts": dict(sorted(statuses.items())),
        "unresolved_proxy_gaps": unresolved,
        "required_joint_loss_components": payload[
            "required_joint_loss_components"
        ],
        "alignment_status": "contract_complete_proxy_correlation_pending",
        "evidence_boundary": payload["evidence_boundary"],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
