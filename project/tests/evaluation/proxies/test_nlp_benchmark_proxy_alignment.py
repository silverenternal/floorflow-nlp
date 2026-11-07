from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _module():
    path = ROOT / "project/scripts/audit/evaluation/benchmarks/audit_nlp_benchmark_proxy_alignment.py"
    spec = importlib.util.spec_from_file_location("benchmark_proxy_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load benchmark proxy audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payloads() -> tuple[dict, dict]:
    alignment = json.loads(
        (
            ROOT / "project/configs/evaluation/proxies/nlp_benchmark_proxy_alignment.json"
        ).read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (ROOT / "project/configs/evaluation/metrics/nlp_metric_contract.json").read_text(
            encoding="utf-8"
        )
    )
    return alignment, metrics


def test_all_external_metrics_have_a_proxy() -> None:
    alignment, metrics = _payloads()

    assert _module().validate_alignment(alignment, metrics) == ()


def test_missing_external_proxy_is_rejected() -> None:
    alignment, metrics = _payloads()
    alignment = copy.deepcopy(alignment)
    alignment["proxy_map"] = [
        proxy
        for proxy in alignment["proxy_map"]
        if proxy["external_metric"] != "patch_apply_rate"
    ]

    errors = _module().validate_alignment(alignment, metrics)

    assert any("lack proxies" in error for error in errors)


def test_unlinked_joint_loss_component_is_rejected() -> None:
    alignment, metrics = _payloads()
    alignment = copy.deepcopy(alignment)
    alignment["required_joint_loss_components"].append(
        "L_unmapped_proxy"
    )

    errors = _module().validate_alignment(alignment, metrics)

    assert any("required joint losses" in error for error in errors)
