#!/usr/bin/env python3
"""Audit repository, Triton, Linux, and performance executor contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from newera.nlp_flow.components.generation.execution.backends.triton_executor import (
    score_linux_diagnostic,
    score_performance_patch,
)

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_executor_contract_audit_v1.json",
    )
    args = parser.parse_args()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "project/tests/other/contracts_runtime/executor_triton/test_nlp_triton_executor.py", "-q"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    linux = score_linux_diagnostic(
        {
            "diagnosis": "disk_exhaustion",
            "recommended_action": "rotate_logs_then_retry",
            "evidence": ["No space left on device"],
        },
        expected_diagnosis="disk_exhaustion",
        expected_action="rotate_logs_then_retry",
    )
    performance = score_performance_patch(
        baseline_latency_ms=100.0,
        candidate_latency_ms=105.0,
        correctness_passed=True,
    )
    errors = []
    if result.returncode != 0:
        errors.append("Triton/Linux executor tests failed")
    if not linux.passed:
        errors.append("Linux diagnostic golden case failed")
    if not performance.passed:
        errors.append("performance guard golden case failed")
    report = {
        "version": "nlp-executor-contract-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "triton_test_returncode": result.returncode,
        "triton_test_output": result.stdout[-1000:],
        "linux_golden_passed": linux.passed,
        "performance_guard_passed": performance.passed,
        "evidence_boundary": "Executor contract only; no generated-model capability result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
