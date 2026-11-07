from __future__ import annotations

from pathlib import Path

import pytest
import torch

from newera.nlp_flow.components.generation.execution.backends.triton_executor import (
    execute_triton_script,
    score_linux_diagnostic,
    score_performance_patch,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_triton_executor_rejects_broken_result() -> None:
    result = execute_triton_script(
        FIXTURES / "nlp_triton_broken.py",
        timeout_seconds=10,
        atol=1e-4,
        rtol=1e-4,
    )

    assert not result.passed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_executor_numerical_equivalence() -> None:
    result = execute_triton_script(
        FIXTURES / "nlp_triton_add.py",
        timeout_seconds=120,
        atol=1e-4,
        rtol=1e-4,
    )

    assert result.compiled
    assert result.correct
    assert result.max_abs_error <= 1e-4


def test_linux_diagnostic_requires_exact_action_and_evidence() -> None:
    prediction = {
        "diagnosis": "disk_exhaustion",
        "recommended_action": "rotate_logs_then_retry",
        "evidence": ["No space left on device", "df shows 100% use"],
    }

    good = score_linux_diagnostic(
        prediction,
        expected_diagnosis="disk_exhaustion",
        expected_action="rotate_logs_then_retry",
    )
    bad = score_linux_diagnostic(
        {**prediction, "recommended_action": "reboot"},
        expected_diagnosis="disk_exhaustion",
        expected_action="rotate_logs_then_retry",
    )

    assert good.passed
    assert not bad.passed


def test_performance_patch_requires_correctness_and_latency_guard() -> None:
    passing = score_performance_patch(
        baseline_latency_ms=100.0,
        candidate_latency_ms=105.0,
        correctness_passed=True,
    )
    failing = score_performance_patch(
        baseline_latency_ms=100.0,
        candidate_latency_ms=120.0,
        correctness_passed=True,
    )
    incorrect = score_performance_patch(
        baseline_latency_ms=100.0,
        candidate_latency_ms=90.0,
        correctness_passed=False,
    )

    assert passing.passed
    assert not failing.passed
    assert not incorrect.passed
