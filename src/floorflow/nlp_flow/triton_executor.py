"""Bounded Triton and Linux diagnostic execution."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TritonCaseResult:
    returncode: int
    compiled: bool
    correct: bool
    max_abs_error: float
    shape: tuple[int, ...]
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return (
            self.returncode == 0
            and self.compiled
            and self.correct
            and not self.timed_out
        )


@dataclass(frozen=True)
class LinuxDiagnosticResult:
    diagnosis_match: bool
    action_match: bool
    evidence_present: bool

    @property
    def passed(self) -> bool:
        return self.diagnosis_match and self.action_match and self.evidence_present


@dataclass(frozen=True)
class PerformanceGuardResult:
    correct: bool
    latency_ratio: float
    within_guard: bool

    @property
    def passed(self) -> bool:
        return self.correct and self.within_guard


def _parse_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Triton case did not emit a JSON result line")


def execute_triton_script(
    script: Path,
    *,
    timeout_seconds: float,
    atol: float,
    rtol: float,
) -> TritonCaseResult:
    try:
        completed = subprocess.run(
            (sys.executable, str(script.resolve())),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired as error:
        return TritonCaseResult(
            -1,
            False,
            False,
            float("inf"),
            (),
            "" if error.stdout is None else str(error.stdout)[-8000:],
            "" if error.stderr is None else str(error.stderr)[-8000:],
            True,
        )
    try:
        payload = _parse_result(completed.stdout)
    except ValueError:
        payload = {}
    max_error = float(payload.get("max_abs_error", float("inf")))
    tolerance = float(atol) + float(rtol) * max(1.0, abs(max_error))
    correct = bool(payload.get("correct", False)) and max_error <= tolerance
    return TritonCaseResult(
        completed.returncode,
        bool(payload.get("compiled", False)),
        correct,
        max_error,
        tuple(int(value) for value in payload.get("shape", [])),
        completed.stdout[-8000:],
        completed.stderr[-8000:],
    )


def score_linux_diagnostic(
    prediction: dict[str, Any],
    *,
    expected_diagnosis: str,
    expected_action: str,
) -> LinuxDiagnosticResult:
    required = {"diagnosis", "recommended_action", "evidence"}
    if not required.issubset(prediction):
        return LinuxDiagnosticResult(False, False, False)
    evidence = prediction["evidence"]
    evidence_present = isinstance(evidence, list) and bool(evidence) and all(
        isinstance(item, str) and item for item in evidence
    )
    return LinuxDiagnosticResult(
        diagnosis_match=str(prediction["diagnosis"]) == expected_diagnosis,
        action_match=str(prediction["recommended_action"]) == expected_action,
        evidence_present=evidence_present,
    )


def score_performance_patch(
    *,
    baseline_latency_ms: float,
    candidate_latency_ms: float,
    correctness_passed: bool,
    max_latency_ratio: float = 1.1,
) -> PerformanceGuardResult:
    if baseline_latency_ms <= 0.0 or candidate_latency_ms <= 0.0:
        raise ValueError("latencies must be positive")
    if max_latency_ratio <= 0.0:
        raise ValueError("max_latency_ratio must be positive")
    ratio = float(candidate_latency_ms) / float(baseline_latency_ms)
    return PerformanceGuardResult(
        correct=bool(correctness_passed),
        latency_ratio=ratio,
        within_guard=ratio <= float(max_latency_ratio),
    )


__all__ = [
    "LinuxDiagnosticResult",
    "PerformanceGuardResult",
    "TritonCaseResult",
    "execute_triton_script",
    "score_linux_diagnostic",
    "score_performance_patch",
]
