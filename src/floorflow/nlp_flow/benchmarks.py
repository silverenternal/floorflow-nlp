"""Benchmark provenance and evaluation-only quarantine contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPECTED_BENCHMARKS = (
    "humaneval_plus",
    "mbpp_plus",
    "humaneval_python",
    "multipl_e_rust",
    "multipl_e_cpp",
)


def validate_benchmark_manifest(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("training_use") != "forbidden":
        errors.append("all coding benchmarks must be forbidden for training")
    quarantine = str(payload.get("quarantine_policy", "")).lower()
    if "evaluation_only" not in quarantine or "never tokenized" not in quarantine:
        errors.append("benchmarks need an evaluation-only tokenization quarantine")
    generation = payload.get("primary_generation")
    if not isinstance(generation, Mapping):
        errors.append("primary_generation must be an object")
    else:
        if generation.get("best_of_n_allowed") is not False:
            errors.append("best-of-N must be disabled")
        if generation.get("samples_per_task") != 1:
            errors.append("primary sampling must use one sample per task")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        errors.append("benchmarks must be a list")
        return tuple(errors)
    ids = tuple(row.get("id") for row in benchmarks if isinstance(row, Mapping))
    if ids != EXPECTED_BENCHMARKS:
        errors.append(f"benchmarks must be exactly {EXPECTED_BENCHMARKS}")
    for index, row in enumerate(benchmarks):
        if not isinstance(row, Mapping):
            errors.append(f"benchmarks[{index}] must be an object")
            continue
        for field in (
            "language",
            "dataset",
            "dataset_revision",
            "config",
            "source_url",
            "runner_repository",
            "runner_commit",
            "license",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                errors.append(f"benchmark {row.get('id')} lacks {field}")
        if row.get("split") != "test":
            errors.append(f"benchmark {row.get('id')} must use its frozen test split")
        if row.get("usage") != "evaluation_only":
            errors.append(f"benchmark {row.get('id')} must be evaluation-only")
        if (
            row.get("id", "").startswith("multipl_e")
            and "ML-training restriction" not in str(row.get("license", ""))
        ):
            errors.append(f"{row.get('id')} must preserve the MultiPL-E training restriction")
    languages = {row.get("language") for row in benchmarks if isinstance(row, Mapping)}
    if not {"python", "rust", "cpp"}.issubset(languages):
        errors.append("benchmark set must cover python, rust, and cpp")
    return tuple(errors)


def benchmark_protocol(payload: Mapping[str, Any]) -> dict[str, object]:
    errors = validate_benchmark_manifest(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "ids": tuple(row["id"] for row in payload["benchmarks"]),
        "training_use": payload["training_use"],
        "quarantine": payload["quarantine_policy"],
        "primary_generation": dict(payload["primary_generation"]),
    }


def validate_runner_manifest(
    payload: Mapping[str, Any],
    benchmark_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate frozen runner commands and benchmark alignment."""

    errors: list[str] = []
    if payload.get("network_during_execution") is not False:
        errors.append("benchmark execution must run without network")
    if payload.get("training_inputs_allowed") is not False:
        errors.append("benchmark runner outputs cannot become training inputs")
    benchmark_ids = tuple(
        row.get("id")
        for row in benchmark_manifest.get("benchmarks", [])
        if isinstance(row, Mapping)
    )
    runner_ids = tuple(
        row.get("benchmark_id")
        for row in payload.get("runners", [])
        if isinstance(row, Mapping)
    )
    if runner_ids != benchmark_ids:
        errors.append(f"runner ids must match benchmark ids: {benchmark_ids}")
    for index, runner in enumerate(payload.get("runners", [])):
        if not isinstance(runner, Mapping):
            errors.append(f"runners[{index}] must be an object")
            continue
        for field in ("runner_kind", "runner_version", "output_glob", "sandbox", "report_schema"):
            if not isinstance(runner.get(field), str) or not runner[field]:
                errors.append(f"runner {runner.get('benchmark_id')} lacks {field}")
        command = runner.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            errors.append(f"runner {runner.get('benchmark_id')} needs a command list")
        case_timeout = runner.get("case_timeout_seconds")
        suite_timeout = runner.get("suite_timeout_seconds")
        if not isinstance(case_timeout, int) or case_timeout <= 0:
            errors.append(f"runner {runner.get('benchmark_id')} needs a positive case timeout")
        if not isinstance(suite_timeout, int) or suite_timeout < (case_timeout or 0):
            errors.append(f"runner {runner.get('benchmark_id')} suite timeout is invalid")
        if runner.get("runner_kind") == "multipl_e_execution":
            command_text = " ".join(command or [])
            if "--network none" not in command_text:
                errors.append(f"runner {runner.get('benchmark_id')} must disable network")
    return tuple(errors)


__all__ = [
    "EXPECTED_BENCHMARKS",
    "benchmark_protocol",
    "validate_benchmark_manifest",
    "validate_runner_manifest",
]
