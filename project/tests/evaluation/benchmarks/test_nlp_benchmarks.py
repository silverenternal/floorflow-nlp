from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.research.evaluation.manifests.benchmarks import (
    EXPECTED_BENCHMARKS,
    benchmark_protocol,
    validate_benchmark_manifest,
    validate_runner_manifest,
)

ROOT = Path(__file__).resolve().parents[4]


def load_manifest() -> dict:
    return json.loads((ROOT / "project/configs/evaluation/benchmarks/nlp_benchmark_manifest.json").read_text())


def load_runners() -> dict:
    return json.loads((ROOT / "project/configs/evaluation/benchmarks/nlp_benchmark_runners.json").read_text())


def test_benchmark_registry_covers_all_primary_languages() -> None:
    payload = load_manifest()

    assert validate_benchmark_manifest(payload) == ()
    assert benchmark_protocol(payload)["ids"] == EXPECTED_BENCHMARKS


def test_benchmark_manifest_rejects_training_use() -> None:
    payload = load_manifest()
    payload["training_use"] = "allowed"

    errors = validate_benchmark_manifest(payload)

    assert "all coding benchmarks must be forbidden for training" in errors


def test_multipl_e_keeps_training_restriction() -> None:
    payload = load_manifest()
    rust = next(row for row in payload["benchmarks"] if row["id"] == "multipl_e_rust")
    rust["license"] = "MIT"

    errors = validate_benchmark_manifest(payload)

    assert "multipl_e_rust must preserve the MultiPL-E training restriction" in errors


def test_benchmark_manifest_rejects_best_of_n() -> None:
    payload = load_manifest()
    payload["primary_generation"]["best_of_n_allowed"] = True

    errors = validate_benchmark_manifest(payload)

    assert "best-of-N must be disabled" in errors


def test_runner_manifest_freezes_every_benchmark() -> None:
    assert validate_runner_manifest(load_runners(), load_manifest()) == ()


def test_multipl_e_runner_requires_network_isolation() -> None:
    runners = load_runners()
    rust = next(row for row in runners["runners"] if row["benchmark_id"] == "multipl_e_rust")
    rust["command"] = [value for value in rust["command"] if value != "none"]

    errors = validate_runner_manifest(runners, load_manifest())

    assert "runner multipl_e_rust must disable network" in errors
