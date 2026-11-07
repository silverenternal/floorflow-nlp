#!/usr/bin/env python3
"""Validate frozen training inputs and gates without importing torch or allocating a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_preflight(config_path: Path, *, root: Path = ROOT) -> dict[str, object]:
    config = _read_json(config_path)
    preflight = config["preflight"]
    checks: dict[str, object] = {}
    errors: list[str] = []

    def input_path(key: str) -> Path:
        path = (root / str(preflight[key])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"preflight path escapes repository root: {key}") from error
        return path

    ledger_path = input_path("task_ledger")
    train_path = input_path("train_actions")
    validation_manifest_path = input_path("validation_manifest")
    grammar_path = input_path("grammar")
    queue_path = input_path("queue")
    for name, path in (
        ("task_ledger", ledger_path),
        ("train_actions", train_path),
        ("validation_manifest", validation_manifest_path),
        ("grammar", grammar_path),
        ("queue", queue_path),
    ):
        checks[f"{name}_exists"] = path.is_file()
        if not path.is_file():
            errors.append(f"missing {name}: {path.relative_to(root)}")

    if errors:
        return {
            "version": "nlp-implementation-preflight-v11",
            "status": "fail",
            "checks": checks,
            "errors": errors,
            "optimizer_started": False,
            "gpu_allocated": False,
        }

    for key, path, expected_key in (
        ("train_actions", train_path, "train_actions_sha256"),
        ("validation_manifest", validation_manifest_path, "validation_manifest_sha256"),
        ("grammar", grammar_path, "grammar_sha256"),
    ):
        actual = _sha256(path)
        expected = str(preflight[expected_key])
        checks[f"{key}_sha256_matches"] = actual == expected
        checks[f"{key}_sha256"] = actual
        if actual != expected:
            errors.append(f"{key} fingerprint changed")

    manifest = _read_json(validation_manifest_path)
    checks["validation_manifest_has_no_test_content"] = manifest.get("test_content_evaluated") is False
    if manifest.get("test_content_evaluated") is not False:
        errors.append("validation manifest does not affirm the test content boundary")
    checks["training_actions_replayed"] = manifest.get("training_action_replay_status") == "pass"
    if not checks["training_actions_replayed"]:
        errors.append("train action replay evidence is not passing")

    ledger = _read_json(ledger_path)
    tasks = {str(item["id"]): item for item in ledger.get("tasks", []) if isinstance(item, dict) and "id" in item}
    dependencies = tuple(str(item) for item in preflight["required_dependencies"])
    dependency_status = {task_id: tasks.get(task_id, {}).get("status", "missing") for task_id in dependencies}
    checks["dependency_status"] = dependency_status
    checks["dependencies_passed"] = all(status == "passed" for status in dependency_status.values())

    seeds = tuple(int(value) for value in preflight["seeds"])
    checks["seeds_valid"] = bool(seeds) and all(seed >= 0 for seed in seeds) and len(set(seeds)) == len(seeds)
    if not checks["seeds_valid"]:
        errors.append("seeds must be unique non-negative integers")
    checks["single_gpu_lock_available"] = not input_path("gpu_lock").exists()
    if not checks["single_gpu_lock_available"]:
        errors.append("the single-GPU lock is already held")
    if int(preflight["gpu_count"]) != 1:
        errors.append("only one GPU slot is permitted")
    queue = _read_json(queue_path)
    checks["queue_blocks_training"] = queue.get("training_allowed_now") is False
    if queue.get("training_allowed_now") is not False:
        errors.append("GPU queue must remain closed before the implementation freeze")

    no_error = not errors
    return {
        "version": "nlp-implementation-preflight-v11",
        "status": "pass_blocked_until_dependencies" if no_error and not checks["dependencies_passed"] else "pass" if no_error else "fail",
        "checks": checks,
        "errors": errors,
        "launch_allowed": bool(no_error and checks["dependencies_passed"]),
        "optimizer_started": False,
        "gpu_allocated": False,
        "training_or_quality_claim": False,
        "evidence_boundary": "Hash and task-gate preflight only; no optimizer, model, validation examples or GPU access.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/structured_generation/tree_action_corpus/nlp_tree_training_legacy_preflight.json")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_preflight(args.config.resolve())
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
