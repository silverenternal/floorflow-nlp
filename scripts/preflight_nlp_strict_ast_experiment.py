#!/usr/bin/env python3
"""Preflight the strict AST experiment without allocating a device or launching work."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_device_rows(output: str) -> dict[int, tuple[int, int]]:
    devices = {}
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            index, free_mib, total_mib = (int(field) for field in fields)
        except ValueError:
            continue
        devices[index] = (free_mib, total_mib)
    return devices


def build_preflight(
    config: dict[str, Any],
    *,
    freeze: dict[str, Any],
    suite: dict[str, Any],
    training: dict[str, Any],
    queue: dict[str, Any],
    device_rows: dict[int, tuple[int, int]],
    root: Path = ROOT,
) -> dict[str, Any]:
    blockers = []
    if config.get("seed_set") != training.get("preflight", {}).get("seeds"):
        blockers.append("seed_set_differs_from_frozen_training_contract")
    if config.get("qualification", {}).get("per_case_thresholds") != "configs/nlp_legacy60_eval_manifest_v11.json":
        blockers.append("strict_gate_threshold_registry_changed")
    if suite.get("test_content_evaluated") is not False or suite.get("split") != "validation":
        blockers.append("strict_gate_manifest_is_not_validation_only")
    if queue.get("training_allowed_now") is not False or queue.get("quality_evaluation_allowed_now") is not False:
        blockers.append("gpu_queue_policy_is_not_closed")
    if config.get("execution", {}).get("launch_enabled") is not False:
        blockers.append("preflight_config_must_not_launch")
    if not freeze.get("artifacts") or freeze.get("status") != "passed":
        blockers.append("implementation_freeze_is_missing_or_failed")
    for relative, expected in freeze.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            blockers.append(f"implementation_fingerprint_changed:{relative}")
    preflight = training.get("preflight", {})
    corpus = freeze.get("fingerprints", {}).get("corpus", {})
    if preflight.get("train_actions_sha256") != corpus.get("training_actions_sha256"):
        blockers.append("training_corpus_fingerprint_differs_from_freeze")
    device = int(config["device_index"])
    free_memory = device_rows.get(device, (0, 0))[0]
    minimum_free = int(config["minimum_free_memory_mib"])
    if device not in device_rows:
        blockers.append("registered_gpu_is_unavailable")
    elif free_memory < minimum_free:
        blockers.append("registered_gpu_memory_is_in_use")
    return {
        "version": "nlp-strict-ast-preflight-v11",
        "task_id": config["task_id"],
        "status": "blocked" if blockers else "preflight_passed_no_execution",
        "blockers": blockers,
        "registered_device": device,
        "free_memory_mib": free_memory,
        "minimum_free_memory_mib": minimum_free,
        "registered_seed_set": config["seed_set"],
        "thresholds_sha256": _canonical_hash(suite.get("thresholds", {})),
        "training_enabled": False,
        "quality_evaluation_enabled": False,
        "launch_performed": False,
        "quality_claim": "none",
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _device_rows() -> dict[int, tuple[int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}
    return read_device_rows(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nlp_strict_ast_experiment_v11.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / config["implementation_freeze"]).read_text(encoding="utf-8"))
    suite = json.loads((ROOT / config["frozen_suite"]).read_text(encoding="utf-8"))
    training = json.loads((ROOT / config["training_config"]).read_text(encoding="utf-8"))
    queue = json.loads((ROOT / "configs/nlp_gpu_experiment_queue_v11.json").read_text(encoding="utf-8"))
    report = build_preflight(
        config,
        freeze=freeze,
        suite=suite,
        training=training,
        queue=queue,
        device_rows=_device_rows(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"blocked", "preflight_passed_no_execution"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
