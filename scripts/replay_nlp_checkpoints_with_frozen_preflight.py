#!/usr/bin/env python3
"""Replay interrupted checkpoints using their saved, registration-time preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import replay_nlp_structure_ablation_checkpoints as replay_module

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(config_path: Path, interruption_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    interrupted = json.loads(interruption_path.read_text(encoding="utf-8"))
    preflight_path = ROOT / config["preflight_report"]
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    config_sha = _sha(config_path)
    if preflight.get("status") != "ready":
        raise ValueError("registration-time preflight is not ready")
    if preflight["configuration"]["sha256"] != config_sha:
        raise ValueError("saved preflight and config identities differ")
    if interrupted["registration"]["config_sha256"] != config_sha:
        raise ValueError("interruption report and config identities differ")
    if preflight["gpu_queue"]["sha256"] != interrupted["registration"]["queue_sha256"]:
        raise ValueError("saved preflight and interruption registration queues differ")
    train_path = ROOT / config["shared"]["train_actions"]
    if _sha(train_path) != preflight["inputs"]["train_actions"]["sha256"]:
        raise ValueError("frozen training input differs from registration-time preflight")
    original_preflight = replay_module.build_preflight
    replay_module.build_preflight = lambda _config_path, root: preflight
    try:
        report = replay_module.replay(config_path, interruption_path, output_path)
    finally:
        replay_module.build_preflight = original_preflight
    report["frozen_preflight_reused"] = {
        "path": str(preflight_path.relative_to(ROOT)),
        "sha256": _sha(preflight_path),
        "queue_hash_matches_interruption_registration": True,
        "reason": "The historical queue was closed after interruption; use the immutable launch preflight for CPU replay.",
    }
    report["replay_driver"] = {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": _sha(Path(__file__).resolve()),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interruption-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.config.resolve(), args.interruption_report.resolve(), args.output.resolve())
    print(json.dumps({
        "status": report["status"],
        "experiment_id": report["experiment_id"],
        "completed_checkpoint_replays": len(report["runs"]),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
