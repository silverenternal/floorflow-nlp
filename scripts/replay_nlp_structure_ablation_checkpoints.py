#!/usr/bin/env python3
"""Replay frozen validation metrics from interrupted NLP checkpoints on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from floorflow.nlp_flow.prefix_policy_data import encode_prefix_rows
from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy
from floorflow.nlp_flow.prefix_tree_training import evaluate_prefix_policy
from floorflow.nlp_flow.real_corpus_action_data import iter_declaration_fragments
from floorflow.nlp_flow.structure_ablation_runner import (
    _completion_budgets,
    _free_running_metrics,
    _load_inputs,
    build_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def replay(config_path: Path, interruption_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    interrupted = json.loads(interruption_path.read_text(encoding="utf-8"))
    if interrupted.get("status") != "interrupted_for_project_line_gpu_isolation":
        raise ValueError("source report is not a recognized interruption report")
    config_sha = _sha(config_path)
    if interrupted["registration"]["config_sha256"] != config_sha:
        raise ValueError("interruption report and config identities differ")
    preflight = build_preflight(config_path, root=ROOT)
    loaded = _load_inputs(config, ROOT)
    train_batch = encode_prefix_rows(
        loaded["train_rows"], loaded["grammars"], loaded["vocabulary"]
    )
    validation_batch = encode_prefix_rows(
        loaded["validation_rows"], loaded["grammars"], loaded["vocabulary"]
    )
    budgets = _completion_budgets(config, loaded["grammars"], ROOT)
    contract = json.loads(
        (ROOT / config["shared"]["prompt_contract"]).read_text(encoding="utf-8")
    )
    fragments = list(
        iter_declaration_fragments(loaded["validation_sources"], contract["fragments"])
    )
    source_by_case = {str(row["case_id"]): str(row["source"]) for row in fragments}
    ast_by_case = {str(row["case_id"]): str(row["ast"]) for row in fragments}
    checkpoint_rows = [
        item for item in interrupted["runs"]
        if item["status"] in {
            "completed_checkpoint_metrics_not_flushed",
            "partial_checkpoint_interrupted_during_free_running",
        }
    ]
    arm_by_id = {arm["id"]: arm for arm in config["arms"]}
    result: dict[str, Any] = {
        "version": "nlp-structure-matched-cpu-checkpoint-replay-v1",
        "experiment_id": config["experiment_id"],
        "status": "in_progress",
        "source_interruption_report": str(interruption_path.relative_to(ROOT)),
        "source_interruption_report_sha256": _sha(interruption_path),
        "configuration_sha256": config_sha,
        "preflight": preflight,
        "device": "cpu",
        "gpu_used": False,
        "test_content_read": False,
        "selection_on_validation": False,
        "runs": [],
        "evidence_boundary": (
            "Deterministic replay of validation metrics from already trained, frozen checkpoints; "
            "no training, checkpoint reselection, or capability claim."
        ),
    }
    _write_atomic(output_path, result)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    for item in checkpoint_rows:
        checkpoint = ROOT / item["checkpoint"]
        if _sha(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed: {item['checkpoint']}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        arm_id = str(item["arm"])
        seed = int(item["seed"])
        arm = arm_by_id[arm_id]
        if payload.get("arm") != arm or payload.get("seed") != seed:
            raise ValueError(f"checkpoint arm/seed identity mismatch: {item['checkpoint']}")
        if payload.get("config_sha256") != config_sha:
            raise ValueError(f"checkpoint config identity mismatch: {item['checkpoint']}")
        if payload.get("train_actions_sha256") != preflight["inputs"]["train_actions"]["sha256"]:
            raise ValueError(f"checkpoint train-data identity mismatch: {item['checkpoint']}")
        model = PrefixTreePolicy(
            loaded["vocabulary"],
            hidden_size=int(config["shared"]["hidden_size"]),
            layers=int(config["shared"]["decoder_layers"]),
            heads=int(config["shared"]["attention_heads"]),
            max_steps=train_batch.action.shape[1],
            blueprint_conditioning=bool(arm["blueprint_conditioning"]),
        ).cpu()
        model.load_state_dict(payload["model"])
        model.eval()
        started = time.perf_counter()
        teacher_metrics = evaluate_prefix_policy(
            model,
            validation_batch,
            batch_rows=int(config["shared"]["batch_rows"]),
        )
        free_running = _free_running_metrics(
            model,
            loaded["validation_rows"],
            source_by_case,
            ast_by_case,
            loaded["grammars"],
            loaded["vocabulary"],
            device=torch.device("cpu"),
            arm=arm,
            completion_budgets=budgets,
            max_steps=int(config["shared"]["maximum_action_steps"]),
        )
        run = {
            "arm": arm_id,
            "seed": seed,
            "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "checkpoint_origin_status": item["status"],
            "teacher_forced_replay": teacher_metrics,
            "free_running_replay": free_running,
            "replay_wall_seconds": time.perf_counter() - started,
            "selection_on_validation": False,
        }
        result["runs"].append(run)
        result["status"] = "partial_replay"
        _write_atomic(output_path, result)
        del model, payload
    result["status"] = "replay_complete_for_available_checkpoints"
    result["completed_checkpoint_replays"] = len(result["runs"])
    result["not_replayed"] = [
        {"arm": item["arm"], "seed": item["seed"], "status": item["status"]}
        for item in interrupted["runs"] if item not in checkpoint_rows
    ]
    _write_atomic(output_path, result)
    return result


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
        "device": report["device"],
        "gpu_used": report["gpu_used"],
        "completed_checkpoint_replays": report.get("completed_checkpoint_replays", len(report["runs"])),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
