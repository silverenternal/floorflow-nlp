#!/usr/bin/env python3
"""CPU-only preflight and budget audit for the registered structured-head screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import audit_prefix_batch, encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.research.experiments.studies.structure_ablation_runner import _completion_budgets, _load_inputs

ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    data_path = ROOT / config["data_config"]
    if _sha(data_path) != config["data_config_sha256"]:
        errors.append("frozen matched data configuration hash changed")
    implementation = ROOT / config["implementation_config"]
    if _sha(implementation) != config["implementation_config_sha256"]:
        errors.append("head implementation configuration hash changed")
    queue_path = ROOT / config["gpu_queue"]
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registrations = [
        row for row in queue["experiments"]
        if row.get("experiment_id") == config["experiment_id"]
    ]
    if len(registrations) != 1:
        errors.append("GPU queue must contain exactly one registration for this screen")
    else:
        registration = registrations[0]
        if registration.get("config_sha256") != _sha(config_path):
            errors.append("queue and screen configuration hashes differ")
        if registration.get("seed_set") != [int(config["shared"]["seed"])]:
            errors.append("queue seed differs from the frozen screen seed")
        if registration.get("arms") != [arm["id"] for arm in config["arms"]]:
            errors.append("queue arms differ from the frozen screen arms")
        for name in ("max_gpu_hours_total", "max_optimizer_steps", "max_training_examples", "minimum_free_memory_mib"):
            if registration["budget"].get(name) != config["budget"].get(name):
                errors.append(f"queue budget field {name} differs from the screen config")

    data_config = json.loads(data_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(data_config, ROOT)
    budgets = _completion_budgets(data_config, inputs["grammars"], ROOT)
    minimums = {language: budget.minimum_root_nodes() for language, budget in budgets.items()}
    train = encode_prefix_rows(
        inputs["train_rows"], inputs["grammars"], inputs["vocabulary"], budget_minimums=minimums
    )
    validation = encode_prefix_rows(
        inputs["validation_rows"], inputs["grammars"], inputs["vocabulary"], budget_minimums=minimums
    )
    train_audit = audit_prefix_batch(train)
    validation_audit = audit_prefix_batch(validation)
    if train_audit["status"] != "pass" or validation_audit["status"] != "pass":
        errors.append("train or validation prefix batch failed its contract audit")

    shared = config["shared"]
    torch.manual_seed(int(shared["seed"]))
    parameter_counts = {}
    train_position_count = int(train.action.shape[1])
    for arm in config["arms"]:
        model = PrefixTreePolicy(
            inputs["vocabulary"],
            hidden_size=int(shared["hidden_size"]),
            layers=int(shared["decoder_layers"]),
            heads=int(shared["attention_heads"]),
            max_steps=train_position_count,
            blueprint_conditioning=bool(shared["blueprint_conditioning"]),
            node_kind_rank=arm["node_kind_rank"],
            budget_residual=bool(arm["budget_residual"]),
        )
        parameter_counts[arm["id"]] = sum(parameter.numel() for parameter in model.parameters())
    if max(parameter_counts.values()) > int(config["parameter_cap"]):
        errors.append("screen candidate exceeds the registered parameter cap")
    steps = ((len(inputs["train_rows"]) + int(shared["batch_rows"]) - 1) // int(shared["batch_rows"])) * int(shared["epochs"])
    total_steps = steps * len(config["arms"])
    total_examples = len(inputs["train_rows"]) * int(shared["epochs"]) * len(config["arms"])
    if total_steps > int(config["budget"]["max_optimizer_steps"]):
        errors.append("registered optimizer-step budget is insufficient")
    if total_examples > int(config["budget"]["max_training_examples"]):
        errors.append("registered training-example budget is insufficient")
    output_paths = (ROOT / config["report"], ROOT / config["checkpoint_dir"])
    if any(path.exists() for path in output_paths):
        errors.append("screen outputs already exist; no retries or checkpoint reuse are allowed")

    source_paths = (
        "src/newera/nlp_flow/components/research/training/efficiency/data_efficient_screen_runner.py",
        "src/newera/nlp_flow/components/research/architecture/experts/efficient_output_heads.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_policy_data.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_policy.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_runtime.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_training.py",
        "src/newera/nlp_flow/components/generation/trees/runtime/tree_completion_runtime.py",
        "project/scripts/experiments/probes/run_nlp_data_efficient_heads_screen.py",
        "project/scripts/evaluation/heads/preflight_nlp_data_efficient_heads_screen.py",
    )
    report = {
        "version": "nlp-data-efficient-heads-screen-preflight-v1",
        "status": "ready" if not errors else "blocked",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "queue": {"path": str(queue_path.relative_to(ROOT)), "sha256": _sha(queue_path),
                  "training_allowed_now": queue.get("training_allowed_now")},
        "data": {"train_rows": len(inputs["train_rows"]), "validation_fragments": len(inputs["validation_rows"]),
                 "minimum_budget_by_language": minimums, "train_batch_audit": train_audit,
                 "validation_batch_audit": validation_audit},
        "budget": {"steps_per_arm": steps, "total_steps": total_steps,
                   "total_examples": total_examples, "max_gpu_hours_total": config["budget"]["max_gpu_hours_total"]},
        "decoder_positions": {
            "train_batch_position_count": train_position_count,
            "free_running_generation_limit": int(shared["maximum_action_steps"]),
            "model_position_embedding_source": "train batch action width, matching train_prefix_tree_policy",
        },
        "parameter_counts": parameter_counts,
        "implementation_hashes": {path: _sha(ROOT / path) for path in source_paths},
        "gpu_used": False,
        "test_content_read": False,
        "errors": errors,
        "evidence_boundary": "CPU data, grammar, parameter and budget checks only; no quality evidence.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/models_training/learning/heads/nlp_data_efficient_heads_screen.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_data_efficient_heads_screen_preflight_v1.json")
    args = parser.parse_args()
    report = preflight(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
