"""Run the preregistered hierarchical tree experiment behind explicit gates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "project/scripts/experiments/research"))

from preflight_nlp_implementation import build_preflight

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_training import RolloutCurriculum
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import audit_prefix_batch, encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import train_prefix_tree_policy
from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device_rows() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    devices = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                index, free, total = (int(part.strip()) for part in line.split(","))
            except ValueError:
                continue
            devices[index] = (free, total)
    return devices


def _active_compute_devices() -> set[int]:
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if devices.returncode != 0 or processes.returncode != 0:
        return {-1}
    index_by_uuid = {}
    for line in devices.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            try:
                index_by_uuid[fields[1]] = int(fields[0])
            except ValueError:
                continue
    active = set()
    for line in processes.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] in index_by_uuid:
            active.add(index_by_uuid[fields[0]])
    return active


def audit_exact_training_validation_overlap(
    training_rows: list[dict[str, str]], validation_manifest: dict[str, Any]
) -> dict[str, Any]:
    validation = validation_manifest.get("cases", [])
    validation_by_hash = {
        str(row["gold_source_sha256"]): str(row["case_id"])
        for row in validation
        if isinstance(row, dict) and row.get("gold_source_sha256")
    }
    if len(validation_by_hash) != len(validation):
        raise ValueError("frozen validation manifest has duplicate or missing source hashes")
    if any(row.get("split") != "train" for row in training_rows):
        raise ValueError("exact overlap audit accepts train rows only")
    matches = [
        {"train_case_id": row["case_id"], "validation_case_id": validation_by_hash[row["source_sha256"]]}
        for row in training_rows
        if row["source_sha256"] in validation_by_hash
    ]
    return {
        "status": "pass" if not matches else "blocked",
        "training_rows": len(training_rows),
        "validation_rows": len(validation),
        "exact_source_hash_matches": matches,
        "split_boundary": "training rows and frozen validation manifest only; test rows are not parsed",
    }


def _read_split_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) != expected_split:
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError("split metadata mismatch during exact overlap audit")
            rows.append(
                {
                    "case_id": str(row["case_id"]),
                    "split": expected_split,
                    "source_sha256": str(row["target_source_sha256"]),
                }
            )
    return rows


def _read_action_splits(path: Path, allowed_splits: set[str]) -> list[dict[str, Any]]:
    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) not in allowed_splits:
                continue
            row = json.loads(line)
            if row.get("split") != match.group(1):
                raise ValueError("split metadata mismatch while loading training actions")
            rows.append(row)
    return rows


def _budget_estimate(root: Path, training: dict[str, Any], experiment: dict[str, Any]) -> dict[str, int]:
    rows, _validation, _filter_report = _load_rows(root, training)
    epochs = int(training["model"]["epochs"])
    batch_rows = int(training["model"]["batch_rows"])
    seed_count = len(experiment["seed_set"])
    actions_per_epoch = sum(len(row["actions"]) for row in rows)
    return {
        "training_rows": len(rows),
        "registered_seeds": seed_count,
        "epochs": epochs,
        "expected_examples": len(rows) * epochs * seed_count,
        "expected_optimizer_steps": ((len(rows) + batch_rows - 1) // batch_rows) * epochs * seed_count,
        "expected_action_tokens": actions_per_epoch * epochs * seed_count,
    }


def build_launch_preflight(root: Path = ROOT) -> dict[str, Any]:
    training_path = root / "project/configs/structured_generation/tree_action_corpus/nlp_tree_training_corrected_contract.json"
    training = _read(training_path)
    preflight = training["preflight"]
    config_path = root / str(preflight["experiment_config"])
    experiment = _read(config_path)
    queue_path = root / str(preflight["queue"])
    queue = _read(queue_path)
    freeze = _read(root / experiment["implementation_freeze"])
    base = build_preflight(training_path, root=root)
    blockers = []
    validation_manifest = _read(root / str(experiment["frozen_suite"]))
    training_rows = []
    training_rows.extend(_read_split_rows(root / str(preflight["legacy_training_corpus"]), "train"))
    training_rows.extend(_read_split_rows(root / str(preflight["train_actions"]), "train"))
    exact_overlap = audit_exact_training_validation_overlap(training_rows, validation_manifest)
    if exact_overlap["status"] != "pass":
        blockers.append("exact_train_validation_source_overlap")
    for path_key, digest_key in (
        ("legacy_training_corpus", "legacy_training_corpus_sha256"),
        ("legacy_grammar", "legacy_grammar_sha256"),
    ):
        path = root / str(preflight[path_key])
        if not path.is_file() or _sha(path) != preflight[digest_key]:
            blockers.append(f"frozen_input_fingerprint_mismatch:{path_key}")
    registered = [row for row in queue.get("experiments", []) if row.get("task_id") == experiment.get("task_id")]
    if not registered:
        blockers.append("experiment_not_registered_in_gpu_queue")
    elif (
        registered[0].get("seed_set") != experiment.get("seed_set")
        or registered[0].get("budget") != experiment.get("budget")
        or registered[0].get("status") != "registered"
    ):
        blockers.append("gpu_queue_registration_differs_from_preregistered_experiment")
    if queue.get("training_allowed_now") is not True:
        blockers.append("training_not_authorized_by_gpu_queue")
    if experiment.get("execution", {}).get("training_enabled") is not True:
        blockers.append("training_disabled_in_experiment_config")
    if experiment.get("execution", {}).get("launch_enabled") is not True:
        blockers.append("experiment_launch_disabled")
    if not base.get("checks", {}).get("dependencies_passed"):
        blockers.append("implementation_dependencies_not_passed")
    base_checks = base.get("checks", {})
    required_base_checks = (
        "train_actions_sha256_matches",
        "validation_manifest_sha256_matches",
        "grammar_sha256_matches",
        "validation_manifest_has_no_test_content",
        "training_actions_replayed",
    )
    if any(base_checks.get(name) is not True for name in required_base_checks):
        blockers.append("implementation_input_contract_failed")
    if freeze.get("status") != "passed":
        blockers.append("implementation_freeze_not_passed")
    estimate = _budget_estimate(root, training, experiment)
    budget = experiment.get("budget", {})
    for key, estimate_key in (
        ("max_training_examples", "expected_examples"),
        ("max_optimizer_steps", "expected_optimizer_steps"),
        ("max_action_tokens", "expected_action_tokens"),
    ):
        if int(budget.get(key, 0)) < estimate[estimate_key]:
            blockers.append(f"preregistered_budget_exceeded:{key}")
    lock_path = root / str(preflight["gpu_lock"])
    if lock_path.exists():
        blockers.append("single_gpu_lock_already_held")
    if (root / str(preflight["attempt_marker"])).exists():
        blockers.append("registered_single_attempt_already_consumed")
    device = int(experiment["device_index"])
    free_memory = _device_rows().get(device, (0, 0))[0]
    active_devices = _active_compute_devices()
    if device in active_devices or -1 in active_devices:
        blockers.append("registered_gpu_has_an_active_or_unverifiable_compute_process")
    if free_memory < int(experiment["minimum_free_memory_mib"]):
        blockers.append("registered_gpu_below_free_memory_threshold")
    for relative, expected in freeze.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            blockers.append(f"implementation_freeze_fingerprint_mismatch:{relative}")
    return {
        "version": "nlp-hierarchical-tree-launch-preflight-v11",
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "implementation_gate": {
            "status": "pass" if not any(name.startswith("implementation_") for name in blockers) else "blocked",
            "checks": {
                key: base_checks.get(key)
                for key in (*required_base_checks, "dependency_status", "dependencies_passed")
            },
            "evidence_scope": "Frozen implementation dependencies and input identities; run authorization is checked separately.",
        },
        "exact_source_overlap": exact_overlap,
        "budget_estimate": estimate,
        "preregistered_budget": budget,
        "registered_device": device,
        "free_memory_mib": free_memory,
        "active_compute_devices": sorted(active_devices),
        "minimum_free_memory_mib": int(experiment["minimum_free_memory_mib"]),
        "launch_performed": False,
    }


def _load_rows(
    root: Path, training: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    preflight = training["preflight"]
    corpus_path = root / str(preflight["legacy_training_corpus"])
    supplement_path = root / str(preflight["train_actions"])
    corpus_rows = _read_action_splits(corpus_path, {"train", "validation"})
    supplement_rows = _read_action_splits(supplement_path, {"train"})
    candidates = [row for row in corpus_rows + supplement_rows if row.get("split") == "train"]
    validation = [row for row in corpus_rows if row.get("split") == "validation"]
    max_nodes = int(training["data_filter"]["max_nodes"])
    train = []
    excluded = []
    for row in candidates:
        node_count = sum(action.get("type") == "open_node" for action in row["actions"])
        if node_count > max_nodes:
            excluded.append(
                {"case_id": row["case_id"], "language": row["language"], "node_count": node_count}
            )
        else:
            train.append(row)
    if len(validation) != 60 or any(row.get("split") != "train" for row in train):
        raise ValueError("training and frozen validation split boundary is invalid")
    validation_overflow = [
        str(row["case_id"])
        for row in validation
        if sum(action.get("type") == "open_node" for action in row["actions"]) > max_nodes
    ]
    if validation_overflow:
        raise ValueError("frozen validation cases exceed the registered node cap")
    return train, validation, {
        "maximum_nodes": max_nodes,
        "candidate_train_rows": len(candidates),
        "used_train_rows": len(train),
        "excluded_oversize_rows": excluded,
        "validation_rows_over_cap": validation_overflow,
        "policy": training["data_filter"]["oversize_train_policy"],
    }


def execute_experiment(root: Path = ROOT) -> dict[str, Any]:
    gate = build_launch_preflight(root)
    if gate["status"] != "ready":
        raise RuntimeError("experiment launch is blocked: " + ", ".join(gate["blockers"]))
    training_path = root / "project/configs/structured_generation/tree_action_corpus/nlp_tree_training_corrected_contract.json"
    training = _read(training_path)
    experiment = _read(root / str(training["preflight"]["experiment_config"]))
    preflight = training["preflight"]
    lock_path = root / str(preflight["gpu_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = lock_path.open("x", encoding="utf-8")
    try:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        device_index = int(experiment["device_index"])
        free_memory = _device_rows().get(device_index, (0, 0))[0]
        if device_index in _active_compute_devices():
            raise RuntimeError("registered CUDA device has another active compute process")
        if free_memory < int(experiment["minimum_free_memory_mib"]):
            raise RuntimeError("registered CUDA device became busy after preflight")
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError("registered CUDA device is unavailable")
        train_rows, validation_rows, training_filter = _load_rows(root, training)
        grammar_path = root / str(preflight["grammar"])
        grammars = {
            language: load_ast_grammar(grammar_path, language)
            for language in ("python", "rust", "cpp")
        }
        vocabulary = build_vocabulary(train_rows, grammars)
        train_batch = encode_prefix_rows(train_rows, grammars, vocabulary)
        validation_batch = encode_prefix_rows(validation_rows, grammars, vocabulary)
        batch_audits = {
            "train": audit_prefix_batch(train_batch),
            "validation": audit_prefix_batch(validation_batch),
        }
        if any(report["status"] != "pass" for report in batch_audits.values()):
            raise ValueError("encoded training or validation batch violates the registered tensor contract")
        model_config = training["model"]
        budget = experiment["budget"]
        device = torch.device(f"cuda:{device_index}")
        output_dir = root / str(preflight["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        attempt_marker = root / str(preflight["attempt_marker"])
        attempt_marker.parent.mkdir(parents=True, exist_ok=True)
        with attempt_marker.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "task_id": experiment["task_id"],
                    "status": "launch_started",
                    "seed_set": training["preflight"]["seeds"],
                    "budget": budget,
                    "retry_allowed": False,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        seed_reports = []
        import time

        total_started = time.monotonic()
        for seed in training["preflight"]["seeds"]:
            remaining_total = float(budget["max_gpu_hours_total"]) * 3600 - (
                time.monotonic() - total_started
            )
            if remaining_total <= 0:
                raise RuntimeError("total registered GPU-hour budget is exhausted")
            import signal

            seed_started = time.monotonic()
            seed_limit = min(float(budget["max_gpu_hours_per_seed"]) * 3600, remaining_total)
            signal.setitimer(signal.ITIMER_REAL, seed_limit)
            try:
                model, metrics = train_prefix_tree_policy(
                    train_batch,
                    validation_batch,
                    vocabulary,
                    device=device,
                    seed=int(seed),
                    epochs=int(model_config["epochs"]),
                    batch_rows=int(model_config["batch_rows"]),
                    hidden_size=int(model_config["hidden_size"]),
                    layers=int(model_config["decoder_layers"]),
                    heads=int(model_config["decoder_heads"]),
                    learning_rate=float(model_config["learning_rate"]),
                    warmup_steps=int(model_config["warmup_steps"]),
                    gradient_clip_norm=float(model_config["gradient_clip_norm"]),
                    loss_weights=training["objective"]["weights"],
                    train_rows=train_rows,
                    grammars=grammars,
                    rollout_curriculum=RolloutCurriculum(
                        warmup_steps=int(training["curriculum"]["warmup_steps"]),
                        decay_steps=int(training["curriculum"]["decay_steps"]),
                        initial_teacher_probability=float(
                            training["curriculum"]["initial_teacher_probability"]
                        ),
                        final_teacher_probability=float(
                            training["curriculum"]["final_teacher_probability"]
                        ),
                    ),
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            metrics["seed_wall_seconds"] = time.monotonic() - seed_started
            checkpoint_path = output_dir / f"seed_{int(seed)}.pt"
            torch.save(
                {
                    "version": "nlp-hierarchical-tree-v11",
                    "model": model.state_dict(),
                    "vocabulary": vocabulary,
                    "metrics": metrics,
                    "config": model_config,
                    "seed": int(seed),
                    "training_actions_sha256": _sha(root / str(preflight["train_actions"])),
                    "legacy_corpus_sha256": _sha(root / str(preflight["legacy_training_corpus"])),
        "grammar_sha256": _sha(grammar_path),
                },
                checkpoint_path,
            )
            seed_reports.append(
                {
                    "seed": int(seed),
                    "checkpoint": str(checkpoint_path.relative_to(root)),
                    "checkpoint_sha256": _sha(checkpoint_path),
                    "teacher_forced_validation": metrics,
                }
            )
        report = {
            "version": "nlp-hierarchical-tree-training-v11",
            "status": "training_completed_quality_unverified",
            "device": str(device),
            "seed_reports": seed_reports,
            "total_wall_seconds": time.monotonic() - total_started,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "training_filter": training_filter,
            "encoded_batch_audits": batch_audits,
            "training_split_only": True,
            "test_content_read": False,
            "oracle_input": False,
            "repair_or_retry": False,
            "quality_claim": "none",
            "evidence_boundary": "Train-only teacher and free-running supervision; strict target-free generation gate is separate.",
        }
        attempt_marker.write_text(
            json.dumps(
                {
                    "task_id": experiment["task_id"],
                    "status": "completed",
                    "seed_set": training["preflight"]["seeds"],
                    "budget": budget,
                    "retry_allowed": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        descriptor.close()
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_strict_ast_training_v12.json")
    args = parser.parse_args()
    try:
        result = build_launch_preflight() if args.dry_run else execute_experiment()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        result = {
            "version": "nlp-hierarchical-tree-training-v11",
            "status": "blocked",
            "error": str(error),
            "launch_performed": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] not in {"ready", "training_completed_quality_unverified"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
