"""Preflight or execute the preregistered matched NLP structure ablation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
from pathlib import Path

import torch

from newera.nlp_flow.components.research.experiments.studies.structure_ablation_runner import build_preflight, run_experiment

ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _active_gpu_pids(gpu_uuid: str) -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        int(pid.strip())
        for line in result.stdout.splitlines()
        for uuid, pid in [line.split(",", 1)]
        if uuid.strip() == gpu_uuid
    }


def _watch_gpu_ownership(gpu_uuid: str, stop: threading.Event, reason: list[str]) -> None:
    while not stop.wait(0.5):
        try:
            foreign = _active_gpu_pids(gpu_uuid) - {os.getpid()}
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            reason.append(f"GPU ownership check failed: {type(error).__name__}: {error}")
            os.kill(os.getpid(), signal.SIGINT)
            return
        if foreign:
            reason.append(f"foreign compute process appeared on selected GPU: {sorted(foreign)}")
            os.kill(os.getpid(), signal.SIGINT)
            return


def _interruption_report(config_path: Path, queue_path: Path, reason: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    preflight_path = ROOT / config["preflight_report"]
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    train_path = ROOT / config["shared"]["train_actions"]
    runs = []
    for arm in config["arms"]:
        for seed in config["shared"]["seeds"]:
            base = ROOT / config["checkpoint_dir"] / arm["id"] / f"seed_{seed}"
            candidates = [
                (base.with_suffix(".pt"), "completed_checkpoint_metrics_not_flushed"),
                (base.with_name(base.name + ".partial.pt"), "partial_checkpoint_interrupted_during_free_running"),
            ]
            checkpoint = next(((path, status) for path, status in candidates if path.is_file()), None)
            if checkpoint is None:
                runs.append({
                    "arm": arm["id"],
                    "seed": seed,
                    "status": "incomplete_no_checkpoint",
                    "quality_status": "unqualified; exact phase not persisted",
                })
                continue
            path, status = checkpoint
            item: dict[str, object] = {
                "arm": arm["id"],
                "seed": seed,
                "status": status,
                "checkpoint": str(path.relative_to(ROOT)),
                "checkpoint_sha256": _sha(path),
                "checkpoint_bytes": path.stat().st_size,
                "quality_status": "unqualified; free-running metrics not flushed",
                "metadata_verified": False,
            }
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                item["metadata_verified"] = bool(
                    payload.get("arm") == arm
                    and payload.get("seed") == seed
                    and payload.get("config_sha256") == _sha(config_path)
                    and payload.get("train_actions_sha256") == _sha(train_path)
                )
                if not item["metadata_verified"]:
                    item["status"] = "checkpoint_metadata_mismatch"
            except (OSError, RuntimeError, ValueError, EOFError) as error:
                item["status"] = "checkpoint_unreadable"
                item["metadata_error"] = f"{type(error).__name__}: {error}"
            runs.append(item)
    registration = next(
        item for item in queue["experiments"]
        if item["experiment_id"] == config["experiment_id"]
    )
    return {
        "version": "nlp-structure-matched-interruption-v1",
        "experiment_id": config["experiment_id"],
        "status": "interrupted_for_project_line_gpu_isolation",
        "interruption": {"reason": reason, "checkpoint_reuse_allowed": False},
        "registration": {
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha(config_path),
            "queue": str(queue_path.relative_to(ROOT)),
            "queue_sha256": _sha(queue_path),
            "preflight": str(preflight_path.relative_to(ROOT)),
            "preflight_status": preflight["status"],
            "checkpoint_reuse_allowed": False,
            "retry_allowed": False,
            "seed_set": registration["seed_set"],
        },
        "device": "cuda:0",
        "training_split": "train",
        "metrics_split": "validation",
        "test_content_read": False,
        "validation_selection_allowed": False,
        "winner_selected": False,
        "counts": {
            "completed_checkpoints": sum(
                item["status"] == "completed_checkpoint_metrics_not_flushed" for item in runs
            ),
            "partial_checkpoints": sum(
                item["status"] == "partial_checkpoint_interrupted_during_free_running" for item in runs
            ),
            "unavailable_runs": sum("checkpoint" not in item for item in runs),
            "registered_runs": len(runs),
        },
        "runs": runs,
        "metrics_note": "No in-memory metrics were promoted; replay only hash-verified frozen checkpoints on CPU.",
        "evidence_boundary": config["evidence_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/structured_generation/structure_matched_strict/matched_structure/nlp_structure_matched_queued_four_arm_run.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        result = build_preflight(config_path, root=ROOT)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        default_report = ROOT / config.get(
            "preflight_report",
            "outputs/reports/nlp_structure_matched_preflight_v12.json",
        )
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        queue_path = ROOT / config["gpu_queue"]
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        registration = next(
            item for item in queue["experiments"]
            if item["experiment_id"] == config["experiment_id"]
        )
        if registration.get("device", "cuda:0") != args.device:
            raise SystemExit(
                f"registered experiment requires {registration.get('device', 'cuda:0')}"
            )
        lock_path = ROOT / queue["lock_path"]
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit("another registered NLP GPU experiment holds the queue lock") from error
            device = torch.device(args.device)
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise SystemExit("CUDA was requested but is not available")
                processes = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                gpu_info = {
                    int(index.strip()): (uuid.strip(), int(free.strip()))
                    for index, uuid, free in (
                        line.split(",", 2) for line in processes.stdout.splitlines()
                    )
                }
                compute = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                selected_uuid, free_mib = gpu_info[int(device.index or 0)]
                if any(line.split(",", 1)[0].strip() == selected_uuid for line in compute.stdout.splitlines()):
                    raise SystemExit("selected GPU has another active compute process")
                if free_mib < int(registration["budget"]["minimum_free_memory_mib"]):
                    raise SystemExit("selected GPU has less than the preregistered free-memory minimum")
                torch.cuda.set_device(device)
                guard_stop = threading.Event()
                guard_reason: list[str] = []
                guard = threading.Thread(
                    target=_watch_gpu_ownership,
                    args=(selected_uuid, guard_stop, guard_reason),
                    daemon=True,
                )
                guard.start()
                try:
                    result = run_experiment(config_path, root=ROOT, device=device)
                except KeyboardInterrupt:
                    guard_stop.set()
                    guard.join(timeout=2)
                    reason = (
                        guard_reason[0]
                        if guard_reason
                        else "operator SIGINT; no checkpoint reuse or retry"
                    )
                    result = _interruption_report(config_path, queue_path, reason)
                finally:
                    guard_stop.set()
                    guard.join(timeout=2)
            else:
                result = run_experiment(config_path, root=ROOT, device=device)
        default_report = ROOT / config["report"]
    report_path = args.report or default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report_path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
