#!/usr/bin/env python3
"""Preflight or run the single registered data-efficient-head GPU screen."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.research.training.efficiency.data_efficient_screen_runner import run_screen

ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _own_gpu_pids() -> set[int]:
    pids = {os.getpid()}
    status = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    nspid = next((line for line in status if line.startswith("NSpid:")), "")
    pids.update(map(int, nspid.split()[1:]))
    return pids


def _active_pids(uuid: str) -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        int(pid.strip())
        for line in result.stdout.splitlines()
        for gpu, pid in [line.split(",", 1)]
        if gpu.strip() == uuid
    }


def _device_info(device: torch.device, minimum_free_mib: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("registered head screen requires CUDA")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        index, name, uuid, driver, total, free = (part.strip() for part in line.split(",", 5))
        if int(index) != int(device.index or 0):
            continue
        if _active_pids(uuid) - _own_gpu_pids():
            raise BlockingIOError("selected GPU has another active compute process")
        if int(free) < minimum_free_mib:
            raise BlockingIOError("selected GPU has less than the registered free-memory minimum")
        return {
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "driver_version": driver,
            "total_memory_mib": int(total),
            "initial_free_memory_mib": int(free),
        }
    raise SystemExit(f"CUDA device {device} is missing from nvidia-smi")


def _wait_for_device(device: torch.device, minimum_free_mib: int) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    stable_since: float | None = None
    deadline = start + 1800
    while time.monotonic() < deadline:
        try:
            info = _device_info(device, minimum_free_mib)
        except BlockingIOError:
            stable_since = None
            time.sleep(0.5)
            continue
        if stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= 10:
            return info, time.monotonic() - start
        time.sleep(0.5)
    raise SystemExit("cuda:0 did not remain exclusively idle for 10 seconds within 30 minutes")


def _watch(uuid: str, stop: threading.Event, reason: list[str]) -> None:
    own = _own_gpu_pids()
    while not stop.wait(0.5):
        try:
            foreign = _active_pids(uuid) - own
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            reason.append(f"GPU ownership check failed: {type(error).__name__}: {error}")
            os.kill(os.getpid(), signal.SIGINT)
            return
        if foreign:
            reason.append(f"foreign compute process entered cuda:0: {sorted(foreign)}")
            os.kill(os.getpid(), signal.SIGINT)
            return


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _failure_report(config: dict[str, Any], error: BaseException, reason: list[str]) -> dict[str, Any]:
    path = ROOT / config["report"]
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report["status"] = "interrupted_or_failed_incomplete"
    report["stop_reason"] = reason[0] if reason else f"{type(error).__name__}: {error}"
    return report


def run(config_path: Path, preflight_path: Path, device_name: str) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    queue_path = ROOT / config["gpu_queue"]
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matches = [item for item in queue["experiments"] if item.get("experiment_id") == config["experiment_id"]]
    if len(matches) != 1 or queue.get("training_allowed_now") is not True:
        raise SystemExit("head screen is not the single enabled queue registration")
    registration = matches[0]
    if registration.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise SystemExit("queue registration and screen configuration hashes differ")
    if preflight_path.exists() is False:
        raise SystemExit("CPU preflight report is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready" or preflight["configuration"]["sha256"] != _sha(config_path):
        raise SystemExit("screen CPU preflight is blocked or bound to another configuration")
    if preflight["queue"].get("sha256") != _sha(queue_path):
        raise SystemExit("GPU queue changed after the screen CPU preflight")
    for relative_path, expected in preflight["implementation_hashes"].items():
        if _sha(ROOT / relative_path) != expected:
            raise SystemExit(f"screen implementation changed after CPU preflight: {relative_path}")
    lock_path = ROOT / config["gpu_lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another registered NLP GPU experiment holds the shared queue lock") from error
        device = torch.device(device_name)
        if device.type != "cuda" or int(device.index or 0) != 0:
            raise SystemExit("this registration is pinned to cuda:0")
        gpu, idle_wait = _wait_for_device(device, int(config["budget"]["minimum_free_memory_mib"]))
        torch.cuda.set_device(device)
        stop = threading.Event()
        reason: list[str] = []
        watcher = threading.Thread(target=_watch, args=(gpu["uuid"], stop, reason), daemon=True)
        watcher.start()
        try:
            report = run_screen(config_path, preflight_path, device)
        except KeyboardInterrupt as error:
            report = _failure_report(config, error, reason)
        except (SystemExit, RuntimeError, ValueError, OSError, TypeError, KeyError, IndexError) as error:
            report = _failure_report(config, error, reason)
        finally:
            stop.set()
            watcher.join(timeout=2)
        report["gpu"] = gpu
        report["exclusive_idle_wait_seconds"] = round(idle_wait, 2)
        report["peak_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        report["peak_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / (1024**2), 2)
        if reason:
            report["ownership_watchdog_reason"] = reason[0]
        _write_report(ROOT / config["report"], report)
        return 0 if report.get("status") == "completed_unqualified" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/models_training/learning/heads/nlp_data_efficient_heads_screen.json")
    parser.add_argument("--preflight", type=Path, default=ROOT / "outputs/reports/nlp_data_efficient_heads_screen_preflight_v1.json")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    return run(args.config.resolve(), args.preflight.resolve(), args.device)


if __name__ == "__main__":
    raise SystemExit(main())
