#!/usr/bin/env python3
"""Run the registered one-seed CodeT5-small Python generation diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.synthetic_tree_policy import build_synthetic_tasks

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _own_gpu_pids() -> set[int]:
    pids = {os.getpid()}
    status = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    namespace_line = next((line for line in status if line.startswith("NSpid:")), "")
    pids.update(map(int, namespace_line.split()[1:]))
    return pids


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _device_uuid(device: torch.device) -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        if int(index) == int(device.index or 0):
            return uuid
    raise RuntimeError(f"CUDA device {device} is absent from nvidia-smi")


def _wait_for_idle_gpu(
    gpu_uuid: str, device: torch.device, minimum_free_mib: int,
    stable_seconds: float = 10, timeout_seconds: float = 1800,
) -> float:
    started = time.monotonic()
    idle_since: float | None = None
    own_pids = _own_gpu_pids()
    while time.monotonic() - started < timeout_seconds:
        foreign = _active_gpu_pids(gpu_uuid) - own_pids
        if foreign:
            idle_since = None
        elif idle_since is None:
            idle_since = time.monotonic()
        elif time.monotonic() - idle_since >= stable_seconds:
            try:
                _check_device(device, minimum_free_mib)
                return time.monotonic() - started
            except SystemExit as error:
                if "another active compute process" not in str(error):
                    raise
                idle_since = None
        time.sleep(0.5)
    raise SystemExit("cuda:0 did not remain exclusively idle for 10 seconds within 30 minutes")


def _watch_gpu_ownership(gpu_uuid: str, stop: threading.Event, reason: list[str]) -> None:
    own_pids = _own_gpu_pids()
    while not stop.wait(0.5):
        try:
            foreign = _active_gpu_pids(gpu_uuid) - own_pids
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            reason.append(f"GPU ownership check failed: {type(error).__name__}: {error}")
            os.kill(os.getpid(), signal.SIGINT)
            return
        if foreign:
            reason.append(f"foreign compute process appeared on selected GPU: {sorted(foreign)}")
            os.kill(os.getpid(), signal.SIGINT)
            return


def _check_device(device: torch.device, minimum_free_mib: int) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the registered pretrained pilot requires an available CUDA device")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = {
        int(index.strip()): (uuid.strip(), int(free.strip()))
        for index, uuid, free in (line.split(",", 2) for line in query.stdout.splitlines())
    }
    selected_uuid, free_mib = devices[int(device.index or 0)]
    if _active_gpu_pids(selected_uuid) - _own_gpu_pids():
        raise SystemExit("selected GPU has another active compute process")
    if free_mib < minimum_free_mib:
        raise SystemExit("selected GPU has less than the preregistered free-memory minimum")
    return selected_uuid


def _gpu_metadata(gpu_uuid: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        index, name, uuid, driver, total, free = (part.strip() for part in line.split(",", 5))
        if uuid == gpu_uuid:
            return {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "total_memory_mib": int(total),
                "initial_free_memory_mib": int(free),
            }
    raise RuntimeError(f"selected GPU {gpu_uuid} disappeared from nvidia-smi")


def _tokenize_batch(tokenizer: Any, prompts: list[str], targets: list[str], config: dict[str, Any]):
    source = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=int(config["training"]["max_source_tokens"]),
        return_tensors="pt",
    )
    target = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=int(config["training"]["max_target_tokens"]),
        return_tensors="pt",
    )
    labels = target["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return source, labels


def _execute(source: str, function_name: str, expected: int) -> tuple[bool, str | None]:
    harness = (
        "import sys\n"
        "namespace = {}\n"
        "exec(sys.stdin.read(), namespace)\n"
        "print(namespace[sys.argv[1]](5))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", harness, function_name],
            input=source,
            text=True,
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, type(error).__name__
    if result.returncode != 0:
        return False, result.stderr[-1000:]
    return result.stdout.strip() == str(expected), None


def _evaluate(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    baseline_rows: dict[str, dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    deadline: float,
    on_case: Any | None = None,
) -> list[dict[str, Any]]:
    outputs = []
    generation = config["training"]["generation"]
    model.eval()
    for task in tasks:
        if time.monotonic() >= deadline:
            raise TimeoutError("two-hour pilot budget reached during validation")
        encoded = tokenizer(
            task.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(config["training"]["max_source_tokens"]),
        ).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=bool(generation["do_sample"]),
                num_beams=int(generation["num_beams"]),
                max_new_tokens=int(generation["max_new_tokens"]),
            )
        latency = (time.perf_counter() - started) * 1000.0
        source = tokenizer.decode(generated[0], skip_special_tokens=True)
        try:
            parse_ast_graph(source, "python")
            parser_valid = True
            parse_error = None
        except (TypeError, ValueError, UnicodeError) as error:
            parser_valid = False
            parse_error = str(error)
        executed, execution_error = (
            _execute(source, task.function_name, 5 + task.constant)
            if parser_valid
            else (False, "parser_invalid")
        )
        baseline = baseline_rows[task.task_id]
        outputs.append(
            {
                "task_id": task.task_id,
                "function_name_seen_by_model": False,
                "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                "generated_source": source,
                "parser_valid": parser_valid,
                "parse_error": parse_error,
                "execution_success": executed,
                "execution_error": execution_error,
                "baseline_parser_valid": bool(baseline["parser_valid"]),
                "baseline_execution_success": bool(baseline["execution_success"]),
                "latency_ms": latency,
            }
        )
        if on_case is not None:
            on_case(outputs[-1])
    return outputs


def _summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(case_rows)
    successful = sum(row["execution_success"] for row in case_rows)
    baseline_successful = sum(row["baseline_execution_success"] for row in case_rows)
    wins = sum(
        row["execution_success"] and not row["baseline_execution_success"]
        for row in case_rows
    )
    losses = sum(
        not row["execution_success"] and row["baseline_execution_success"]
        for row in case_rows
    )
    return {
        "cases": count,
        "parser_validity": sum(row["parser_valid"] for row in case_rows) / max(1, count),
        "executable_success_rate": successful / max(1, count),
        "baseline_executable_success_rate": baseline_successful / max(1, count),
        "paired_execution_wins": wins,
        "paired_execution_losses": losses,
        "paired_execution_ties": count - wins - losses,
        "latency_ms_median": median(row["latency_ms"] for row in case_rows)
        if count else None,
    }


def run(config_path: Path, preflight_path: Path, device_name: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_overlay = ROOT / config["training"]["runtime"]["pythonpath_overlay"]
    if str(runtime_overlay) not in sys.path:
        sys.path.insert(0, str(runtime_overlay))
    import tokenizers
    import transformers
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready":
        raise SystemExit("pretrained Python CPU preflight is not ready")
    if preflight["configuration"]["sha256"] != _sha(config_path):
        raise SystemExit("preflight and pilot configuration identities differ")
    expected_transformers = str(config["training"]["runtime"]["transformers"])
    expected_tokenizers = str(config["training"]["runtime"]["tokenizers"])
    if transformers.__version__ != expected_transformers:
        raise SystemExit(
            f"expected Transformers {expected_transformers}; found {transformers.__version__}"
        )
    if tokenizers.__version__ != expected_tokenizers:
        raise SystemExit(
            f"expected tokenizers {expected_tokenizers}; found {tokenizers.__version__}"
        )
    expected_torch = str(config["training"]["runtime"]["torch"])
    if torch.__version__ != expected_torch:
        raise SystemExit(f"expected torch {expected_torch}; found {torch.__version__}")
    expected_overlay = ROOT / config["training"]["runtime"]["pythonpath_overlay"]
    if not str(transformers.__file__).startswith(str(expected_overlay)):
        raise SystemExit("Transformers was not loaded from the isolated pinned runtime overlay")
    if preflight.get("implementation_hashes", {}).get("pilot_runner") != _sha(Path(__file__).resolve()):
        raise SystemExit("pilot runner differs from the source-hashed CPU preflight")
    model_spec = config["pretrained_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        use_fast=True,
    )
    if type(tokenizer).__name__ != "RobertaTokenizerFast":
        raise SystemExit("pinned checkpoint did not load the audited tokenizer class")
    train = build_synthetic_tasks(
        start=int(config["dataset"]["train"]["start"]),
        count=int(config["dataset"]["train"]["count"]),
    )
    validation = build_synthetic_tasks(
        start=int(config["dataset"]["validation"]["start"]),
        count=int(config["dataset"]["validation"]["count"]),
    )
    baseline_path = ROOT / config["baseline"]["report"]
    if _sha(baseline_path) != config["baseline"]["report_sha256"]:
        raise SystemExit("frozen simple-baseline report changed")
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = {row["task_id"]: row for row in baseline_report["case_rows"]}
    if [task.task_id for task in validation] != list(baseline_rows):
        raise SystemExit("baseline cases differ from frozen validation task IDs")
    if transformers.__version__ != expected_transformers:
        raise SystemExit("Transformers version changed after tokenizer audit")

    device = torch.device(device_name)
    queue_path = ROOT / "outputs/.locks/nlp_gpu_training.lock"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock = queue_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise SystemExit("another registered NLP experiment holds the GPU queue lock") from error
    selected_uuid = _device_uuid(device)
    torch.cuda.set_device(device)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    model = T5ForConditionalGeneration.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(model_spec["parameter_count"]["exact"]):
        raise SystemExit("loaded checkpoint parameter count differs from the frozen audit")
    sources, labels = _tokenize_batch(
        tokenizer,
        [task.prompt for task in train],
        [task.source for task in train],
        config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    output_path = ROOT / config.get(
        "pilot_report", "outputs/reports/nlp_pretrained_python_pilot_v1.json"
    )
    checkpoint_path = ROOT / config.get(
        "checkpoint_output", ".data/nlp/pretrained_python_v1/codet5_small_seed17.pt"
    )
    partial_checkpoint = checkpoint_path.with_suffix(".partial.pt")
    final_model_dir = checkpoint_path.parent / "final_model"
    final_tokenizer_dir = checkpoint_path.parent / "final_tokenizer"
    if (
        checkpoint_path.exists()
        or partial_checkpoint.exists()
        or output_path.exists()
        or final_model_dir.exists()
        or final_tokenizer_dir.exists()
    ):
        raise SystemExit("pilot output already exists; retries and checkpoint reuse are forbidden")
    idle_wait_seconds = _wait_for_idle_gpu(
        selected_uuid, device, int(config.get("minimum_free_memory_mib", 24000))
    )
    gpu_metadata = _gpu_metadata(selected_uuid)
    budget_deadline = time.monotonic() + float(config["training"]["max_gpu_hours"]) * 3600
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "version": "nlp-pretrained-python-pilot-v1",
        "experiment_id": config["experiment_id"],
        "status": "training_in_progress",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "preflight_report": str(preflight_path.relative_to(ROOT)),
        "preflight_report_sha256": _sha(preflight_path),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "model": model_spec,
        "parameter_count_loaded": parameter_count,
        "runtime": {"transformers": transformers.__version__, "torch": torch.__version__},
        "device": str(device),
        "device_uuid": selected_uuid,
        "gpu": gpu_metadata,
        "exclusive_idle_wait_seconds": round(idle_wait_seconds, 2),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
        "ownership_watchdog": "poll every 0.5s; SIGINT only this process on foreign compute PID",
        "seed": seed,
        "train_tasks": len(train),
        "validation_tasks": len(validation),
        "optimizer_steps_planned": total_steps,
        "optimizer_steps_completed": 0,
        "completed_epochs": 0,
        "train_loss_by_epoch": [],
        "gpu_used": True,
        "test_content_read": False,
        "validation_used_for_selection": False,
        "checkpoint_reuse": False,
        "retry_or_best_of_n": False,
        "case_rows": [],
        "evidence_boundary": config["evidence_boundary"],
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training_started = time.monotonic()
    guard_stop = threading.Event()
    guard_reason: list[str] = []
    guard = threading.Thread(
        target=_watch_gpu_ownership,
        args=(selected_uuid, guard_stop, guard_reason),
        daemon=True,
    )
    guard.start()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model.to(device)
        torch.cuda.manual_seed_all(seed)
        sources = {name: value.to(device) for name, value in sources.items()}
        labels = labels.to(device)
        optimizer_steps = 0
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(train), generator=generator)
            loss_sum = 0.0
            examples = 0
            for start in range(0, len(train), batch_size):
                if time.monotonic() >= budget_deadline:
                    raise TimeoutError("two-hour pilot budget reached during training")
                indices = order[start : start + batch_size].to(device)
                batch_input = {name: value.index_select(0, indices) for name, value in sources.items()}
                batch_labels = labels.index_select(0, indices)
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch_input, labels=batch_labels)
                output.loss.backward()
                optimizer.step()
                loss_sum += float(output.loss.detach()) * len(indices)
                examples += len(indices)
                optimizer_steps += 1
            report["optimizer_steps_completed"] = optimizer_steps
            report["completed_epochs"] = epoch + 1
            report["train_loss_by_epoch"].append(loss_sum / max(1, examples))
            report["status"] = "training_in_progress"
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "optimizer_steps": optimizer_steps,
                    "configuration_sha256": _sha(config_path),
                    "pretrained_revision": model_spec["revision"],
                    "quality_status": "unqualified; validation pending",
                },
                partial_checkpoint,
            )
            _atomic_json(output_path, report)
        model.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_tokenizer_dir)
        report["status"] = "validation_in_progress"
        replayed_cases: list[dict[str, Any]] = []

        def record_case(case: dict[str, Any]) -> None:
            replayed_cases.append(case)
            report["case_rows"] = list(replayed_cases)
            report["partial_metrics"] = _summary(replayed_cases)
            _atomic_json(output_path, report)

        case_rows = _evaluate(
            model,
            tokenizer,
            validation,
            baseline_rows,
            config,
            device,
            budget_deadline,
            on_case=record_case,
        )
        report["case_rows"] = case_rows
        report["metrics"] = _summary(case_rows)
        report["status"] = "completed_unqualified"
        report["selected_checkpoint"] = "fixed final optimizer step"
        report["checkpoint_dir"] = str((checkpoint_path.parent / "final_model").relative_to(ROOT))
        report["checkpoint_files"] = {
            path.name: _sha(path)
            for path in sorted((checkpoint_path.parent / "final_model").glob("*"))
            if path.is_file()
        }
        report["tokenizer_dir"] = str((checkpoint_path.parent / "final_tokenizer").relative_to(ROOT))
        partial_checkpoint.unlink(missing_ok=True)
    except (TimeoutError, KeyboardInterrupt) as error:
        report["status"] = "budget_stopped_incomplete" if isinstance(error, TimeoutError) else "interrupted"
        report["stop_reason"] = guard_reason[0] if guard_reason else (str(error) or "SIGINT")
        if not partial_checkpoint.exists():
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "optimizer_steps": report["optimizer_steps_completed"],
                    "configuration_sha256": _sha(config_path),
                    "pretrained_revision": model_spec["revision"],
                    "quality_status": "unqualified; incomplete run",
                },
                partial_checkpoint,
            )
    except Exception as error:
        report["status"] = "failed_incomplete"
        report["stop_reason"] = f"{type(error).__name__}: {error}"
        if not partial_checkpoint.exists():
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "optimizer_steps": report["optimizer_steps_completed"],
                    "configuration_sha256": _sha(config_path),
                    "pretrained_revision": model_spec["revision"],
                    "quality_status": "unqualified; failed incomplete run",
                },
                partial_checkpoint,
            )
        raise
    finally:
        guard_stop.set()
        guard.join(timeout=2)
        report["wall_time_seconds"] = time.monotonic() - training_started
        report["peak_memory_allocated_mib"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 2
        )
        report["peak_memory_reserved_mib"] = round(
            torch.cuda.max_memory_reserved(device) / (1024**2), 2
        )
        _atomic_json(output_path, report)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nlp_pretrained_python_v1.json")
    parser.add_argument("--preflight", type=Path, default=ROOT / "outputs/reports/nlp_pretrained_python_preflight_v1.json")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    report = run(args.config.resolve(), args.preflight.resolve(), args.device)
    print(json.dumps({
        "status": report["status"],
        "experiment_id": report["experiment_id"],
        "optimizer_steps_completed": report["optimizer_steps_completed"],
        "metrics": report.get("metrics"),
        "gpu_used": report["gpu_used"],
    }, indent=2))
    return 0 if report["status"] == "completed_unqualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
