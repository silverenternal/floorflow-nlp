#!/usr/bin/env python3
"""Compact NLP training, chat specialization, and reproducibility smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorflow.nlp_flow.engineering import (
    BudgetReport,
    ExperimentSpec,
    ScheduleFreeAdamW,
    audit_parameter_budget,
    build_runtime,
    count_unique_trainable_parameters,
    load_nlp_checkpoint,
    parameter_breakdown,
    restore_nlp_checkpoint,
    save_nlp_checkpoint,
)
from floorflow.nlp_flow.experiment_config import candidate_config
from floorflow.nlp_flow.instruction_data import RowAlignedInstructionDataset
from floorflow.nlp_flow.model import CompactNLPDecoder
from floorflow.nlp_flow.training_data import PackedTokenDataset
from floorflow.nlp_flow.training_losses import batch_loss

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _resolve_paths(value: str | list[str]) -> tuple[Path, ...]:
    values = [value] if isinstance(value, str) else value
    return tuple(_resolve(item) for item in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/nlp_training_p2m_cpu_smoke_v1.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-sequences", type=int, default=16)
    parser.add_argument("--validation-sequences", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-allow-spec-change", action="store_true")
    parser.add_argument("--resume-model-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/train/nlp_p2m_candidate_cpu_smoke_v1",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    args.output = _resolve(str(args.output))
    if args.resume is not None:
        args.resume = _resolve(str(args.resume))
    if args.workers < 0 or args.checkpoint_every <= 0:
        raise ValueError("workers must be non-negative and checkpoint-every positive")

    spec = ExperimentSpec.from_json(args.spec)
    runtime_spec = spec.payload["runtime"]
    requested_device = str(runtime_spec["device"])
    if requested_device.startswith("cuda") and runtime_spec.get("gpu_authorized") is not True:
        raise RuntimeError("CUDA training is not authorized by the experiment spec")
    runtime = build_runtime(
        device=requested_device,
        precision=str(runtime_spec["precision"]),
        compile_model=bool(runtime_spec.get("compile_model", False)),
        seed=spec.seed,
        deterministic=bool(runtime_spec["deterministic"]),
        triton_kernels=bool(runtime_spec.get("triton_kernels", False)),
    )
    model = CompactNLPDecoder(candidate_config(spec))
    model = runtime.prepare_model(model)
    parameters = count_unique_trainable_parameters(model)
    planned_corpus_tokens = int(
        spec.payload["model"]["planned_corpus_tokens"]
    )
    budget_mode = str(
        spec.payload["model"].get("parameter_budget_mode", "matched_20_to_1")
    )
    if budget_mode == "matched_20_to_1":
        budget = audit_parameter_budget(parameters, planned_corpus_tokens)
    elif budget_mode == "open_capability_exploratory":
        target_parameters = planned_corpus_tokens // 20
        budget = BudgetReport(
            parameters=parameters,
            corpus_tokens=planned_corpus_tokens,
            tokens_per_parameter=(planned_corpus_tokens / parameters),
            target_parameters=target_parameters,
            relative_error=(
                abs(parameters - target_parameters)
                / max(1, target_parameters)
            ),
            tolerance=0.02,
        )
    else:
        raise ValueError(f"unsupported parameter budget mode: {budget_mode}")
    tokenizer_path = _resolve(str(spec.payload["model"]["tokenizer"]))
    train_paths = _resolve_paths(spec.payload["data"]["train"])
    validation_paths = _resolve_paths(spec.payload["data"]["validation"])
    data_manifest_path = _resolve(str(spec.payload["data"]["manifest"]))
    text_mode = str(spec.payload["data"].get("text_mode", "raw"))
    dataset_kind = str(
        spec.payload["data"].get("dataset", "packed_token")
    )
    family_values = spec.payload["data"].get("families")
    families = (
        frozenset(str(item) for item in family_values)
        if isinstance(family_values, list)
        else None
    )
    if dataset_kind == "row_aligned_instruction":
        train_data = RowAlignedInstructionDataset(
            train_paths,
            tokenizer_path,
            context_length=int(
                spec.payload["model"]["max_context_tokens"]
            ),
            max_sequences=args.train_sequences,
            seed=spec.seed,
        )
        validation_data = RowAlignedInstructionDataset(
            validation_paths,
            tokenizer_path,
            context_length=int(
                spec.payload["model"]["max_context_tokens"]
            ),
            max_sequences=args.validation_sequences,
            seed=spec.seed + 1,
        )
    elif dataset_kind == "packed_token":
        train_data = PackedTokenDataset(
            train_paths,
            tokenizer_path,
            context_length=int(spec.payload["model"]["max_context_tokens"]),
            max_sequences=args.train_sequences,
            text_mode=text_mode,
            families=families,
        )
        validation_data = PackedTokenDataset(
            validation_paths,
            tokenizer_path,
            context_length=int(spec.payload["model"]["max_context_tokens"]),
            max_sequences=args.validation_sequences,
            text_mode=text_mode,
            families=families,
        )
    else:
        raise ValueError(f"unsupported dataset kind: {dataset_kind}")
    loader_options = runtime.dataloader_options(args.workers)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        **loader_options,
    )
    optimizer_spec = spec.payload["optimizer"]
    optimizer_name = str(optimizer_spec.get("name", "adamw"))
    if optimizer_name == "schedule_free_adamw":
        optimizer = ScheduleFreeAdamW(
            model.parameters(),
            lr=float(optimizer_spec["learning_rate"]),
            weight_decay=float(optimizer_spec["weight_decay"]),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_spec["learning_rate"]),
            weight_decay=float(optimizer_spec["weight_decay"]),
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    warmup_steps = max(1, min(500, args.steps // 20))
    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = (
        None
        if optimizer_name == "schedule_free_adamw"
        else torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            learning_rate_scale,
        )
    )
    start_step = 0
    samples_seen = 0
    if args.resume is not None:
        resume_payload = load_nlp_checkpoint(
            args.resume,
            expected_spec_sha256=(
                None if args.resume_allow_spec_change else spec.sha256
            ),
            map_location=runtime.device,
        )
        if args.resume_model_only:
            load_result = model.load_state_dict(
                resume_payload["model"],
                strict=False,
            )
            unexpected = list(load_result.unexpected_keys)
            missing = [
                key for key in load_result.missing_keys
                if not key.startswith("emergence.")
            ]
            if unexpected or missing:
                raise RuntimeError(
                    f"model-only resume mismatch: unexpected={unexpected}, "
                    f"missing={missing}"
                )
        else:
            restored = restore_nlp_checkpoint(
                resume_payload,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=runtime.device,
            )
            start_step = restored["step"]
            samples_seen = restored["samples_seen"]
    losses = []
    emergence_losses = []
    emergence_history = []
    dynamic_history = []
    tokens_seen = 0
    validation_history = []
    best_validation = float("inf")
    best_checkpoint = args.output / "best.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    model.train()
    if isinstance(optimizer, ScheduleFreeAdamW):
        optimizer.train()
    emergence_target = float(
        spec.payload["model"].get("emergence_loss_weight", 0.0)
    )
    dynamic_depth_weight = float(
        spec.payload["model"].get("dynamic_depth_weight", 0.0)
    )
    dynamic_target_depth = float(
        spec.payload["model"].get("dynamic_target_depth", 2.0)
    )
    conversation_control_weight = float(
        spec.payload["model"].get("conversation_control_loss_weight", 0.0)
    )
    benchmark_proxy_weight = float(
        spec.payload["model"].get("benchmark_proxy_loss_weight", 0.0)
    )
    train_iterator = iter(train_loader)
    log_started = time.perf_counter()
    log_tokens = 0
    for step in range(start_step, args.steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        optimizer.zero_grad(set_to_none=True)
        emergence_weight = (
            emergence_target
            * min(1.0, (step + 1) / max(1, args.steps // 10))
        )
        loss, emergence_loss, dynamic_metrics, emergence_metrics = batch_loss(
            model,
            batch,
            device=runtime.device,
            autocast=runtime.autocast,
            emergence_weight=emergence_weight,
            dynamic_depth_weight=dynamic_depth_weight,
            dynamic_target_depth=dynamic_target_depth,
            conversation_control_weight=conversation_control_weight,
            benchmark_proxy_weight=benchmark_proxy_weight,
        )
        if emergence_loss is not None:
            emergence_losses.append(float(emergence_loss.detach()))
        if emergence_metrics is not None and (step + 1) % 50 == 0:
            emergence_history.append(
                {"step": step + 1, **emergence_metrics}
            )
        if dynamic_metrics is not None and (step + 1) % 50 == 0:
            dynamic_history.append({"step": step + 1, **dynamic_metrics})
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))
        samples_seen += int(batch["input_ids"].shape[0])
        tokens_seen += int(batch["input_ids"].numel())
        log_tokens += int(batch["input_ids"].numel())
        if (step + 1) % 50 == 0:
            elapsed = time.perf_counter() - log_started
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "lr": (
                            scheduler.get_last_lr()[0]
                            if scheduler is not None
                            else optimizer.param_groups[0]["scheduled_lr"]
                        ),
                        "tokens": tokens_seen,
                        "tokens_per_second": log_tokens / max(elapsed, 1.0e-6),
                        "dynamic": dynamic_metrics,
                        "emergence": emergence_metrics,
                    }
                ),
                flush=True,
            )
            log_started = time.perf_counter()
            log_tokens = 0
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            model.eval()
            if isinstance(optimizer, ScheduleFreeAdamW):
                optimizer.eval()
            measured = []
            with torch.no_grad():
                for validation_index, validation_batch in enumerate(validation_loader):
                    if validation_index >= args.validation_batches:
                        break
                    measured.append(
                        float(
                            batch_loss(
                                model,
                                validation_batch,
                                device=runtime.device,
                                autocast=runtime.autocast,
                                emergence_weight=emergence_target,
                                dynamic_depth_weight=dynamic_depth_weight,
                                dynamic_target_depth=dynamic_target_depth,
                                conversation_control_weight=(
                                    conversation_control_weight
                                ),
                                benchmark_proxy_weight=benchmark_proxy_weight,
                            )[0]
                        )
                    )
            validation_loss = sum(measured) / max(1, len(measured))
            validation_history.append(
                {"step": step + 1, "loss": validation_loss}
            )
            periodic = args.output / f"step_{step + 1:07d}.pt"
            save_nlp_checkpoint(
                periodic,
                step=step + 1,
                samples_seen=samples_seen,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                spec_sha256=spec.sha256,
                data_manifest_sha256=file_sha256(data_manifest_path),
                tokenizer_sha256=file_sha256(tokenizer_path),
                config=spec.payload,
                device=runtime.device,
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                shutil.copy2(periodic, best_checkpoint)
            model.train()
            if isinstance(optimizer, ScheduleFreeAdamW):
                optimizer.train()
    validation_losses = []
    validation_batches = []
    validation_dynamic = []
    model.eval()
    if isinstance(optimizer, ScheduleFreeAdamW):
        optimizer.eval()
    with torch.no_grad():
        for batch in validation_loader:
            validation_batches.append(batch)
            validation_loss, _, dynamic_metrics, _ = batch_loss(
                model,
                batch,
                device=runtime.device,
                autocast=runtime.autocast,
                emergence_weight=emergence_target,
                dynamic_depth_weight=dynamic_depth_weight,
                dynamic_target_depth=dynamic_target_depth,
                conversation_control_weight=conversation_control_weight,
                benchmark_proxy_weight=benchmark_proxy_weight,
            )
            validation_losses.append(float(validation_loss))
            if dynamic_metrics is not None:
                validation_dynamic.append(dynamic_metrics)
    checkpoint = args.output / "checkpoint.pt"
    save_nlp_checkpoint(
        checkpoint,
        step=args.steps,
        samples_seen=samples_seen,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        spec_sha256=spec.sha256,
        data_manifest_sha256=file_sha256(data_manifest_path),
        tokenizer_sha256=file_sha256(tokenizer_path),
        config=spec.payload,
        device=runtime.device,
    )
    payload = load_nlp_checkpoint(
        checkpoint,
        expected_spec_sha256=spec.sha256,
        map_location=runtime.device,
    )
    restored_model = CompactNLPDecoder(candidate_config(spec))
    restore_nlp_checkpoint(
        payload,
        model=restored_model,
        device=runtime.device,
    )
    restored_model = restored_model.to(runtime.device).eval()
    with torch.no_grad():
        restored_validation_loss = (
            float(
                batch_loss(
                    restored_model,
                    validation_batches[0],
                    device=runtime.device,
                    autocast=runtime.autocast,
                    emergence_weight=emergence_target,
                    dynamic_depth_weight=dynamic_depth_weight,
                    dynamic_target_depth=dynamic_target_depth,
                    conversation_control_weight=conversation_control_weight,
                    benchmark_proxy_weight=benchmark_proxy_weight,
                )[0]
            )
            if validation_batches
            else None
        )
    restore_exact = (
        restored_validation_loss is not None
        and validation_losses
        and math.isclose(
            restored_validation_loss,
            validation_losses[0],
            rel_tol=0.0,
            abs_tol=1.0e-7,
        )
    )
    report = {
        "version": "nlp-p2m-cpu-smoke-v1",
        "status": "pass" if all(math.isfinite(x) for x in losses) else "fail",
        "evidence_boundary": "Bounded CPU training-stack smoke only; no capability or quality claim.",
        "spec": spec.manifest(),
        "runtime": runtime.manifest(),
        "parameters": parameters,
        "parameter_breakdown": parameter_breakdown(model),
        "budget": {
            "mode": budget_mode,
            "corpus_tokens": budget.corpus_tokens,
            "tokens_per_parameter": budget.tokens_per_parameter,
            "target_parameters": budget.target_parameters,
            "relative_error": budget.relative_error,
            "tolerance": budget.tolerance,
            "passed": budget.passed,
            "claim_eligible": budget_mode == "matched_20_to_1",
        },
        "steps": args.steps,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "train_losses": losses[-200:],
        "train_loss_count": len(losses),
        "validation_losses": validation_losses,
        "validation_history": validation_history,
        "emergence_losses": emergence_losses[-200:],
        "emergence_loss_count": len(emergence_losses),
        "emergence_history": emergence_history,
        "dynamic_stack": {
            "enabled": bool(spec.payload["model"].get("enable_dynamic_stack", False)),
            "train_history": dynamic_history,
            "validation_batches": validation_dynamic,
            "physical_parameter_count_is_depth_independent": (
                bool(spec.payload["model"].get("enable_dynamic_stack", False))
            ),
            "evidence_boundary": (
                "Runtime routing and compute diagnostics only; no capability "
                "equivalence or matched-compute advantage is implied."
            ),
        },
        "best_validation_loss": (
            best_validation if math.isfinite(best_validation) else None
        ),
        "best_checkpoint": (
            str(best_checkpoint.relative_to(ROOT))
            if best_checkpoint.exists()
            else None
        ),
        "gradient_reached_prefixes": sorted(
            {
                name.split(".", 1)[0]
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        ),
        "checkpoint_restore_exact": restore_exact,
        "restored_validation_loss": restored_validation_loss,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data": {
            "train": [str(path.relative_to(ROOT)) for path in train_paths],
            "validation": [str(path.relative_to(ROOT)) for path in validation_paths],
            "test_read_during_smoke": False,
            "target_used_at_inference": False,
            "text_mode": text_mode,
            "dataset_kind": dataset_kind,
            "families": sorted(families) if families is not None else None,
        },
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not restore_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
