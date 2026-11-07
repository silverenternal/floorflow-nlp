"""Matched train-only structural ablation execution and audit reporting."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ast_metrics, parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar, load_ast_grammar
from newera.nlp_flow.components.research.experiments.config.experiment_deadline import BudgetDeadlineExceeded, ExperimentDeadline
from newera.nlp_flow.components.research.experiments.config.experiment_fingerprint import implementation_artifact_paths
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_training import RolloutCurriculum
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import audit_prefix_batch, encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_runtime import generate_prefix_tree
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_training import train_prefix_tree_policy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_action_data import (
    build_real_corpus_action_dataset,
    iter_declaration_fragments,
)
from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary
from newera.nlp_flow.components.generation.trees.runtime.tree_completion_runtime import (
    TreeCompletionBudget,
    load_tree_completion_budget,
)

ARM_IDS = ("flat", "local_mask", "lexical_mask", "hierarchical")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_split_rows(path: Path, split: str) -> list[dict[str, Any]]:
    import re

    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) != split:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError("split metadata changed while reading selected rows")
            rows.append(row)
    return rows


def _load_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    shared = config["shared"]
    train_path = root / shared["train_actions"]
    train_rows = _read_split_rows(train_path, "train")
    if not train_rows or any(row["split"] != "train" for row in train_rows):
        raise ValueError("matched training actions must contain train rows only")

    ast_path = root / shared["ast_corpus"]
    validation_sources = _read_split_rows(ast_path, shared["validation_split"])
    prompt_contract = json.loads((root / shared["prompt_contract"]).read_text(encoding="utf-8"))
    validation_fragments = iter_declaration_fragments(
        validation_sources,
        prompt_contract["fragments"],
    )
    validation_rows, validation_audit = build_real_corpus_action_dataset(
        validation_fragments,
        {
            language: load_ast_grammar(root / shared["transition_grammar"], language)
            for language in shared["languages"]
        },
        prompt_contract["prompt"],
    )
    validation_audit["fragment_granularity"] = "top_level_declaration"
    if validation_audit["status"] != "pass" or not validation_rows:
        raise ValueError("frozen validation action construction failed")
    grammars = {
        language: load_ast_grammar(root / shared["transition_grammar"], language)
        for language in shared["languages"]
    }
    return {
        "train_path": train_path,
        "train_rows": train_rows,
        "validation_sources": validation_sources,
        "validation_rows": [row.as_dict() for row in validation_rows],
        "grammars": grammars,
        "vocabulary": build_vocabulary(train_rows, grammars),
        "validation_audit": validation_audit,
    }


def _completion_budgets(
    config: dict[str, Any],
    grammars: dict[str, ASTTransitionGrammar],
    root: Path,
) -> dict[str, TreeCompletionBudget]:
    grammar_payload = json.loads((root / config["shared"]["transition_grammar"]).read_text())
    budgets = {}
    for language, grammar in grammars.items():
        schema = grammar_payload["provenance"][language]["parser_schema"]
        schema_path = Path(schema["path"])
        if not schema_path.is_absolute():
            schema_path = root / schema_path
        budgets[language] = load_tree_completion_budget(
            grammar,
            schema_path,
            expected_schema_sha256=str(schema["sha256"]),
        )
    return budgets


def build_preflight(config_path: Path, *, root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") not in {
        "nlp-structure-matched-v12",
        "nlp-structure-matched-v13",
    }:
        raise ValueError("unsupported matched structure experiment version")
    arms = config.get("arms")
    if not isinstance(arms, list) or tuple(arm.get("id") for arm in arms) != ARM_IDS:
        raise ValueError(f"expected arms in registered order: {ARM_IDS}")
    seeds = config["shared"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("registered seeds must be non-empty and unique")
    queue_path = root / config["gpu_queue"]
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    registrations = [
        item
        for item in queue["experiments"]
        if item.get("experiment_id") == config["experiment_id"]
    ]
    if len(registrations) != 1 or queue.get("training_allowed_now") is not True:
        raise ValueError("matched experiment must have one active GPU queue registration")
    registration = registrations[0]
    if registration.get("seed_set") != seeds or registration.get("arms") != [arm["id"] for arm in arms]:
        raise ValueError("GPU queue seeds and arms must match the experiment config")
    loaded = _load_inputs(config, root)
    budgets = _completion_budgets(config, loaded["grammars"], root)
    train_batch = encode_prefix_rows(loaded["train_rows"], loaded["grammars"], loaded["vocabulary"])
    validation_batch = encode_prefix_rows(
        loaded["validation_rows"], loaded["grammars"], loaded["vocabulary"]
    )
    train_batch_audit = audit_prefix_batch(train_batch)
    validation_batch_audit = audit_prefix_batch(validation_batch)
    if train_batch_audit["status"] != "pass" or validation_batch_audit["status"] != "pass":
        raise ValueError("matched train or validation action batch failed its shape audit")
    grammar_path = root / config["shared"]["transition_grammar"]
    action_steps = sum(len(row["actions"]) for row in loaded["train_rows"])
    steps_per_run = (
        (len(loaded["train_rows"]) + int(config["shared"]["batch_rows"]) - 1)
        // int(config["shared"]["batch_rows"])
    ) * int(config["shared"]["epochs"])
    budget = registration["budget"]
    total_steps = steps_per_run * len(seeds) * len(arms)
    total_action_tokens = action_steps * int(config["shared"]["epochs"]) * len(seeds) * len(arms)
    total_examples = len(loaded["train_rows"]) * int(config["shared"]["epochs"]) * len(seeds) * len(arms)
    if total_steps > int(budget["max_optimizer_steps"]):
        raise ValueError("registered optimizer-step budget is too small")
    if total_action_tokens > int(budget["max_action_tokens"]):
        raise ValueError("registered action-token budget is too small")
    if total_examples > int(budget["max_training_examples"]):
        raise ValueError("registered training-example budget is too small")
    return {
        "version": f"nlp-structure-matched-preflight-{config['version'].rsplit('v', 1)[-1]}",
        "status": "ready",
        "configuration": {"path": str(config_path), "sha256": _sha(config_path)},
        "implementation_artifacts": {
            relative: _sha(root / relative)
            for relative in implementation_artifact_paths(config)
        },
        "gpu_queue": {
            "path": str(queue_path.relative_to(root)),
            "sha256": _sha(queue_path),
            "experiment_id": config["experiment_id"],
            "budget": budget,
        },
        "inputs": {
            "train_actions": {
                "path": str(loaded["train_path"].relative_to(root)),
                "sha256": _sha(loaded["train_path"]),
                "rows": len(loaded["train_rows"]),
                "split_scope": "train only",
            },
            "ast_corpus": {
                "path": config["shared"]["ast_corpus"],
                "validation_rows_parsed": len(loaded["validation_sources"]),
            "validation_source_identity_sha256": hashlib.sha256(
                "\n".join(
                    str(row["case_id"]) + ":" + str(row["source_sha256"])
                    for row in loaded["validation_sources"]
                ).encode()
            ).hexdigest(),
                "test_rows_parsed": 0,
                "validation_action_rows": len(loaded["validation_rows"]),
                "validation_audit": loaded["validation_audit"],
            "batch_audits": {
                "train": train_batch_audit,
                "validation": validation_batch_audit,
            },
            },
            "transition_grammar_sha256": _sha(grammar_path),
        },
        "arms": [
            {
                **arm,
                "registered_seeds": seeds,
                "optimizer_steps_per_seed": steps_per_run,
                "optimizer_steps_total": steps_per_run * len(seeds),
                "train_action_steps_per_seed": action_steps * int(config["shared"]["epochs"]),
            }
            for arm in arms
        ],
        "budget_estimate": {
            "training_examples": total_examples,
            "optimizer_steps": total_steps,
            "action_tokens": total_action_tokens,
        },
        "completion_contract_languages": sorted(budgets),
        "gpu_used": False,
        "test_content_read": False,
        "winner_selection": False,
        "evidence_boundary": config["evidence_boundary"],
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _free_running_metrics(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    source_by_case: dict[str, str],
    ast_by_case: dict[str, str],
    grammars: dict[str, ASTTransitionGrammar],
    vocabulary: dict[str, Any],
    *,
    device: torch.device,
    arm: dict[str, Any],
    completion_budgets: dict[str, TreeCompletionBudget],
    max_steps: int,
    deadline: ExperimentDeadline | None = None,
) -> dict[str, Any]:
    outcomes = []
    errors: dict[str, int] = {}
    node_f1: list[float] = []
    edge_f1: list[float] = []
    lexical_f1: list[float] = []
    scope_f1: list[float] = []
    tree_distance: list[float] = []
    latencies: list[float] = []
    exact = 0
    budget_failures = 0
    dead_ends = 0
    for row in rows:
        if deadline is not None:
            deadline.check("free_running_validation_case")
        trace: list[dict[str, Any]] = []
        started = time.perf_counter()
        source, _actions, error = generate_prefix_tree(
            model,
            str(row["prompt"]),
            str(row["language"]),
            grammars[str(row["language"])],
            vocabulary,
            device=device,
            max_steps=max_steps,
            trace=trace,
            local_transition_mask=bool(arm["local_transition_mask"]),
            lexical_candidate_mask=bool(arm["lexical_candidate_mask"]),
            completion_budget=(
                completion_budgets[str(row["language"])]
                if arm["terminalizable_completion_mask"]
                else None
            ),
            deadline=deadline,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        error_type = next(
            (str(event["error_type"]) for event in trace if event.get("error_type")),
            "generation_failed" if error else "none",
        )
        errors[error_type] = errors.get(error_type, 0) + int(error is not None)
        if error and ("budget" in error.lower() or "node cap" in error.lower()):
            budget_failures += 1
        if error and "no legal action" in error.lower():
            dead_ends += 1
        metrics = None
        if source is not None:
            try:
                predicted = parse_ast_graph(source, str(row["language"]))
                reference_source = source_by_case[str(row["case_id"])]
                reference = parse_ast_graph(reference_source, str(row["language"]))
                metrics = ast_metrics(predicted, reference)
                exact += int(
                    serialize_ast_graph(predicted) == ast_by_case[str(row["case_id"])]
                )
                node_f1.append(float(metrics["node_type_f1"]))
                edge_f1.append(float(metrics["typed_edge_f1"]))
                lexical_f1.append(float(metrics["lexical_f1"]))
                scope_f1.append(float(metrics["scope_relation_f1"]))
                tree_distance.append(float(metrics["normalized_tree_edit_distance"]))
            except (KeyError, TypeError, ValueError, UnicodeError):
                metrics = None
        outcomes.append(
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "full_sequence_success": source is not None,
                "parser_valid": bool(metrics is not None),
                "first_failure": next(
                    (event.get("error") for event in trace if event.get("error")),
                    error,
                ),
            }
        )
    count = len(rows)
    return {
        "cases": count,
        "parser_validity": sum(item["parser_valid"] for item in outcomes) / max(1, count),
        "full_sequence_success_rate": sum(item["full_sequence_success"] for item in outcomes) / max(1, count),
        "exact_tree_match": exact / max(1, count),
        "node_type_f1": statistics.fmean(node_f1) if node_f1 else 0.0,
        "typed_edge_f1": statistics.fmean(edge_f1) if edge_f1 else 0.0,
        "lexical_f1": statistics.fmean(lexical_f1) if lexical_f1 else 0.0,
        "scope_relation_f1": statistics.fmean(scope_f1) if scope_f1 else 0.0,
        "normalized_tree_edit_distance": statistics.fmean(tree_distance) if tree_distance else 1.0,
        "dead_end_rate": dead_ends / max(1, count),
        "budget_exhaustion_rate": budget_failures / max(1, count),
        "first_failure_counts": errors,
        "latency_ms_median": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "case_outcomes": outcomes,
        "selection_on_validation": False,
    }


def _execute_arm_seed(
    config: dict[str, Any],
    arm: dict[str, Any],
    seed: int,
    *,
    root: Path,
    config_path: Path,
    device: torch.device,
    loaded: dict[str, Any],
    train_batch: Any,
    validation_batch: Any,
    source_by_case: dict[str, str],
    ast_by_case: dict[str, str],
    budgets: dict[str, TreeCompletionBudget],
    deadline: ExperimentDeadline,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    curriculum = RolloutCurriculum(**config["rollout_curriculum"])
    train_rows = loaded["train_rows"]
    grammars = loaded["grammars"]
    vocabulary = loaded["vocabulary"]
    model, teacher_metrics = train_prefix_tree_policy(
        train_batch,
        validation_batch,
        vocabulary,
        device=device,
        seed=seed,
        epochs=int(config["shared"]["epochs"]),
        batch_rows=int(config["shared"]["batch_rows"]),
        hidden_size=int(config["shared"]["hidden_size"]),
        layers=int(config["shared"]["decoder_layers"]),
        heads=int(config["shared"]["attention_heads"]),
        learning_rate=float(config["shared"]["learning_rate"]),
        train_rows=train_rows if arm["train_only_rollout"] else None,
        grammars=grammars if arm["train_only_rollout"] else None,
        rollout_curriculum=curriculum if arm["train_only_rollout"] else None,
        local_transition_mask=bool(arm["local_transition_mask"]),
        lexical_candidate_mask=bool(arm["lexical_candidate_mask"]),
        completion_budgets=budgets if arm["terminalizable_completion_mask"] else None,
        blueprint_conditioning=bool(arm["blueprint_conditioning"]),
        deadline=deadline,
    )
    checkpoint = root / config["checkpoint_dir"] / arm["id"] / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint.with_name(f"seed_{seed}.partial.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "vocabulary": vocabulary,
            "arm": arm,
            "seed": seed,
            "config_sha256": _sha(config_path),
            "train_actions_sha256": _sha(loaded["train_path"]),
            "quality_status": "unqualified; free-running metrics pending",
        },
        partial_checkpoint,
    )
    try:
        generation_metrics = _free_running_metrics(
            model,
            loaded["validation_rows"],
            source_by_case,
            ast_by_case,
            grammars,
            vocabulary,
            device=device,
            arm=arm,
            completion_budgets=budgets,
            max_steps=int(config["shared"]["maximum_action_steps"]),
            deadline=deadline,
        )
    except BudgetDeadlineExceeded as error:
        error.partial_checkpoint = str(partial_checkpoint.relative_to(root))
        error.training_metrics = {
            "optimizer_steps": teacher_metrics.get("optimizer_steps"),
            "training_regime": teacher_metrics.get("training_regime"),
        }
        raise
    torch.save(
        {
            "model": model.state_dict(),
            "vocabulary": vocabulary,
            "arm": arm,
            "seed": seed,
            "config_sha256": _sha(config_path),
            "train_actions_sha256": _sha(loaded["train_path"]),
            "quality_status": "unqualified; matched validation only",
        },
        checkpoint,
    )
    partial_checkpoint.unlink(missing_ok=True)
    return {
        "status": "completed_unqualified",
        "seed": seed,
        "teacher_forced": teacher_metrics,
        "free_running": generation_metrics,
        "optimizer_steps": teacher_metrics["optimizer_steps"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "checkpoint": str(checkpoint.relative_to(root)),
    }


def run_experiment(config_path: Path, *, root: Path, device: torch.device) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = build_preflight(config_path, root=root)
    loaded = _load_inputs(config, root)
    train_rows = loaded["train_rows"]
    validation_rows = loaded["validation_rows"]
    grammars = loaded["grammars"]
    vocabulary = loaded["vocabulary"]
    train_batch = encode_prefix_rows(train_rows, grammars, vocabulary)
    validation_batch = encode_prefix_rows(validation_rows, grammars, vocabulary)
    budgets = _completion_budgets(config, grammars, root)
    source_by_case = {
        str(row["case_id"]): str(row["source"])
        for row in iter_declaration_fragments(
            loaded["validation_sources"],
            json.loads((root / config["shared"]["prompt_contract"]).read_text())["fragments"],
        )
    }
    reference_ast_by_case = {
        str(row["case_id"]): str(row["ast"])
        for row in iter_declaration_fragments(
            loaded["validation_sources"],
            json.loads((root / config["shared"]["prompt_contract"]).read_text())["fragments"],
        )
    }
    arms_report = []
    budget = preflight["gpu_queue"]["budget"]
    started = time.monotonic()
    total_budget_seconds = float(budget["max_gpu_hours_total"]) * 3600
    seed_budget_seconds = float(budget["max_gpu_hours_per_seed"]) * 3600
    elapsed_by_seed: dict[int, float] = {}
    stop_reason: str | None = None
    for arm in config["arms"]:
        runs = []
        for seed in config["shared"]["seeds"]:
            seed = int(seed)
            if stop_reason is not None:
                runs.append({"status": "not_attempted_after_stop", "seed": seed, "reason": stop_reason})
                continue
            if time.monotonic() - started >= float(budget["max_gpu_hours_total"]) * 3600:
                stop_reason = "total_gpu_hour_budget_reached"
                runs.append({"status": "not_attempted_after_stop", "seed": seed, "reason": stop_reason})
                continue
            if elapsed_by_seed.get(seed, 0.0) >= float(budget["max_gpu_hours_per_seed"]) * 3600:
                stop_reason = f"per_seed_gpu_hour_budget_reached:{seed}"
                runs.append({"status": "not_attempted_after_stop", "seed": seed, "reason": stop_reason})
                continue
            run_started = time.monotonic()
            deadline = ExperimentDeadline(
                started_at=started,
                total_deadline=started + total_budget_seconds,
                seed_deadline=run_started + seed_budget_seconds - elapsed_by_seed.get(seed, 0.0),
            )
            try:
                run_result = _execute_arm_seed(
                    config,
                    arm,
                    seed,
                    root=root,
                    config_path=config_path,
                    device=device,
                    loaded=loaded,
                    train_batch=train_batch,
                    validation_batch=validation_batch,
                    source_by_case=source_by_case,
                    ast_by_case=reference_ast_by_case,
                    budgets=budgets,
                    deadline=deadline,
                )
                elapsed = time.monotonic() - run_started
                elapsed_by_seed[seed] = elapsed_by_seed.get(seed, 0.0) + elapsed
                run_result["wall_time_seconds"] = elapsed
                runs.append(run_result)
                if elapsed_by_seed[seed] > float(budget["max_gpu_hours_per_seed"]) * 3600:
                    stop_reason = f"per_seed_gpu_hour_budget_exceeded:{seed}"
            except BudgetDeadlineExceeded as error:
                elapsed = time.monotonic() - run_started
                elapsed_by_seed[seed] = elapsed_by_seed.get(seed, 0.0) + elapsed
                runs.append(
                    {
                        "status": "budget_stopped",
                        "seed": seed,
                        "scope": error.scope,
                        "phase": error.phase,
                        "wall_time_seconds": elapsed,
                        "partial_checkpoint": error.partial_checkpoint,
                        "training_metrics": error.training_metrics,
                        "error": str(error),
                    }
                )
                stop_reason = f"{error.scope}_gpu_hour_budget_reached:{arm['id']}:{seed}"
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError) as error:
                elapsed = time.monotonic() - run_started
                elapsed_by_seed[seed] = elapsed_by_seed.get(seed, 0.0) + elapsed
                runs.append(
                    {
                        "status": "failed",
                        "seed": seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                stop_reason = f"seed_failure:{arm['id']}:{seed}"
        arms_report.append({"id": arm["id"], "runs": runs})
    return {
        "version": f"nlp-structure-matched-report-{config['version'].rsplit('v', 1)[-1]}",
        "status": (
            "budget_stopped_incomplete"
            if stop_reason is not None and "budget" in stop_reason
            else "completed_unqualified"
            if all(run["status"] == "completed_unqualified" for arm in arms_report for run in arm["runs"])
            else "completed_with_seed_failures"
        ),
        "configuration": {"path": str(config_path), "sha256": _sha(config_path)},
        "preflight": preflight,
        "device": str(device),
        "training_performed": True,
        "gpu_used": device.type == "cuda",
        "test_content_read": False,
        "winner_selected": False,
        "stop_reason": stop_reason,
        "wall_time_seconds": time.monotonic() - started,
        "elapsed_wall_hours_by_seed": {
            str(seed): seconds / 3600 for seed, seconds in elapsed_by_seed.items()
        },
        "gpu_device_hours_measured": None,
        "arms": arms_report,
        "evidence_boundary": config["evidence_boundary"],
    }
