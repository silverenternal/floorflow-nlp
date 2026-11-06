"""Matched CPU/GPU screen for low-data structured prediction heads."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.experiment_deadline import ExperimentDeadline
from floorflow.nlp_flow.prefix_policy_data import audit_prefix_batch, encode_prefix_rows
from floorflow.nlp_flow.prefix_tree_runtime import generate_prefix_tree
from floorflow.nlp_flow.prefix_tree_training import train_prefix_tree_policy
from floorflow.nlp_flow.real_corpus_action_data import action_record as record_action
from floorflow.nlp_flow.structure_ablation_runner import _completion_budgets, _load_inputs

ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_callback(checkpoint_path, arm, arm_id, seed, config_path, report, report_path):
    def save(model, epoch: int, loss: float) -> None:
        temporary = checkpoint_path.with_suffix(".partial.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "arm": arm,
                "seed": seed,
                "epoch": epoch,
                "loss": loss,
                "config_sha256": _sha(config_path),
                "quality_status": "unqualified",
            },
            temporary,
        )
        report["arms"][arm_id]["completed_epochs"] = epoch
        report["arms"][arm_id]["last_train_loss"] = loss
        _atomic_json(report_path, report)

    return save


def _case_metrics(model, rows, inputs, budgets, config, device, report, arm_id, deadline):
    model.eval()
    case_rows = []
    candidate_sizes = []
    for row in rows:
        trace: list[dict[str, object]] = []
        started = time.perf_counter()
        source, actions, generation_error = generate_prefix_tree(
            model,
            str(row["prompt"]),
            str(row["language"]),
            inputs["grammars"][str(row["language"])],
            inputs["vocabulary"],
            device=device,
            max_steps=int(config["shared"]["maximum_action_steps"]),
            trace=trace,
            completion_budget=budgets[str(row["language"])],
            use_completion_candidate_mask=False,
            deadline=deadline,
            local_transition_mask=bool(config["shared"]["local_transition_mask"]),
            lexical_candidate_mask=bool(config["shared"]["lexical_candidate_mask"]),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        parse_valid = False
        parse_error = generation_error
        if source is not None:
            try:
                parse_ast_graph(source, str(row["language"]))
                parse_valid = True
                parse_error = None
            except (TypeError, ValueError, UnicodeError) as error:
                parse_error = str(error)
        generated_records = [record_action(action) for action in actions]
        exact_action_match = generated_records == row["actions"]
        sizes = [
            len(choice.get("allowed_ids", []))
            for step in trace
            for choice in step.get("choices", [])
            if choice.get("head") == "node_kind"
        ]
        candidate_sizes.extend(sizes)
        case_rows.append(
            {
                "case_id": row["case_id"],
                "language": row["language"],
                "generated_source": source,
                "parser_valid": parse_valid,
                "parse_error": parse_error,
                "exact_action_sequence_match": exact_action_match,
                "gold_action_count": len(row["actions"]),
                "generated_action_count": len(actions),
                "node_kind_legal_candidate_sizes": sizes,
                "generation_latency_ms": latency_ms,
                "execution_success": None,
                "execution_not_evaluated_reason": "AST validation rows have no bounded expected-input/result contract; arbitrary repository code is not executed.",
            }
        )
        report["arms"][arm_id]["case_rows"] = list(case_rows)
        report["arms"][arm_id]["partial_metrics"] = {
            "cases_completed": len(case_rows),
            "parser_valid": sum(value["parser_valid"] for value in case_rows),
            "exact_action_matches": sum(value["exact_action_sequence_match"] for value in case_rows),
        }
        _atomic_json(ROOT / config["report"], report)
    count = len(case_rows)
    return {
        "cases": count,
        "parser_validity": sum(value["parser_valid"] for value in case_rows) / max(1, count),
        "exact_action_sequence_match_rate": sum(
            value["exact_action_sequence_match"] for value in case_rows
        ) / max(1, count),
        "node_kind_legal_candidate_count_mean": sum(candidate_sizes) / max(1, len(candidate_sizes)),
        "node_kind_decisions": len(candidate_sizes),
        "execution_evaluated_cases": 0,
        "latency_ms_median": float(torch.tensor(
            [value["generation_latency_ms"] for value in case_rows]
        ).median()) if case_rows else None,
        "case_rows": case_rows,
    }


def run_screen(config_path: Path, preflight_path: Path, device: torch.device) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready":
        raise SystemExit("data-efficient head screen CPU preflight is not ready")
    if preflight["configuration"]["sha256"] != _sha(config_path):
        raise SystemExit("screen and CPU preflight configuration identities differ")
    data_config_path = ROOT / config["data_config"]
    if _sha(data_config_path) != config["data_config_sha256"]:
        raise SystemExit("frozen source data configuration changed")
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(data_config, ROOT)
    budgets = _completion_budgets(data_config, inputs["grammars"], ROOT)
    minimums = {language: budget.minimum_root_nodes() for language, budget in budgets.items()}
    train = encode_prefix_rows(
        inputs["train_rows"], inputs["grammars"], inputs["vocabulary"], budget_minimums=minimums
    )
    validation = encode_prefix_rows(
        inputs["validation_rows"], inputs["grammars"], inputs["vocabulary"], budget_minimums=minimums
    )
    if audit_prefix_batch(train)["status"] != "pass" or audit_prefix_batch(validation)["status"] != "pass":
        raise SystemExit("frozen train or validation batch contract failed after preflight")
    shared = config["shared"]
    seed = int(shared["seed"])
    torch.manual_seed(seed)
    from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy

    initial_model = PrefixTreePolicy(
        inputs["vocabulary"],
        hidden_size=int(shared["hidden_size"]),
        layers=int(shared["decoder_layers"]),
        heads=int(shared["attention_heads"]),
        max_steps=int(shared["maximum_action_steps"]),
        blueprint_conditioning=bool(shared["blueprint_conditioning"]),
    ).cpu()
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    checkpoint_dir = ROOT / config["checkpoint_dir"]
    report_path = ROOT / config["report"]
    if checkpoint_dir.exists() or report_path.exists():
        raise SystemExit("screen outputs already exist; retries and checkpoint reuse are forbidden")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = ExperimentDeadline(
        started_at=started,
        total_deadline=started + float(config["budget"]["max_gpu_hours_total"]) * 3600,
        seed_deadline=started + float(config["budget"]["max_gpu_hours_total"]) * 3600,
    )
    report: dict[str, Any] = {
        "version": "nlp-data-efficient-heads-screen-v1",
        "experiment_id": config["experiment_id"],
        "status": "in_progress",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "preflight_report_sha256": _sha(preflight_path),
        "data_config_sha256": _sha(data_config_path),
        "seed": seed,
        "train_rows": len(inputs["train_rows"]),
        "validation_fragments": len(inputs["validation_rows"]),
        "grammar_minimum_nodes": minimums,
        "device": str(device),
        "arms": {},
        "test_content_read": False,
        "validation_selection": False,
        "retry_or_checkpoint_reuse": False,
        "execution_boundary": config["evaluation"]["execution"],
        "evidence_boundary": config["evidence_boundary"],
    }
    _atomic_json(report_path, report)
    try:
        for arm in config["arms"]:
            arm_id = arm["id"]
            report["arms"][arm_id] = {"status": "training_in_progress", "case_rows": []}
            checkpoint_path = checkpoint_dir / f"{arm_id}.pt"
            checkpoint_progress = _checkpoint_callback(
                checkpoint_path, arm, arm_id, seed, config_path, report, report_path
            )

            model, train_metrics = train_prefix_tree_policy(
                train,
                validation,
                inputs["vocabulary"],
                device=device,
                seed=seed,
                epochs=int(shared["epochs"]),
                batch_rows=int(shared["batch_rows"]),
                hidden_size=int(shared["hidden_size"]),
                layers=int(shared["decoder_layers"]),
                heads=int(shared["attention_heads"]),
                learning_rate=float(shared["learning_rate"]),
                warmup_steps=int(shared["warmup_steps"]),
                gradient_clip_norm=float(shared["gradient_clip_norm"]),
                local_transition_mask=bool(shared["local_transition_mask"]),
                lexical_candidate_mask=bool(shared["lexical_candidate_mask"]),
                blueprint_conditioning=bool(shared["blueprint_conditioning"]),
                node_kind_rank=arm["node_kind_rank"],
                budget_residual=bool(arm["budget_residual"]),
                initial_state_dict=initial_state,
                progress_callback=checkpoint_progress,
                deadline=deadline,
            )
            torch.save(
                {"model": model.state_dict(), "arm": arm, "seed": seed,
                 "config_sha256": _sha(config_path), "quality_status": "unqualified"},
                checkpoint_path,
            )
            checkpoint_path.with_suffix(".partial.pt").unlink(missing_ok=True)
            evaluation = _case_metrics(
                model, inputs["validation_rows"], inputs, budgets, config, device, report,
                arm_id, deadline,
            )
            report["arms"][arm_id].update(
                {
                    "status": "completed_unqualified",
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "checkpoint_sha256": _sha(checkpoint_path),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "teacher_forced_metrics": train_metrics,
                    "free_running_metrics": {key: value for key, value in evaluation.items() if key != "case_rows"},
                    "case_rows": evaluation["case_rows"],
                }
            )
            _atomic_json(report_path, report)
        report["status"] = "completed_unqualified"
        report["wall_time_seconds"] = time.monotonic() - started
    except (KeyboardInterrupt, Exception) as error:
        report["status"] = "budget_stopped_incomplete" if type(error).__name__ == "BudgetDeadlineExceeded" else "interrupted_or_failed_incomplete"
        report["stop_reason"] = f"{type(error).__name__}: {error}"
        report["wall_time_seconds"] = time.monotonic() - started
        raise
    finally:
        _atomic_json(report_path, report)
    return report


__all__ = ["run_screen"]
