#!/usr/bin/env python3
"""Capture target-free frozen-suite traces and explicitly tagged oracle projections."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.ast.diagnostics.recovery.ast_oracle_controls import oracle_projection
from newera.nlp_flow.components.research.evaluation.manifests.legacy60_adapter import evaluate_legacy60_prediction
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_runtime import generate_prefix_tree
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import action_from_record
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import decode_actions, render_tree_structure

ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation_rows(path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    wanted = set(case_ids)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) != "validation":
                continue
            row = json.loads(line)
            if row.get("split") != "validation":
                raise ValueError("validation split marker differs from decoded row")
            case_id = str(row["case_id"])
            if case_id in wanted:
                rows[case_id] = row
    if set(rows) != wanted:
        raise ValueError("validation rows differ from the frozen identity list")
    return [rows[case_id] for case_id in case_ids]


def _leaves(tree: dict[str, Any]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for node in tree["nodes"]:
        if not node["children"]:
            key = (str(node["kind"]), str(node.get("leaf_class")))
            groups.setdefault(key, []).append(int(node["index"]))
    return groups


def typed_leaf_alignment(target_topology: dict[str, Any], lexical_source: dict[str, Any]) -> dict[int, int] | None:
    """Build an occurrence-ordered bijection over exact typed leaf groups."""
    if target_topology.get("language") != lexical_source.get("language"):
        return None
    target_groups, source_groups = _leaves(target_topology), _leaves(lexical_source)
    if target_groups.keys() != source_groups.keys():
        return None
    alignment: dict[int, int] = {}
    for key in sorted(target_groups):
        target_ids, source_ids = target_groups[key], source_groups[key]
        if len(target_ids) != len(source_ids):
            return None
        alignment.update(zip(target_ids, source_ids, strict=True))
    return alignment


def _gate(source: str | None, *, case_id: str, reference: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    if source is None:
        return {"case_id": case_id, "status": "unavailable", "strict_passed": False, "metrics": None}
    try:
        prediction = serialize_ast_graph(parse_ast_graph(source, reference.language))
    except (TypeError, ValueError) as error:
        return {
            "case_id": case_id,
            "status": "invalid",
            "strict_passed": False,
            "metrics": None,
            "parse_error": str(error),
        }
    return evaluate_legacy60_prediction(
        prediction,
        case_id=case_id,
        reference_graph=reference,
        manifest=manifest,
    )


def _checkpoint_model(path: Path, seed: int, device: torch.device) -> tuple[PrefixTreePolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint seed identity mismatch: {seed}")
    config, vocabulary = checkpoint["config"], checkpoint["vocabulary"]
    max_steps = int(checkpoint["model"]["step_position_embedding.weight"].shape[0])
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=int(config["hidden_size"]),
        layers=int(config["decoder_layers"]),
        heads=int(config["decoder_heads"]),
        max_steps=max_steps,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocabulary


def _trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    for event in trace:
        if "step" not in event:
            continue
        action = event.get("action")
        decisions.append(
            {
                "step": event["step"],
                "action": None if not isinstance(action, dict) else action.get("type"),
                "error": event.get("error"),
                "choices": [
                    {
                        "head": choice.get("head"),
                        "selected_id": choice.get("selected_id"),
                        "allowed_ids": choice.get("allowed_ids"),
                    }
                    for choice in event.get("choices", [])
                ],
            }
        )
    terminal = next((event for event in trace if event.get("event") == "generation_end"), {})
    return {"decision_count": len(decisions), "decisions": decisions, "terminal_error": terminal.get("error")}


def run_decomposition(
    *, manifest: dict[str, Any], rows: list[dict[str, Any]], checkpoints: dict[int, Path],
    grammar_path: Path, device: torch.device, max_seconds_per_seed: int,
) -> dict[str, Any]:
    if manifest.get("split") != "validation" or manifest.get("test_content_evaluated") is not False:
        raise ValueError("oracle diagnostic requires the untouched validation-only manifest")
    if [row["case_id"] for row in rows] != manifest["ordered_case_ids"]:
        raise ValueError("validation order differs from the frozen manifest")
    manifest_impl = hashlib.sha256(
        json.dumps(manifest["implementation_sha256"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    seed_results = []
    for seed, checkpoint_path in sorted(checkpoints.items()):
        seed_started = time.monotonic()
        model, vocabulary = _checkpoint_model(checkpoint_path, seed, device)
        cases = []
        for row, frozen in zip(rows, manifest["cases"], strict=True):
            if time.monotonic() - seed_started > max_seconds_per_seed:
                raise TimeoutError(f"oracle diagnostics exceeded the per-seed budget: {seed}")
            case_id, language = str(row["case_id"]), str(row["language"])
            if row["target_source_sha256"] != frozen["gold_source_sha256"]:
                raise ValueError(f"gold source identity changed: {case_id}")
            grammar = load_ast_grammar(grammar_path, language)
            teacher_actions = [action_from_record(item) for item in row["actions"]]
            teacher_tree = decode_actions(grammar, teacher_actions)
            teacher_source = render_tree_structure(teacher_tree)
            teacher_graph = parse_ast_graph(teacher_source, language)
            if hashlib.sha256(serialize_ast_graph(teacher_graph).encode()).hexdigest() != frozen["gold_graph_sha256"]:
                raise ValueError(f"teacher graph identity changed: {case_id}")

            trace: list[dict[str, Any]] = []
            generated_source, generated_actions, generation_error = generate_prefix_tree(
                model, str(row["prompt"]), language, grammar, vocabulary,
                device=device, max_steps=int(model.step_position_embedding.num_embeddings), trace=trace,
            )
            generated_tree = None
            if generated_source is not None:
                generated_tree = decode_actions(grammar, generated_actions)
            alignment_teacher_to_generated = (
                None if generated_tree is None else typed_leaf_alignment(teacher_tree, generated_tree)
            )
            alignment_generated_to_teacher = (
                None if generated_tree is None else typed_leaf_alignment(generated_tree, teacher_tree)
            )
            projections: dict[str, Any] = {
                "teacher_topology_teacher_lexical": _gate(
                    teacher_source, case_id=case_id, reference=teacher_graph, manifest=manifest
                ),
                "teacher_topology_candidate_lexical": None,
                "candidate_topology_teacher_lexical": None,
                "candidate_topology_candidate_lexical": _gate(
                    generated_source, case_id=case_id, reference=teacher_graph, manifest=manifest
                ),
            }
            candidate_graph_hash = None
            if generated_source is not None:
                candidate_graph_hash = hashlib.sha256(
                    serialize_ast_graph(parse_ast_graph(generated_source, language)).encode()
                ).hexdigest()
            if alignment_teacher_to_generated is not None:
                oracle_source = render_tree_structure(
                    oracle_projection(teacher_tree, generated_tree, alignment_teacher_to_generated)
                )
                projections["teacher_topology_candidate_lexical"] = _gate(
                    oracle_source, case_id=case_id, reference=teacher_graph, manifest=manifest
                )
            if alignment_generated_to_teacher is not None:
                oracle_source = render_tree_structure(
                    oracle_projection(generated_tree, teacher_tree, alignment_generated_to_teacher)
                )
                projections["candidate_topology_teacher_lexical"] = _gate(
                    oracle_source, case_id=case_id, reference=teacher_graph, manifest=manifest
                )
            cases.append(
                {
                    "case_id": case_id,
                    "language": language,
                    "gold_source_sha256": frozen["gold_source_sha256"],
                    "gold_graph_sha256": frozen["gold_graph_sha256"],
                    "generated_graph_sha256": candidate_graph_hash,
                    "generation_error": generation_error,
                    "generated_actions": len(generated_actions),
                    "live_trace": _trace_summary(trace),
                    "oracle_alignment_coverage": {
                        "teacher_topology_candidate_lexical": alignment_teacher_to_generated is not None,
                        "candidate_topology_teacher_lexical": alignment_generated_to_teacher is not None,
                    },
                    "oracle_projections": projections,
                    "oracle_inputs": [] if generated_source is None else ["frozen_validation_teacher_actions"],
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_results.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha(checkpoint_path),
                "wall_seconds": time.monotonic() - seed_started,
                "case_count": len(cases),
                "baseline_parse_valid": sum(case["generated_graph_sha256"] is not None for case in cases),
                "baseline_strict_passed": sum(
                    bool(case["oracle_projections"]["candidate_topology_candidate_lexical"].get("strict_passed"))
                    for case in cases
                ),
                "oracle_projection_coverage": {
                    key: sum(
                        case["oracle_projections"][key] is not None
                        and case["oracle_projections"][key].get("status") != "unavailable"
                        for case in cases
                    )
                    for key in (
                        "teacher_topology_candidate_lexical", "candidate_topology_teacher_lexical"
                    )
                },
                "cases": cases,
            }
        )
    return {
        "version": "nlp-ast-oracle-decomposition-v12",
        "status": "completed_diagnostic_quality_unverified",
        "split": "validation",
        "test_content_read": False,
        "target_free_generation": True,
        "oracle_controls_used_for_generation": False,
        "oracle_projection_inputs": "frozen validation teacher topology or lexical values; diagnostic projections are never model outputs",
        "manifest_sha256": _sha(ROOT / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_legacy60_eval_manifest.json"),
        "evaluator_identity_sha256": manifest_impl,
        "seed_ids": sorted(checkpoints),
        "cases_per_seed": len(rows),
        "seeds": seed_results,
        "strict_thresholds": manifest["thresholds"],
        "quality_claim": "none; strict thresholds are reported diagnostically and this file does not promote a model",
        "evidence_boundary": "Target-free live traces plus explicit oracle projections over the frozen validation set; no test rows are parsed.",
    }


def _device_rows() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                index, free, total = (int(part.strip()) for part in line.split(","))
            except ValueError:
                continue
            rows[index] = (free, total)
    return rows


def _active_compute_devices() -> set[int]:
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if devices.returncode or processes.returncode:
        return {-1}
    mapping = {}
    for line in devices.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2:
            mapping[fields[1]] = int(fields[0])
    return {
        mapping[fields[0]]
        for line in processes.stdout.splitlines()
        if len(fields := [part.strip() for part in line.split(",")]) >= 2
        and fields[0] in mapping
    }


def build_preflight(registration_path: Path) -> dict[str, Any]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    queue_path = ROOT / str(registration["queue"])
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    blockers = []
    registered = [row for row in queue.get("experiments", []) if row.get("diagnostic_id") == registration["diagnostic_id"]]
    if len(registered) != 1 or registered[0].get("status") != "registered":
        blockers.append("diagnostic_not_registered_exactly_once")
    elif registered[0].get("budget") != registration["budget"]:
        blockers.append("diagnostic_budget_registration_mismatch")
    if queue.get("quality_evaluation_allowed_now") is not True:
        blockers.append("quality_evaluation_not_authorized_by_queue")
    if registration.get("status") != "registered":
        blockers.append("diagnostic_registration_not_frozen")
    if registration.get("test_content_allowed") is not False:
        blockers.append("test_content_must_remain_closed")
    runner_path = ROOT / registration["runner_path"]
    if not runner_path.is_file() or _sha(runner_path) != registration["runner_sha256"]:
        blockers.append("diagnostic_runner_fingerprint_mismatch")
    manifest_path = ROOT / registration["manifest"]
    corpus_path = ROOT / registration["corpus"]
    training_report_path = ROOT / registration["training_report"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if _sha(manifest_path) != registration["manifest_sha256"]:
        blockers.append("frozen_manifest_hash_mismatch")
    if _sha(corpus_path) != registration["corpus_sha256"]:
        blockers.append("validation_corpus_hash_mismatch")
    if training_report.get("status") != "training_completed_quality_unverified":
        blockers.append("all_seed_training_report_not_complete")
    if sorted(int(row["seed"]) for row in training_report.get("seed_reports", [])) != registration["seeds"]:
        blockers.append("trained_seed_set_mismatch")
    report_hashes = {
        str(row["seed"]): row["checkpoint_sha256"]
        for row in training_report.get("seed_reports", [])
    }
    if report_hashes != registration["checkpoint_sha256"]:
        blockers.append("training_report_checkpoint_identity_mismatch")
    checkpoints = {int(seed): ROOT / path for seed, path in registration["checkpoints"].items()}
    for seed, path in checkpoints.items():
        if not path.is_file() or _sha(path) != registration["checkpoint_sha256"][str(seed)]:
            blockers.append(f"checkpoint_fingerprint_mismatch:{seed}")
    marker = ROOT / registration["attempt_marker"]
    if marker.exists():
        blockers.append("registered_diagnostic_attempt_already_consumed")
    lock = ROOT / registration["lock_path"]
    if lock.exists():
        blockers.append("shared_gpu_lock_already_held")
    device = int(registration["device_index"])
    free_memory = _device_rows().get(device, (0, 0))[0]
    active = _active_compute_devices()
    if device in active or -1 in active:
        blockers.append("registered_gpu_has_active_or_unverifiable_compute")
    if free_memory < int(registration["minimum_free_memory_mib"]):
        blockers.append("registered_gpu_below_memory_threshold")
    return {
        "version": "nlp-ast-oracle-diagnostic-preflight-v12",
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "diagnostic_id": registration["diagnostic_id"],
        "seed_set": registration["seeds"],
        "case_count": int(manifest["case_count"]),
        "manifest_sha256": _sha(manifest_path),
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "test_content_read": False,
        "registered_device": device,
        "free_memory_mib": free_memory,
        "active_compute_devices": sorted(active),
        "budget": registration["budget"],
        "launch_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--registration", type=Path, default=ROOT / "project/configs/structured_generation/ast_corpus_diagnostic/ast_diagnostic_freeze/nlp_ast_oracle_diagnostic.json")
    args = parser.parse_args()
    registration_path = args.registration.resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    output = ROOT / registration["output"]
    gate = build_preflight(registration_path)
    if args.dry_run:
        gate["launch_performed"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        preflight_path = ROOT / registration["preflight_report"]
        preflight_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return 0 if gate["status"] == "ready" else 1
    if gate["status"] != "ready":
        raise RuntimeError("oracle diagnostic launch is blocked: " + ", ".join(gate["blockers"]))
    lock_path = ROOT / registration["lock_path"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path = ROOT / registration["attempt_marker"]
    descriptor = lock_path.open("x", encoding="utf-8")
    try:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if int(registration["device_index"]) in _active_compute_devices():
            raise RuntimeError("registered GPU became busy after diagnostic preflight")
        if _device_rows().get(int(registration["device_index"]), (0, 0))[0] < int(registration["minimum_free_memory_mib"]):
            raise RuntimeError("registered GPU fell below the memory threshold after preflight")
        with marker_path.open("x", encoding="utf-8") as handle:
            json.dump({"diagnostic_id": registration["diagnostic_id"], "status": "started", "budget": registration["budget"]}, handle, indent=2)
            handle.write("\n")
        previous_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("oracle diagnostic total budget exceeded")))
        signal.setitimer(signal.ITIMER_REAL, int(registration["budget"]["max_wall_seconds"]))
        started = time.monotonic()
        try:
            manifest_path = ROOT / registration["manifest"]
            corpus_path = ROOT / registration["corpus"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = load_validation_rows(corpus_path, list(manifest["ordered_case_ids"]))
            checkpoints = {int(seed): ROOT / path for seed, path in registration["checkpoints"].items()}
            result = run_decomposition(
                manifest=manifest, rows=rows, checkpoints=checkpoints,
                grammar_path=ROOT / registration["grammar"],
                device=torch.device(f"cuda:{registration['device_index']}"),
                max_seconds_per_seed=int(registration["budget"]["max_wall_seconds_per_seed"]),
            )
            result["wall_seconds"] = time.monotonic() - started
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            marker_path.write_text(json.dumps({"diagnostic_id": registration["diagnostic_id"], "status": "completed", "wall_seconds": result["wall_seconds"]}, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": result["status"], "seed_ids": result["seed_ids"], "wall_seconds": result["wall_seconds"]}, indent=2))
            return 0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    except BaseException as error:
        if marker_path.exists():
            marker_path.write_text(json.dumps({"diagnostic_id": registration["diagnostic_id"], "status": "failed_no_retry", "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        descriptor.close()
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
