#!/usr/bin/env python3
"""Audit NLP implementation closure, pain ownership, experiment gates and freeze hashes."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
IMPLEMENTATION_TASKS = {
    "NLP-065-AST-FAILURE-ANALYSIS",
    "NLP-069-CORPUS-AND-REPLAY-CONTRACTS",
    "NLP-066-HIERARCHICAL-STRUCTURE-PLANNER",
    "NLP-067-TYPED-LEXICAL-PROJECTION",
    "NLP-070-ROLLOUT-TRAINING-RUNTIME",
    "NLP-062-PARSED-AST-CONDITIONING",
    "NLP-071-TYPED-OUTPUT-AND-VERIFIER-CONTRACTS",
    "NLP-072-BENCHMARK-AND-SEARCH-PLUMBING",
    "NLP-073-BOUNDED-COMPETENCE-AND-BUDGETS",
    "NLP-074-IMPLEMENTATION-INTEGRATION",
}
BARRIER_ID = "NLP-074-IMPLEMENTATION-INTEGRATION"
MUTABLE_RUN_ARTIFACTS = {
    "project/configs/other/general/gpu_queue/nlp_gpu_experiment_queue_closed_after_failure.json",
    "outputs/reports/nlp_tree_training_launch_preflight_v11.json",
    "project/configs/other/general/gpu_queue/nlp_gpu_experiment_queue_corrective_registration.json",
    "outputs/reports/nlp_strict_ast_launch_preflight_v12.json",
}
CORRECTIVE_EXPERIMENT_ARTIFACTS = {
    "project/configs/structured_generation/tree_action_corpus/nlp_tree_training_corrected_contract.json",
    "project/configs/structured_generation/structure_matched_strict/ast_strict/nlp_strict_ast_corrective_run.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _todo_contract_digest(todo: dict[str, Any]) -> str:
    value = copy.deepcopy(todo)
    for task in value.get("tasks", []):
        task.pop("status", None)
        task.pop("current_results", None)
    for package in value.get("work_packages", []):
        package.pop("status", None)
        package.pop("evidence", None)
    for pain in value.get("pain_points", []):
        pain.pop("status", None)
    return _canonical_digest(value)


def _gpu_descends_from_barrier(task_id: str, tasks: dict[str, dict[str, Any]], trail: set[str]) -> bool:
    if task_id == BARRIER_ID:
        return True
    if task_id in trail:
        return False
    trail.add(task_id)
    task = tasks.get(task_id)
    return bool(task) and any(
        _gpu_descends_from_barrier(dependency, tasks, trail.copy())
        for dependency in task.get("depends_on", [])
    )


def _has_stub(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        if (
            len(body) == 1
            and isinstance(body[0], ast.Raise)
            and isinstance(body[0].exc, ast.Name)
            and body[0].exc.id == "NotImplementedError"
        ):
            return True
    return False


def _tokenizer_fingerprint(root: Path) -> dict[str, Any]:
    from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
    from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import load_action_rows
    from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary

    train_path = root / ".data/nlp/multilingual_supplement_v11/train_actions.jsonl"
    grammar_path = root / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_cpp_extended.json"
    rows = [row for row in load_action_rows(train_path) if row.get("split") == "train"]
    grammars = {
        language: load_ast_grammar(grammar_path, language)
        for language in ("python", "rust", "cpp")
    }
    vocabulary = build_vocabulary(rows, grammars)
    implementation = root / "src/newera/nlp_flow/components/generation/policies/corpus/real_corpus_policy_data.py"
    vocabulary_hash = _canonical_digest(vocabulary)
    return {
        "kind": "train_only_tree_action_lexeme_vocabulary_and_fixed_hashed_prompt_tokens",
        "training_rows": len(rows),
        "training_action_sha256": _sha(train_path),
        "grammar_sha256": _sha(grammar_path),
        "vocabulary_sha256": vocabulary_hash,
        "prompt_tokenizer_implementation_sha256": _sha(implementation),
        "prompt_token_buckets": 32768,
        "learned_subword_tokenizer": False,
    }


def _fingerprints(root: Path) -> dict[str, Any]:
    from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_contract import (
        canonicalizer_runtime_identity,
        parser_runtime_identity,
    )

    parsers = {}
    for language in ("python", "rust", "cpp"):
        version, digest = parser_runtime_identity(language)
        parsers[language] = {"version": version, "sha256": digest}
    canonicalizer_version, canonicalizer_hash = canonicalizer_runtime_identity()
    train_path = root / ".data/nlp/multilingual_supplement_v11/train_actions.jsonl"
    metadata_path = root / ".data/nlp/multilingual_supplement_v11/metadata.jsonl"
    validation_manifest = root / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_legacy60_eval_manifest.json"
    evaluator_paths = [
        "src/newera/nlp_flow/components/research/evaluation/manifests/legacy60_adapter.py",
        "src/newera/nlp_flow/components/ast/representation/codec/ast_codec_metrics.py",
        "src/newera/nlp_flow/components/ast/diagnostics/strict/strict_ast_audit.py",
        "project/scripts/audit/ast/validation/audit_nlp_strict_ast_closure.py",
    ]
    return {
        "corpus": {
            "training_actions_sha256": _sha(train_path),
            "metadata_sha256": _sha(metadata_path),
            "legacy60_validation_manifest_sha256": _sha(validation_manifest),
        },
        "tokenizer": _tokenizer_fingerprint(root),
        "parser": parsers,
        "canonicalizer": {"version": canonicalizer_version, "sha256": canonicalizer_hash},
        "evaluator": {
            path: _sha(root / path)
            for path in evaluator_paths
        },
    }


def build_readiness(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    todo = json.loads((root / "nlptodo.json").read_text(encoding="utf-8"))
    tasks = {row["id"]: row for row in todo["tasks"]}
    packages_by_task: dict[str, list[dict[str, Any]]] = {}
    for package in todo.get("work_packages", []):
        packages_by_task.setdefault(package.get("task_id", ""), []).append(package)
    errors: list[str] = []
    artifact_paths: set[str] = set()
    for task_id in IMPLEMENTATION_TASKS:
        task = tasks.get(task_id)
        if task is None:
            errors.append(f"missing implementation task {task_id}")
            continue
        if task.get("execution_kind") != "cpu_implementation":
            errors.append(f"{task_id} is not declared as CPU implementation")
        if task.get("status") != "passed":
            errors.append(f"implementation task is not passed: {task_id}")
        owner = task.get("owner")
        if not owner:
            errors.append(f"implementation task has no owner: {task_id}")
        packages = packages_by_task.get(task_id, [])
        if not packages or any(package.get("status") != "passed" for package in packages):
            errors.append(f"implementation work packages are incomplete: {task_id}")
        for package in packages:
            artifact_paths.update(package.get("outputs", []))
            for artifact in package.get("outputs", []):
                if not (root / artifact).is_file():
                    errors.append(f"missing implementation artifact: {artifact}")
    for artifact in CORRECTIVE_EXPERIMENT_ARTIFACTS:
        if (root / artifact).is_file():
            artifact_paths.add(artifact)
        else:
            errors.append(f"missing corrective experiment registration: {artifact}")
    pain_coverage = []
    for pain in todo.get("pain_points", []):
        implementation_tasks = pain.get("implementation_tasks", [])
        work_package_ids = set(pain.get("implementation_work_packages", []))
        owners = sorted({tasks[item].get("owner") for item in implementation_tasks if item in tasks and tasks[item].get("owner")})
        artifacts = sorted(
            {
                artifact
                for package in todo.get("work_packages", [])
                if package.get("id") in work_package_ids
                for artifact in package.get("outputs", [])
            }
        )
        if not implementation_tasks or not work_package_ids or not owners or not artifacts:
            errors.append(f"pain point lacks concrete owner/artifact mapping: {pain.get('id')}")
        pain_coverage.append(
            {
                "pain_id": pain["id"],
                "implementation_tasks": implementation_tasks,
                "work_packages": sorted(work_package_ids),
                "owners": owners,
                "artifacts": artifacts,
                "deferred_experiment_tasks": pain.get("deferred_experiment_tasks", []),
            }
        )
    for pain in todo.get("pain_points", []):
        if not pain.get("strengthening_design") or not pain.get("deferred_experiment_tasks"):
            errors.append(f"pain point lacks design or deferred experiment owner: {pain['id']}")
        for deferred_id in pain.get("deferred_experiment_tasks", []):
            deferred = tasks.get(deferred_id)
            if deferred is None or deferred.get("execution_kind") != "deferred_gpu_experiment":
                errors.append(f"pain point has an invalid deferred experiment owner: {pain['id']}:{deferred_id}")

    gpu_tasks = [task for task in todo["tasks"] if task.get("gpu") != "none"]
    for task in gpu_tasks:
        if not _gpu_descends_from_barrier(task["id"], tasks, set()):
            errors.append(f"GPU task does not depend transitively on the implementation barrier: {task['id']}")
    for task in todo["tasks"]:
        if (
            task.get("execution_kind") == "cpu_implementation"
            and any(tasks.get(dep, {}).get("gpu") != "none" for dep in task.get("depends_on", []))
        ):
            errors.append(f"CPU implementation depends on GPU work: {task['id']}")

    queue = json.loads((root / "project/configs/other/general/gpu_queue/nlp_gpu_experiment_queue_closed_after_failure.json").read_text())
    training = json.loads((root / "project/configs/structured_generation/tree_action_corpus/nlp_tree_training_legacy_preflight.json").read_text())
    if queue.get("training_allowed_now") is not False or queue.get("quality_evaluation_allowed_now") is not False:
        errors.append("GPU experiment queue is open before the freeze")
    if training.get("execution", {}).get("training_enabled") is not False:
        errors.append("training is enabled in the implementation phase")
    if any(path.exists() for path in (root / "outputs/.locks/nlp_gpu_training.lock",)):
        errors.append("GPU training lock is held")

    source_artifacts = sorted(
        path
        for path in artifact_paths
        if Path(path).suffix in {".py", ".rs"}
    )
    source_lengths = {}
    for relative in source_artifacts:
        path = root / relative
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        source_lengths[relative] = line_count
        if line_count >= 600:
            errors.append(f"source file exceeds 599 lines: {relative} ({line_count})")
        if _has_stub(path):
            errors.append(f"implementation artifact contains an empty stub: {relative}")

    report = {
        "version": "nlp-implementation-readiness-v11",
        "status": "passed" if not errors else "rejected",
        "implementation_tasks": sorted(IMPLEMENTATION_TASKS),
        "implementation_artifact_count": len(artifact_paths),
        "pain_point_count": len(pain_coverage),
        "pain_coverage": pain_coverage,
        "gpu_task_count": len(gpu_tasks),
        "all_gpu_tasks_descend_from_barrier": not any("does not depend transitively" in error for error in errors),
        "source_line_counts": source_lengths,
        "quality_status": "unverified",
        "gpu_runs_performed": False,
        "errors": errors,
    }
    fingerprints = _fingerprints(root)
    normalized = _todo_contract_digest(todo)
    source_config_paths = sorted(
        path
        for path in artifact_paths
        if (root / path).is_file()
        and Path(path).suffix in {".py", ".rs", ".json", ".md"}
        and path not in {
            "project/configs/other/contracts_runtime/freeze_implementation/nlp_implementation_freeze_initial.json",
            "project/configs/other/contracts_runtime/freeze_implementation/nlp_implementation_freeze_corrected.json",
            "outputs/reports/nlp_implementation_readiness_v11.json",
            "outputs/reports/nlp_implementation_readiness_v12.json",
        }
        and path not in MUTABLE_RUN_ARTIFACTS
    )
    freeze = {
        "version": "nlp-implementation-freeze-v11",
        "status": report["status"],
        "todo_contract_sha256": normalized,
        "artifacts": {path: _sha(root / path) for path in source_config_paths},
        "self_referential_outputs_excluded": [
            "project/configs/other/contracts_runtime/freeze_implementation/nlp_implementation_freeze_initial.json",
            "project/configs/other/contracts_runtime/freeze_implementation/nlp_implementation_freeze_corrected.json",
            "outputs/reports/nlp_implementation_readiness_v11.json",
            "outputs/reports/nlp_implementation_readiness_v12.json",
        ],
        "mutable_execution_artifacts_excluded": sorted(MUTABLE_RUN_ARTIFACTS),
        "fingerprints": fingerprints,
        "supported_languages": ["python", "rust", "cpp"],
        "supported_tracks": {
            "reconstruction_recovery": "internal_only_quality_unverified",
            "user_task_generation": "interface_implemented_quality_unverified",
        },
        "exact_cpu_commands": [
            "uv run pytest project/tests/other/contracts_runtime/implementation_flow_integration/test_nlp_implementation_integration.py -q",
            "uv run pytest tests -q",
            "uvx ruff check src/newera/nlp_flow scripts tests",
            "uv run python project/scripts/audit/engineering/contracts/audit_nlp_implementation_readiness.py --report outputs/reports/nlp_implementation_readiness_v11.json",
        ],
        "deferred_experiments": [
            "NLP-068-STRICT-AST-CLOSURE",
            "NLP-044-TWO-STAGE-GATE",
            "NLP-054-BENCHMARK-PROXY",
            "NLP-056-EFFICIENT-ARCH-SEARCH",
            "NLP-075-CONVERSATION-AND-COMPACT-SPECIALIZATION",
            "NLP-076-FROZEN-EXTERNAL-EVALUATION",
            "NLP-017-MATCHED-SCREEN",
        ],
        "unresolved_risks": [
            "No trained checkpoint or quality result is included.",
            "Multilingual supplement has short-fragment near-overlap and external benchmark scan gaps.",
            "C++ and user-task generation quality remain unqualified.",
            "GPU training and all deferred quality experiments remain closed until this barrier is accepted.",
        ],
        "quality_claim": "none",
    }
    return report, freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    report, freeze = build_readiness()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.freeze.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "pain_coverage"}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
