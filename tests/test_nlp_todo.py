from __future__ import annotations

import copy
import json
from pathlib import Path

from floorflow.nlp_flow.todo import ready_task_ids, validate_todo

ROOT = Path(__file__).resolve().parents[1]


def load_todo() -> dict:
    return json.loads((ROOT / "nlptodo.json").read_text())


def test_nlp_todo_graph_and_evidence_are_valid() -> None:
    payload = load_todo()

    assert validate_todo(payload, ROOT, check_artifacts=True) == ()
    active = set(payload["current_round"]["active_tasks"])
    statuses = {task["id"]: task["status"] for task in payload["tasks"]}
    assert set(ready_task_ids(payload)) <= active
    assert all(statuses[task_id] in {"ready", "in_progress"} for task_id in active)
    assert {task_id for milestone in payload["milestones"] for task_id in milestone["tasks"]} == set(statuses)
    assert {package["task_id"] for package in payload["work_packages"]} == set(statuses)
    active_packages = set(payload["current_round"]["active_work_packages"])
    if payload["current_round"]["status"] == "implementation_frozen_gpu_deferred":
        assert active == set()
        assert active_packages == set()
        assert all(
            task["status"] == "passed"
            for task in payload["tasks"]
            if task["execution_kind"] == "cpu_implementation"
        )
    else:
        assert active_packages
    assert all(
        package["task_id"] in active and package["status"] in {"ready", "in_progress"}
        for package in payload["work_packages"]
        if package["id"] in active_packages
    )


def test_nlp_todo_defers_gpu_experiments_until_implementation_is_integrated() -> None:
    payload = load_todo()
    tasks = {task["id"]: task for task in payload["tasks"]}
    contract = payload["implementation_first_contract"]
    implementation = set(contract["implementation_task_ids"])
    experiments = set(contract["gpu_experiment_task_ids"])
    barrier = contract["implementation_barrier_task"]

    assert implementation.isdisjoint(experiments)
    assert implementation | experiments == set(tasks)
    assert barrier in implementation
    gpu_active = [
        task_id
        for task_id in payload["current_round"]["active_tasks"]
        if tasks[task_id]["gpu"].startswith("cuda:")
    ]
    assert payload["current_round"]["gpu_authorized"] is bool(gpu_active)
    for task_id in implementation:
        assert tasks[task_id]["gpu"] == "none"
        assert set(tasks[task_id]["depends_on"]) <= implementation
    for task_id in experiments:
        assert tasks[task_id]["gpu"].startswith("cuda:")
        ancestors: set[str] = set()
        pending = list(tasks[task_id]["depends_on"])
        while pending:
            dependency = pending.pop()
            if dependency not in ancestors:
                ancestors.add(dependency)
                pending.extend(tasks[dependency]["depends_on"])
        assert barrier in ancestors


def test_nlp_todo_removes_rejected_architectures_and_freezes_exact_ast_gate() -> None:
    payload = load_todo()
    task_ids = {task["id"] for task in payload["tasks"]}

    assert "NLP-055-CFM-FAST-QUALITY" not in task_ids
    assert "NLP-064-PREFIX-TRANSFORMER" not in task_ids
    assert payload["ast_recovery_contract"]["validation_cases"] == 60
    assert payload["ast_recovery_contract"]["hard_acceptance"]["parser_validity"] == "== 1.0"
    assert payload["ast_recovery_contract"]["hard_acceptance"]["strict_passed"] == "== 1.0"


def test_nlp_todo_rejects_dependency_cycles() -> None:
    payload = copy.deepcopy(load_todo())
    task = next(
        task
        for task in payload["tasks"]
        if task["id"] == "NLP-056-EFFICIENT-ARCH-SEARCH"
    )
    task["depends_on"].append("NLP-017-MATCHED-SCREEN")

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("cycle" in error for error in errors)


def test_nlp_todo_rejects_undefined_dependency() -> None:
    payload = copy.deepcopy(load_todo())
    payload["tasks"][0]["depends_on"] = ["NLP-999-MISSING"]

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("undefined dependencies" in error for error in errors)


def test_nlp_todo_rejects_passed_task_with_missing_artifact() -> None:
    payload = copy.deepcopy(load_todo())
    task = payload["tasks"][0]
    task["status"] = "passed"
    task["current_results"] = {"status": "pass"}
    task["artifacts"] = ["outputs/reports/does-not-exist.json"]

    errors = validate_todo(payload, ROOT, check_artifacts=True)

    assert any("missing artifacts" in error for error in errors)


def test_nlp_todo_rejects_unauthorized_gpu_active_task() -> None:
    payload = copy.deepcopy(load_todo())
    task = next(
        task
        for task in payload["tasks"]
        if task["id"] == "NLP-065-AST-FAILURE-ANALYSIS"
    )
    task["status"] = "in_progress"
    task["gpu"] = "cuda:1"
    package = next(
        package
        for package in payload["work_packages"]
        if package["id"] == "WP-065-A"
    )
    package["status"] = "in_progress"
    payload["current_round"]["active_tasks"] = ["NLP-065-AST-FAILURE-ANALYSIS"]
    payload["current_round"]["active_work_packages"] = ["WP-065-A"]
    payload["current_round"]["gpu_authorized"] = False

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("GPU tasks active without authorization" in error for error in errors)


def test_nlp_todo_rejects_ready_task_without_active_work_package() -> None:
    payload = copy.deepcopy(load_todo())
    for task in payload["tasks"]:
        if task["id"] == "NLP-017-MATCHED-SCREEN":
            task["status"] = "ready"
    for package in payload["work_packages"]:
        if package["task_id"] == "NLP-017-MATCHED-SCREEN":
            package["status"] = "planned"

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("has no active work package" in error for error in errors)


def test_nlp_todo_rejects_active_package_from_inactive_task() -> None:
    payload = copy.deepcopy(load_todo())
    package = next(
        package
        for package in payload["work_packages"]
        if package["id"] == "WP-062-A"
    )
    package["status"] = "ready"
    payload["current_round"]["active_work_packages"].append(package["id"])

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("belong to inactive tasks" in error for error in errors)


def test_nlp_todo_rejects_unknown_milestone_task() -> None:
    payload = copy.deepcopy(load_todo())
    payload["milestones"][0]["tasks"].append("NLP-999-MISSING")

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert any("references unknown tasks" in error for error in errors)


def test_nlp_todo_rejects_large_model_stacking() -> None:
    payload = copy.deepcopy(load_todo())
    payload["anti_scaling_contract"]["primary_claim_parameter_cap"] = 100_000_000
    payload["anti_scaling_contract"]["net_parameter_growth_allowed"] = True

    errors = validate_todo(payload, ROOT, check_artifacts=False)

    assert "primary claim cap must not exceed 23M" in errors
    assert "net parameter growth must be disabled" in errors
