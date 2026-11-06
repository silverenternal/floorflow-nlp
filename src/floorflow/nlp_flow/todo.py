"""Dependency and evidence audit for the nlp_flow research todo."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.protocol import EXPECTED_STAGES

ALLOWED_STATUSES = frozenset(
    {
        "passed",
        "in_progress",
        "ready",
        "blocked",
        "conditional",
        "planned",
        "rejected",
    }
)
ACTIVE_STATUSES = frozenset({"in_progress", "ready"})
TASK_ID = re.compile(r"^NLP-\d{3}-[A-Z0-9-]+$")
WORK_PACKAGE_ID = re.compile(r"^WP-\d{3}-[A-Z]$")
TASK_FIELDS = (
    "id",
    "phase",
    "priority",
    "status",
    "depends_on",
    "title",
    "owner",
    "gpu",
    "change",
    "inputs",
    "deliverables",
    "gate",
    "stop_rules",
)
WORK_PACKAGE_FIELDS = (
    "id",
    "task_id",
    "order",
    "title",
    "status",
    "effort",
    "outputs",
    "acceptance",
)


def _strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def _cycles(tasks: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    errors: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str, path: list[str]) -> None:
        if task_id in visiting:
            start = path.index(task_id) if task_id in path else 0
            errors.append(path[start:] + [task_id])
            return
        if task_id in visited:
            return
        visiting.add(task_id)
        path.append(task_id)
        for dependency in tasks[task_id].get("depends_on", []):
            if dependency in tasks:
                visit(dependency, path)
        path.pop()
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id, [])
    return errors


def validate_todo(
    payload: Mapping[str, Any],
    root: Path,
    *,
    check_artifacts: bool = True,
) -> tuple[str, ...]:
    """Return all research-task graph, status, and evidence violations."""

    errors: list[str] = []
    if not payload.get("version"):
        errors.append("version is required")
    if not payload.get("status"):
        errors.append("status is required")
    phases = payload.get("phases")
    if not isinstance(phases, list) or tuple(phases) != EXPECTED_STAGES:
        errors.append("phases must exactly match EXPECTED_STAGES")
        phase_order = {phase: index for index, phase in enumerate(EXPECTED_STAGES)}
    else:
        phase_order = {phase: index for index, phase in enumerate(phases)}

    vocabulary = payload.get("status_vocabulary")
    if not isinstance(vocabulary, Mapping) or not set(vocabulary) >= ALLOWED_STATUSES:
        errors.append("status_vocabulary must define every supported status")

    anti_scaling = payload.get("anti_scaling_contract")
    if not isinstance(anti_scaling, Mapping):
        errors.append("anti_scaling_contract must be an object")
    else:
        claim_cap = anti_scaling.get("primary_claim_parameter_cap")
        absolute_cap = anti_scaling.get("absolute_experiment_parameter_cap")
        if not isinstance(claim_cap, int) or claim_cap > 23_000_000:
            errors.append("primary claim cap must not exceed 23M")
        if not isinstance(absolute_cap, int) or absolute_cap > 46_000_000:
            errors.append("absolute experiment cap must not exceed 46M")
        if anti_scaling.get("net_parameter_growth_allowed") is not False:
            errors.append("net parameter growth must be disabled")
        efficiency = anti_scaling.get("required_efficiency_metrics")
        if not isinstance(efficiency, list) or len(efficiency) < 4:
            errors.append("anti-scaling efficiency metrics are incomplete")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        return tuple(errors)
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            errors.append(f"tasks[{index}] must be an object")
            continue
        missing = [field for field in TASK_FIELDS if field not in task]
        if missing:
            errors.append(f"tasks[{index}] is missing {missing}")
            continue
        task_id = str(task["id"])
        if not TASK_ID.fullmatch(task_id):
            errors.append(f"invalid task id: {task_id}")
        if task_id in by_id:
            errors.append(f"duplicate task id: {task_id}")
        by_id[task_id] = task

        if task["status"] not in ALLOWED_STATUSES:
            errors.append(f"{task_id} has unsupported status {task['status']}")
        if task["phase"] not in phase_order:
            errors.append(f"{task_id} has unknown phase {task['phase']}")
        if task["priority"] not in {"P0", "P1", "P2"}:
            errors.append(f"{task_id} has invalid priority {task['priority']}")
        if not isinstance(task["depends_on"], list) or not all(
            isinstance(dependency, str) for dependency in task["depends_on"]
        ):
            errors.append(f"{task_id}.depends_on must be a list of task ids")
        for field in ("change", "deliverables", "stop_rules"):
            if not _strings(task[field]):
                errors.append(f"{task_id}.{field} must be a non-empty string list")
        inputs = task["inputs"]
        if not isinstance(inputs, list) or not all(isinstance(item, str) and item for item in inputs):
            errors.append(f"{task_id}.inputs must be a string list")
        gate = task["gate"]
        if not isinstance(gate, Mapping) or not _strings(gate.get("pass")) or not _strings(gate.get("fail_if")):
            errors.append(f"{task_id}.gate must define non-empty pass and fail_if lists")
        if (
            task["status"] == "passed"
            and not isinstance(task.get("current_results"), Mapping)
        ):
            errors.append(f"{task_id} passed without current_results")

    for task_id, task in by_id.items():
        dependencies = task.get("depends_on", [])
        if task_id in dependencies:
            errors.append(f"{task_id} depends on itself")
        undefined = [dependency for dependency in dependencies if dependency not in by_id]
        if undefined:
            errors.append(f"{task_id} has undefined dependencies {undefined}")
        if task["status"] in {"passed", "ready", "in_progress"}:
            unmet = [
                dependency
                for dependency in dependencies
                if dependency in by_id and by_id[dependency]["status"] != "passed"
            ]
            if unmet:
                errors.append(f"{task_id} is {task['status']} with unmet dependencies {unmet}")
        if task["phase"] in phase_order:
            later = [
                dependency
                for dependency in dependencies
                if dependency in by_id
                and by_id[dependency]["phase"] in phase_order
                and phase_order[by_id[dependency]["phase"]] > phase_order[task["phase"]]
            ]
            if later:
                errors.append(f"{task_id} depends on later phases {later}")
        if task["status"] == "passed" and check_artifacts:
            artifacts = task.get("artifacts")
            if not _strings(artifacts):
                errors.append(f"{task_id} passed without concrete artifacts")
            else:
                missing_artifacts = [artifact for artifact in artifacts if not (root / artifact).exists()]
                if missing_artifacts:
                    errors.append(f"{task_id} has missing artifacts {missing_artifacts}")

    for cycle in _cycles(by_id):
        errors.append(f"task dependency cycle: {' -> '.join(cycle)}")

    work_packages = payload.get("work_packages")
    packages_by_id: dict[str, Mapping[str, Any]] = {}
    packages_by_task: dict[str, list[Mapping[str, Any]]] = {}
    if not isinstance(work_packages, list) or not work_packages:
        errors.append("work_packages must be a non-empty list")
        work_packages = []
    for index, package in enumerate(work_packages):
        if not isinstance(package, Mapping):
            errors.append(f"work_packages[{index}] must be an object")
            continue
        missing = [field for field in WORK_PACKAGE_FIELDS if field not in package]
        if missing:
            errors.append(f"work_packages[{index}] is missing {missing}")
            continue
        package_id = str(package["id"])
        task_id = str(package["task_id"])
        if not WORK_PACKAGE_ID.fullmatch(package_id):
            errors.append(f"invalid work package id: {package_id}")
        if package_id in packages_by_id:
            errors.append(f"duplicate work package id: {package_id}")
        if task_id not in by_id:
            errors.append(f"{package_id} references unknown task {task_id}")
        if package["status"] not in ALLOWED_STATUSES:
            errors.append(f"{package_id} has unsupported status {package['status']}")
        if package["effort"] not in {"XS", "S", "M", "L", "XL"}:
            errors.append(f"{package_id} has invalid effort {package['effort']}")
        if not isinstance(package["order"], int) or package["order"] <= 0:
            errors.append(f"{package_id}.order must be a positive integer")
        if not _strings(package["outputs"]):
            errors.append(f"{package_id}.outputs must be a non-empty string list")
        if not _strings(package["acceptance"]):
            errors.append(f"{package_id}.acceptance must be a non-empty string list")
        packages_by_id[package_id] = package
        packages_by_task.setdefault(task_id, []).append(package)

    for task_id, task in by_id.items():
        task_packages = packages_by_task.get(task_id, [])
        if task["status"] != "passed" and not task_packages:
            errors.append(f"{task_id} has no work packages")
        orders = [package["order"] for package in task_packages]
        if len(orders) != len(set(orders)):
            errors.append(f"{task_id} has duplicate work-package order values")
        ordered_packages = sorted(task_packages, key=lambda package: package["order"])
        unfinished_packages = [
            package for package in ordered_packages if package["status"] != "passed"
        ]
        if task["status"] == "ready" and unfinished_packages and not any(
            package["status"] in ACTIVE_STATUSES for package in unfinished_packages
        ):
            errors.append(f"{task_id} is ready but has no active work package")
        if task["status"] == "passed":
            unfinished = [
                package["id"]
                for package in task_packages
                if package["status"] != "passed"
            ]
            if unfinished:
                errors.append(f"{task_id} passed with unfinished work packages {unfinished}")

    milestones = payload.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        errors.append("milestones must be a non-empty list")
    else:
        milestone_ids: set[str] = set()
        for index, milestone in enumerate(milestones):
            if not isinstance(milestone, Mapping):
                errors.append(f"milestones[{index}] must be an object")
                continue
            milestone_id = str(milestone.get("id", ""))
            if not milestone_id or milestone_id in milestone_ids:
                errors.append(f"milestones[{index}] has a missing or duplicate id")
            milestone_ids.add(milestone_id)
            if milestone.get("status") not in ALLOWED_STATUSES:
                errors.append(f"{milestone_id} has unsupported status")
            milestone_tasks = milestone.get("tasks")
            if not _strings(milestone_tasks):
                errors.append(f"{milestone_id}.tasks must be non-empty")
            else:
                unknown = [task_id for task_id in milestone_tasks if task_id not in by_id]
                if unknown:
                    errors.append(f"{milestone_id} references unknown tasks {unknown}")
            if not _strings(milestone.get("acceptance")):
                errors.append(f"{milestone_id}.acceptance must be non-empty")

    decision_points = payload.get("decision_points")
    if not isinstance(decision_points, list) or not decision_points:
        errors.append("decision_points must be a non-empty list")
    else:
        decision_ids: set[str] = set()
        for index, decision in enumerate(decision_points):
            if not isinstance(decision, Mapping):
                errors.append(f"decision_points[{index}] must be an object")
                continue
            decision_id = str(decision.get("id", ""))
            if not decision_id or decision_id in decision_ids:
                errors.append(f"decision_points[{index}] has a missing or duplicate id")
            decision_ids.add(decision_id)
            if decision.get("status") not in {"pending", "passed", "failed"}:
                errors.append(f"{decision_id} has unsupported status")
            after_tasks = decision.get("after_tasks")
            if not _strings(after_tasks):
                errors.append(f"{decision_id}.after_tasks must be non-empty")
            else:
                unknown = [task_id for task_id in after_tasks if task_id not in by_id]
                if unknown:
                    errors.append(f"{decision_id} references unknown tasks {unknown}")
            for field in ("question", "pass_action", "fail_action"):
                if not isinstance(decision.get(field), str) or not decision[field]:
                    errors.append(f"{decision_id}.{field} must be non-empty")

    current = payload.get("current_round")
    if not isinstance(current, Mapping):
        errors.append("current_round must be an object")
    else:
        active = current.get("active_tasks")
        if not isinstance(active, list) or not all(
            isinstance(task_id, str) and task_id for task_id in active
        ):
            errors.append("current_round.active_tasks must be non-empty")
        else:
            invalid = [
                task_id
                for task_id in active
                if task_id not in by_id or by_id[task_id].get("status") not in ACTIVE_STATUSES
            ]
            if invalid:
                errors.append(f"active tasks are missing or inactive: {invalid}")
            if current.get("gpu_authorized") is not True:
                gpu_active = [
                    task_id
                    for task_id in active
                    if task_id in by_id and str(by_id[task_id].get("gpu", "")).startswith("cuda")
                ]
                if gpu_active:
                    errors.append(f"GPU tasks active without authorization: {gpu_active}")
            active_packages = current.get("active_work_packages")
            if not isinstance(active_packages, list) or not all(
                isinstance(package_id, str) and package_id
                for package_id in active_packages
            ):
                errors.append("current_round.active_work_packages must be non-empty")
            else:
                invalid_packages = [
                    package_id
                    for package_id in active_packages
                    if package_id not in packages_by_id
                    or packages_by_id[package_id]["status"] not in ACTIVE_STATUSES
                ]
                if invalid_packages:
                    errors.append(f"active work packages are missing or inactive: {invalid_packages}")
                wrong_task = [
                    package_id
                    for package_id in active_packages
                    if package_id in packages_by_id
                    and packages_by_id[package_id]["task_id"] not in active
                ]
                if wrong_task:
                    errors.append(f"active work packages belong to inactive tasks: {wrong_task}")

    budgets = payload.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets must be an object")
    else:
        tiers = budgets.get("tiers")
        if not isinstance(tiers, list) or not tiers:
            errors.append("budgets.tiers must be non-empty")
        else:
            for tier in tiers:
                if not isinstance(tier, Mapping):
                    errors.append("every budget tier must be an object")
                    continue
                parameters = tier.get("parameters")
                tokens = tier.get("corpus_tokens")
                if not isinstance(parameters, int) or not isinstance(tokens, int):
                    errors.append(f"budget tier {tier.get('id')} has non-integer counts")
                elif tokens != parameters * 20:
                    errors.append(f"budget tier {tier.get('id')} violates the 20:1 ratio")
    return tuple(errors)


def ready_task_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = payload.get("tasks", [])
    status_by_id = {task["id"]: task["status"] for task in tasks}
    return tuple(
        task["id"]
        for task in tasks
        if task["status"] == "ready"
        and all(status_by_id.get(dependency) == "passed" for dependency in task["depends_on"])
    )


__all__ = ["ALLOWED_STATUSES", "ready_task_ids", "validate_todo"]
