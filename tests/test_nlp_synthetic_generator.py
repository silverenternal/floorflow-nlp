from __future__ import annotations

from floorflow.nlp_flow.synthetic import generate_tasks, load_tasks, write_tasks


def test_generator_is_seed_deterministic_and_covers_all_task_kinds() -> None:
    first = generate_tasks(17, 16)
    second = generate_tasks(17, 16)
    changed = generate_tasks(18, 16)

    assert first == second
    assert first != changed
    assert {task.kind for task in first} == {
        "plan",
        "segment",
        "symbol",
        "constraint",
    }


def test_serialization_is_stable_and_input_does_not_leak_target_keys(tmp_path) -> None:
    tasks = generate_tasks(23, 8)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    assert write_tasks(tasks, first) == write_tasks(tasks, second)
    assert load_tasks(first) == tasks
    for task in tasks:
        assert all(key not in task.model_input for key in task.target)
