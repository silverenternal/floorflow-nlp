"""Deterministic target-free synthetic mechanism probes."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    kind: str
    model_input: str
    target: dict[str, object]

    def __post_init__(self) -> None:
        if not self.task_id or not self.model_input:
            raise ValueError("task id and model input are required")
        if any(key in self.model_input for key in self.target):
            raise ValueError("model input leaks an oracle target key")


def generate_tasks(seed: int, count: int = 32) -> tuple[SyntheticTask, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    generator = random.Random(seed)
    tasks = []
    kinds = ("plan", "segment", "symbol", "constraint")
    for index in range(count):
        kind = kinds[index % len(kinds)]
        left = generator.randint(0, 20)
        right = generator.randint(1, 10)
        if kind == "plan":
            prompt = f"Plan a program that adds {left} and {right}."
            target = {
                "steps": ["read constants", "add values", "return result"],
                "result": left + right,
            }
        elif kind == "segment":
            prompt = f"Segment a function returning {left + right}."
            target = {
                "segments": ["signature", "body", "return"],
                "return_value": left + right,
            }
        elif kind == "symbol":
            prompt = f"Resolve identifiers for add_{index}(x) returning x + {right}."
            target = {
                "symbols": [f"add_{index}", "x"],
                "constant": right,
            }
        else:
            prompt = f"Produce JSON below {left + right} with a nonempty status."
            target = {
                "required_keys": ["value", "status"],
                "value": left + right - 1,
            }
        tasks.append(
            SyntheticTask(
                task_id=f"synthetic-{index:04d}",
                kind=kind,
                model_input=prompt,
                target=target,
            )
        )
    return tuple(tasks)


def write_tasks(tasks: tuple[SyntheticTask, ...], path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with target.open("w", encoding="utf-8") as handle:
        for task in tasks:
            line = json.dumps(asdict(task), sort_keys=True, ensure_ascii=True)
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
            handle.write(line + "\n")
    return digest.hexdigest()


def load_tasks(path: str | Path) -> tuple[SyntheticTask, ...]:
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            tasks.append(SyntheticTask(**payload))
    return tuple(tasks)


__all__ = ["SyntheticTask", "generate_tasks", "load_tasks", "write_tasks"]
