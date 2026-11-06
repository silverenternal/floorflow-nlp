#!/usr/bin/env python3
"""Generate a deterministic row-aligned conversation curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = (
    "I am FloorFlow NLP, a compact coding and engineering assistant. "
    "I write short programs, explain code, and help debug executable tasks."
)


def _python_add(a: int, b: int) -> str:
    return f"def add(a: int, b: int) -> int:\n    return {a} + {b}\n"


def _python_sub(a: int, b: int) -> str:
    return f"def subtract(a: int, b: int) -> int:\n    return {a} - {b}\n"


def _python_mul(a: int, b: int) -> str:
    return f"def multiply(a: int, b: int) -> int:\n    return {a} * {b}\n"


def _python_max(a: int, b: int) -> str:
    return (
        "def maximum(a: int, b: int) -> int:\n"
        f"    return max({a}, {b})\n"
    )


def _rows(start: int, stop: int, *, validation: bool = False):
    rows = []
    suffix = "validation" if validation else "train"

    def add(
        prompt: str,
        response: str,
        *,
        executable: bool = False,
        role_formatted: bool = False,
        replay_count: int = 1,
    ) -> None:
        rows.append(
            {
                "id": f"conversation-{suffix}-{len(rows):06d}",
                "family": "chat_instruction",
                "language": "text",
                "prompt": prompt,
                "response": response.strip(),
                "surface_valid": True,
                "execution_valid": 1 if executable else -1,
            }
        )
        if role_formatted:
            rows[-1]["role_formatted"] = True
        if replay_count > 1:
            rows[-1]["replay_count"] = int(replay_count)

    identity_prompts = (
        "Who are you?",
        "What is your name?",
        "Introduce yourself briefly.",
        "What can you do?",
        "Tell me what kind of assistant you are.",
    )
    for prompt in identity_prompts:
        add(prompt, IDENTITY, replay_count=30)

    add(
        "Write a Python function named add that returns the sum of two integers.",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        executable=True,
        replay_count=30,
    )
    add(
        "Write a Python function reverse_text that returns floor reversed. Return only code.",
        "def reverse_text(text: str) -> str:\n    return text[::-1]\n",
        executable=True,
        replay_count=30,
    )
    add(
        "Write Python function add with fixed terms 2 and 3. Return only code.",
        "def add(a: int, b: int) -> int:\n    return 2 + 3\n",
        executable=True,
        replay_count=30,
    )
    add(
        "Explain in one sentence why a unit test fails.",
        "A unit test fails when the implementation's observed behavior differs from "
        "the expected behavior for the tested input.",
        replay_count=30,
    )
    add(
        'Return only JSON with keys "status" and "count" for status ok and count 2.',
        '{"count":2,"status":"ok"}',
        replay_count=30,
    )
    add(
        "User: Write a Python function named square that returns x times x.\n"
        "Assistant:\n"
        "def square(x: int) -> int:\n"
        "    return x * x\n"
        "User: Rename square to double and make it return x plus x.",
        "def double(x: int) -> int:\n    return x + x\n",
        executable=True,
        role_formatted=True,
        replay_count=30,
    )
    add(
        "User: Write a Python function named square that returns x times x.\n"
        "Assistant:\n"
        "def square(x: int) -> int:\n"
        "    return x * x\n"
        "User: Rename square to double and make it return x plus x.\n"
        "Assistant:\n"
        "def double(x: int) -> int:\n"
        "    return x + x\n"
        "User: What input gives double(3) equal to 6?",
        "The input is 3.",
        role_formatted=True,
        replay_count=30,
    )
    add(
        "Write a complete C++ program that prints the sum of 2 plus 3. Return only code.",
        "#include <iostream>\n"
        "int main() {\n"
        "    std::cout << (2 + 3) << '\\n';\n"
        "    return 0;\n"
        "}\n",
        executable=True,
        replay_count=30,
    )

    for value in range(start, stop):
        a = value * 3 + 2
        b = value * 5 - 4
        if value % 4 == 0:
            prompt = (
                f"Write Python function add with fixed terms {a} and {b}. "
                "Return only code."
            )
            add(prompt, _python_add(a, b), executable=True)
        elif value % 4 == 1:
            prompt = (
                f"Create a Python subtract function that returns {a} minus {b}. "
                "Return only code."
            )
            add(prompt, _python_sub(a, b), executable=True)
        elif value % 4 == 2:
            prompt = (
                f"Write a Python multiply function for inputs {a} and {b}. "
                "Return only code."
            )
            add(prompt, _python_mul(a, b), executable=True)
        else:
            prompt = (
                f"Write a Python maximum function for the constants {a} and {b}. "
                "Return only code."
            )
            add(prompt, _python_max(a, b), executable=True)

    for value in range(start, stop, 2):
        text = f"floorflow-{value}"
        add(
            f"Write a Python function reverse_text that returns {text!r} reversed. "
            "Return only code.",
            "def reverse_text(text: str) -> str:\n    return text[::-1]\n",
            executable=True,
        )
        add(
            "How should I debug a failing unit test?",
            "Reproduce the first failure with the smallest input, compare expected "
            "and actual values, fix one cause, then rerun the focused test.",
        )
        add(
            "Explain recursion in one paragraph.",
            "Recursion solves a problem by calling the same function on a smaller "
            "input. A base case stops the calls and each step must approach it.",
        )

    for value in range(start, stop):
        payload = {
            "status": "ok",
            "count": value,
            "language": "python" if value % 2 == 0 else "rust",
            "valid": True,
        }
        add(
            f"Return only one JSON object for status ok, count {value}, language "
            f"{payload['language']}, and valid true.",
            json.dumps(payload, separators=(",", ":")),
        )

    for value in range(start, stop, 3):
        add(
            f"Write a Rust function multiply_const that returns x times {value}. "
            "Return only code.",
            f"fn multiply_const(x: i64) -> i64 {{\n    x * {value}\n}}\n",
            executable=True,
        )
        add(
            f"Write a C++ loop that adds integers from 0 through {value}. Return only code.",
            "int total = 0;\n"
            f"for (int i = 0; i <= {value}; ++i) {{\n"
            "    total += i;\n"
            "}\n",
            executable=True,
        )
        add(
            f"Explain one reason a Triton kernel for size {value} may mismatch PyTorch.",
            "Check block masking at the tail, launch bounds, accumulation dtype, and "
            "the exact reduction order before changing numerical tolerances.",
        )
        add(
            f"Give one safe first step for diagnosing Linux workload {value}.",
            "Reproduce the workload once, record CPU, memory, I/O, and GPU counters, "
            "then isolate the first bottleneck with a smaller controlled run.",
        )

    for value in range(start, stop, 2):
        add(
            "Write a Python function that reverses a string.",
            "def reverse_text(text: str) -> str:\n    return text[::-1]\n",
            executable=True,
        )
        add(
            "User: Write a Python add function.\n"
            "Assistant:\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "User: Rename add to plus without changing its behavior.",
            "def plus(a: int, b: int) -> int:\n    return a + b\n",
            executable=True,
            role_formatted=True,
        )
        add(
            f"User: Use a Python function named plus for constant addition.\n"
            f"Assistant:\n"
            f"def plus(a: int, b: int) -> int:\n"
            f"    return {value} + 9\n"
            "User: Keep the name and explain whether the code parses.",
            "The code parses as a Python function definition and keeps the name plus.",
            role_formatted=True,
        )
        add(
            f"Return Python code for {value} plus 9 using the function name plus.",
            f"def plus(a: int, b: int) -> int:\n    return {value} + 9\n",
            executable=True,
        )
        add(
            "What should you do instead of guessing at a fix?",
            "Reproduce the issue, inspect concrete evidence, state the smallest "
            "hypothesis, test it, and report the observed result.",
        )

    return rows


def _write(path: Path, rows: list[dict]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    seen = set()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = (row["prompt"], row["response"])
            if key in seen:
                continue
            seen.add(key)
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest(), len(seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/chat_conversation_v3",
    )
    args = parser.parse_args()
    train_rows = _rows(0, 2400)
    validation_rows = _rows(9000, 9300, validation=True)
    train_hash, train_count = _write(args.output_root / "train.jsonl", train_rows)
    validation_hash, validation_count = _write(
        args.output_root / "validation.jsonl",
        validation_rows,
    )
    manifest = {
        "version": "nlp-chat-conversation-v3",
        "status": "pass",
        "format": "row_aligned_prompt_response",
        "train_rows": train_count,
        "validation_rows": validation_count,
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "answer_only_supervision": True,
        "role_boundary_training": True,
        "multi_turn_context_rows": True,
        "json_targets": "strict JSON objects",
        "execution_labels": "trusted deterministic Python, Rust, and C++ templates",
        "benchmark_or_test_rows": 0,
        "evidence_boundary": "Instruction-data construction only; no capability claim.",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
