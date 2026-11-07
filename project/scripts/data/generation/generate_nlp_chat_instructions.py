#!/usr/bin/env python3
"""Generate deterministic chat and coding instruction data from frozen train shards."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
IDENTIFIER = re.compile(r"\b(?:def|class|fn|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")

IDENTITY_RESPONSES = (
    (
        "I am a compact coding assistant trained from scratch by the NewEra project. "
        "I can write short programs, explain code, and help with simple debugging."
    ),
    (
        "I am a small language and coding model. I am designed to answer directly, keep "
        "answers short, and prefer executable code when asked."
    ),
    (
        "I am the compact NLP Flow model. I can help with Python, Rust, C++, and short "
        "engineering questions."
    ),
)


def _chat(user: str, assistant: str) -> str:
    return f"User: {user.strip()}\nAssistant: {assistant.strip()}\n"


def _synthetic_examples(offset: int, count: int) -> list[str]:
    examples = []
    for index in range(count):
        value = offset + index + 2
        kind = index % 10
        if kind == 0:
            examples.append(
                _chat(
                    f"Write a Python function that adds {value} to a number.",
                    f"def add_{value}(x):\n    return x + {value}\n",
                )
            )
        elif kind == 1:
            examples.append(
                _chat(
                    f"Write a Rust function that multiplies an integer by {value}.",
                    f"fn multiply_{value}(x: i64) -> i64 {{\n    x * {value}\n}}\n",
                )
            )
        elif kind == 2:
            examples.append(
                _chat(
                    f"Implement a C++ function that returns the maximum of x and {value}.",
                    f"int maximum_{value}(int x) {{\n    return x > {value} ? x : {value};\n}};\n",
                )
            )
        elif kind == 3:
            examples.append(
                _chat(
                    "What is a variable?",
                    "A variable is a named place that stores a value so a program can use "
                    "or change it later.",
                )
            )
        elif kind == 4:
            examples.append(
                _chat(
                    "How should I debug a failing unit test?",
                    "Read the first failure, reproduce it with the smallest input, inspect "
                    "the expected and actual values, then fix one cause and rerun the test.",
                )
            )
        elif kind == 5:
            examples.append(
                _chat(
                    "Write a Python function that reverses a string.",
                    "def reverse_text(text: str) -> str:\n    return text[::-1]\n",
                )
            )
        elif kind == 6:
            examples.append(
                _chat(
                    "Write a Rust function that computes a square.",
                    "fn square(x: i64) -> i64 {\n    x * x\n}\n",
                )
            )
        elif kind == 7:
            examples.append(
                _chat(
                    "Write a C++ loop that sums integers from 0 up to n.",
                    "int total = 0;\nfor (int i = 0; i <= n; ++i) {\n"
                    "    total += i;\n}\n",
                )
            )
        elif kind == 8:
            examples.append(
                _chat(
                    "Explain recursion in one paragraph.",
                    "Recursion solves a problem by having a function call itself on a smaller "
                    "input. A base case stops the calls, and each step must move closer to it.",
                )
            )
        else:
            examples.append(
                _chat(
                    "Give me a short coding checklist.",
                    "Clarify the contract, write the smallest correct implementation, handle "
                    "edge cases, run focused tests, and review complexity.",
                )
            )
    return examples


def _code_examples(limit: int) -> list[str]:
    examples = []
    paths = (
        ROOT / ".data/nlp/mixed_v1/corpus_v1/train.jsonl",
        ROOT / ".data/nlp/p5m_v2/delta_train.jsonl",
    )
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("family") != "code":
                    continue
                code = str(row["text"]).strip()
                digest = hashlib.sha256(code.encode()).hexdigest()
                if digest in seen or not 200 <= len(code) <= 3000:
                    continue
                seen.add(digest)
                match = IDENTIFIER.search(code)
                symbol = match.group(1) if match else "the requested behavior"
                language = str(row.get("language", "code"))
                prompt_kind = len(examples) % 3
                if prompt_kind == 0:
                    user = (
                        f"Explain the purpose of this {language} code, which defines "
                        f"{symbol}."
                    )
                    answer = (
                        f"This snippet defines or implements {symbol} in {language}. "
                        "It should be read by following inputs, transformations, and the "
                        f"returned result.\nCode:\n{code}"
                    )
                elif prompt_kind == 1:
                    user = f"Continue the following {language} code:\n{code[:900]}"
                    answer = code[900:] or code
                else:
                    user = (
                        f"Show a short {language} example related to {symbol} and keep "
                        "the answer executable."
                    )
                    answer = code
                examples.append(_chat(user, answer))
                if len(examples) >= limit:
                    return examples
    return examples


def _story_examples(limit: int) -> list[str]:
    examples = []
    path = ROOT / ".data/nlp/mixed_v1/corpus_v1/train.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("family") != "general_language":
                continue
            story = str(row["text"]).strip()
            if not story:
                continue
            examples.append(_chat("Tell me a short story.", story))
            if len(examples) >= limit:
                break
    return examples


def _python_executes(code: str) -> bool:
    if not code.startswith("def "):
        return False
    namespace = {}
    try:
        compiled = compile(code, "<synthetic-chat-code>", "exec")
        exec(  # noqa: S102 - execution is limited to generated trusted templates
            compiled,
            {"__builtins__": {"max": max, "min": min}},
            namespace,
        )
        function = next(
            (
                value
                for value in namespace.values()
                if inspect.isfunction(value)
            ),
            None,
        )
        if function is None:
            return False
        arguments = [3] * len(inspect.signature(function).parameters)
        function(*arguments)
        return True
    except (SyntaxError, TypeError, ValueError, NameError, ZeroDivisionError, OverflowError):
        return False


def _write(path: Path, rows: list[tuple[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for index, (kind, text) in enumerate(rows):
            assistant = text.rsplit("Assistant:", 1)[-1].strip()
            synthetic_executable = kind in {"synthetic", "validation"} and (
                _python_executes(assistant)
            )
            payload = {
                "id": f"chat-instruction-{kind}-{index:06d}",
                "family": "chat_instruction",
                "language": "text",
                "kind": kind,
                "text": text,
                "surface_valid": True,
                "execution_valid": 1 if synthetic_executable else -1,
            }
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/chat_instructions_v1",
    )
    parser.add_argument("--code-examples", type=int, default=2_000)
    parser.add_argument("--synthetic-examples", type=int, default=10_000)
    parser.add_argument("--story-examples", type=int, default=2_000)
    parser.add_argument("--identity-repetitions", type=int, default=30)
    args = parser.parse_args()
    identity = [
        ("identity", _chat("Who are you?", response))
        for response in IDENTITY_RESPONSES
        for _repeat in range(args.identity_repetitions)
    ]
    synthetic = [
        ("synthetic", text)
        for text in _synthetic_examples(0, args.synthetic_examples)
    ]
    stories = [
        ("story", text)
        for text in _story_examples(args.story_examples)
    ]
    code = [
        ("code", text)
        for text in _code_examples(args.code_examples)
    ]
    validation_synthetic = [
        ("validation", text)
        for text in _synthetic_examples(10_000, 100)
    ]
    train_rows = identity + synthetic + stories + code
    validation_rows = validation_synthetic
    train_hash = _write(args.output_root / "train.jsonl", train_rows)
    validation_hash = _write(args.output_root / "validation.jsonl", validation_rows)
    report = {
        "version": "nlp-chat-instructions-v1",
        "status": "pass",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "source_policy": "generated from frozen train corpus and synthetic templates only",
        "benchmark_rows_used": 0,
        "evidence_boundary": "Instruction data preparation only; no capability claim.",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
