#!/usr/bin/env python3
"""No-repair acceptance checks for executable and multi-turn conversation."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

from newera.nlp_flow.components.generation.outputs.dialogue.chat import (
    generate_reply,
    model_config_from_experiment,
    render_conversation,
)
from newera.nlp_flow.engineering import load_nlp_checkpoint, restore_nlp_checkpoint
from newera.nlp_flow.model import CompactNLPDecoder

ROOT = Path(__file__).resolve().parents[4]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _python_add_check(text: str) -> tuple[bool, str]:
    namespace = {}
    try:
        exec(  # noqa: S102 - generated code is evaluated only through this frozen probe
            text,
            {"__builtins__": {"max": max, "min": min}},
            namespace,
        )
        function = namespace.get("add")
        passed = callable(function) and function(7, 11) == 18
    except Exception as error:  # noqa: BLE001 - all execution failures are measurements
        return False, f"{type(error).__name__}: {error}"
    return passed, "add(7, 11) == 18"


def _python_reverse_check(text: str) -> tuple[bool, str]:
    namespace = {}
    try:
        exec(  # noqa: S102 - generated code is evaluated only through this frozen probe
            text,
            {"__builtins__": {}},
            namespace,
        )
        function = namespace.get("reverse_text")
        passed = callable(function) and function("floor") == "roolf"
    except Exception as error:  # noqa: BLE001 - all execution failures are measurements
        return False, f"{type(error).__name__}: {error}"
    return passed, 'reverse_text("floor") == "roolf"'


def _rust_compile_check(text: str) -> tuple[bool, str]:
    source = text
    if "fn main()" not in source:
        source += (
            "\nfn main() {\n"
            "    assert_eq!(multiply_const(9), 18);\n"
            "}\n"
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "main.rs"
        binary = Path(directory) / "main"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["rustc", str(path), "-o", str(binary)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()[:500]
        run = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    return run.returncode == 0, run.stderr.strip()[:500]


def _cpp_compile_check(text: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "main.cpp"
        binary = Path(directory) / "main"
        source.write_text(text, encoding="utf-8")
        compile_result = subprocess.run(
            ["g++", "-std=c++17", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if compile_result.returncode != 0:
            return False, compile_result.stderr.strip()[:500]
        run = subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    return (
        run.returncode == 0 and run.stdout.strip() == "5",
        run.stderr.strip()[:500],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/nlp_open_conversation_sft_v3/best.pt",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/nlp_conversation_executable_acceptance_v1.json",
    )
    args = parser.parse_args()
    checkpoint_path = _resolve(args.checkpoint)
    payload = load_nlp_checkpoint(checkpoint_path, map_location=args.device)
    model = CompactNLPDecoder(model_config_from_experiment(payload["config"]))
    restore_nlp_checkpoint(payload, model=model, device=args.device)
    model = model.to(args.device).eval()
    tokenizer = Tokenizer.from_file(
        str(ROOT / payload["config"]["model"]["tokenizer"])
    )

    def generate(turns, prompt: str) -> dict:
        result = generate_reply(
            model,
            tokenizer,
            render_conversation(turns, prompt),
            max_new_tokens=96,
            temperature=0.0,
            repetition_penalty=1.0,
            device=args.device,
        )
        return {
            "text": result.text,
            "stop_reason": result.stop_reason,
            "generated_tokens": result.generated_tokens,
        }

    add = generate(
        (),
        "Write a Python function named add that returns the sum of two integers.",
    )
    add["execution_pass"] = _python_add_check(add["text"])
    reverse = generate(
        (),
        "Write a Python function reverse_text that returns floor reversed. Return only code.",
    )
    reverse["execution_pass"] = _python_reverse_check(reverse["text"])
    structured = generate(
        (),
        'Return only JSON with keys "status" and "count" for status ok and count 2.',
    )
    try:
        structured_json = json.loads(structured["text"])
    except json.JSONDecodeError:
        structured_json = None
    structured["schema_pass"] = structured_json == {
        "count": 2,
        "status": "ok",
    }
    rust = generate(
        (),
        "Write a Rust function multiply_const that returns x times 2. Return only code.",
    )
    rust["execution_pass"] = _rust_compile_check(rust["text"])
    cpp = generate(
        (),
        "Write a complete C++ program that prints the sum of 2 plus 3. Return only code.",
    )
    cpp["execution_pass"] = _cpp_compile_check(cpp["text"])

    history = [
        (
            "Write a Python function named square that returns x times x.",
            "def square(x: int) -> int:\n    return x * x",
        ),
        (
            "Rename square to double and make it return x plus x.",
            "def double(x: int) -> int:\n    return x + x",
        ),
    ]
    multi_turn = generate(history, "What input gives double(3) equal to 6?")
    multi_turn["context_pass"] = "3" in multi_turn["text"]

    cases = {
        "python_add": add,
        "python_reverse": reverse,
        "json_schema": structured,
        "rust_compile": rust,
        "cpp_compile": cpp,
        "multi_turn_context": multi_turn,
    }
    checks = {
        "python_add_executes": add["execution_pass"][0],
        "python_reverse_executes": reverse["execution_pass"][0],
        "json_schema_valid": structured["schema_pass"],
        "rust_compiles_and_runs": rust["execution_pass"][0],
        "cpp_compiles_and_runs": cpp["execution_pass"][0],
        "multi_turn_context_preserved": multi_turn["context_pass"],
        "no_max_token_truncation": all(
            case["stop_reason"] != "max_tokens"
            for case in cases.values()
        ),
    }
    report = {
        "version": "nlp-executable-conversation-acceptance-v1",
        "status": "pass" if all(checks.values()) else "gap_detected",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checks": checks,
        "cases": cases,
        "repair_count": 0,
        "retry_count": 0,
        "best_of_n": False,
        "claim_level": "fixed_acceptance_only",
        "evidence_boundary": (
            "Small fixed acceptance suite from training-adjacent templates; "
            "does not establish general coding or benchmark performance."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
