#!/usr/bin/env python3
"""Diagnose the gap between corpus continuation and bounded conversation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from tokenizers import Tokenizer

from floorflow.nlp_flow.chat import (
    generate_reply,
    model_config_from_experiment,
    render_conversation,
)
from floorflow.nlp_flow.engineering import load_nlp_checkpoint, restore_nlp_checkpoint
from floorflow.nlp_flow.model import CompactNLPDecoder

ROOT = Path(__file__).resolve().parents[1]


def _python_parse(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def _json_parse(text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/nlp_open_capability_v1/best.pt",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-role", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_conversation_gap_diagnosis_v1.json",
    )
    args = parser.parse_args()
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else ROOT / args.checkpoint
    )
    checkpoint_role = args.checkpoint_role or (
        "instruction_tuned"
        if "sft" in checkpoint_path.parent.name
        else "raw_corpus_pretraining"
    )
    payload = load_nlp_checkpoint(checkpoint_path, map_location=args.device)
    model = CompactNLPDecoder(model_config_from_experiment(payload["config"]))
    restore_nlp_checkpoint(payload, model=model, device=args.device)
    model = model.to(args.device).eval()
    tokenizer = Tokenizer.from_file(
        str(ROOT / payload["config"]["model"]["tokenizer"])
    )

    probes = [
        ("identity", [], "Who are you?"),
        (
            "python_add",
            [],
            "Write a Python function named add that returns the sum of two integers.",
        ),
        (
            "explanation",
            [],
            "Explain in one sentence why a unit test fails.",
        ),
        (
            "json",
            [],
            'Return only JSON with keys "status" and "count" for status ok and count 2.',
        ),
        (
            "two_turn",
            [
                (
                    "Write a Python function named square that returns x times x.",
                    "def square(x: int) -> int:\n    return x * x",
                )
            ],
            "Rename square to double and make it return x plus x.",
        ),
    ]
    cases = []
    for name, turns, prompt in probes:
        result = generate_reply(
            model,
            tokenizer,
            render_conversation(turns, prompt),
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            device=args.device,
        )
        text = result.text
        code_text = text
        cases.append(
            {
                "id": name,
                "output": text,
                "generated_tokens": result.generated_tokens,
                "stop_reason": result.stop_reason,
                "nonempty": bool(text.strip()),
                "role_marker_leak": "User:" in text or "Assistant:" in text,
                "max_token_truncation": result.stop_reason == "max_tokens",
                "python_parse_valid": _python_parse(code_text),
                "json_parse_valid": _json_parse(text),
            }
        )

    checks = {
        "all_nonempty": all(case["nonempty"] for case in cases),
        "no_role_marker_leakage": all(
            not case["role_marker_leak"] for case in cases
        ),
        "all_stop_before_token_limit": all(
            not case["max_token_truncation"] for case in cases
        ),
        "python_add_parse_valid": next(
            case for case in cases if case["id"] == "python_add"
        )["python_parse_valid"],
        "json_parse_valid": next(
            case for case in cases if case["id"] == "json"
        )["json_parse_valid"],
    }
    report = {
        "version": "nlp-conversation-gap-diagnosis-v1",
        "status": "pass" if all(checks.values()) else "gap_detected",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_role": checkpoint_role,
        "checks": checks,
        "cases": cases,
        "gap_tasks": [
            "replace corpus continuation with role-conditioned answer generation",
            "learn answer completion and final-turn stopping",
            "preserve code fences and generate syntactically complete functions",
            "make JSON valid by native constrained/action generation",
            "carry requested names and edits across turns",
            "validate executable code and structured outputs without repair or retry",
        ],
        "claim_level": "none",
        "evidence_boundary": (
            "Fixed-prompt diagnostic only; no aggregate conversational or coding claim."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
