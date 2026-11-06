#!/usr/bin/env python3
"""Check or synchronize Rust macros and generated Python model contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = "src/floorflow/nlp_flow/_generated_protocol.py"


def _run(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync",
        action="store_true",
        help="regenerate the Python contract before checking it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/nlp_rust_contract_audit_v1.json",
    )
    args = parser.parse_args()
    commands = [
        [
            "cargo",
            "fmt",
            "--manifest-path",
            "rust/Cargo.toml",
            "--all",
            "--",
            "--check",
        ],
        [
            "cargo",
            "test",
            "--manifest-path",
            "rust/Cargo.toml",
            "--workspace",
        ],
    ]
    if args.sync:
        commands.append(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                "rust/Cargo.toml",
                "-p",
                "nlp_flow_codegen",
                "--",
                "--format",
                "python",
                "--output",
                GENERATED,
            ]
        )
    commands.append(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            "rust/Cargo.toml",
            "-p",
            "nlp_flow_codegen",
            "--",
            "--format",
            "python",
            "--check",
            GENERATED,
        ]
    )
    stages = [_run(command) for command in commands]
    report = {
        "version": "nlp-rust-contract-audit-v1",
        "status": (
            "pass"
            if all(stage["status"] == "pass" for stage in stages)
            else "fail"
        ),
        "synced": args.sync,
        "stages": stages,
        "evidence_boundary": (
            "Compile-time contract and generated-file consistency only; "
            "no model-quality evidence."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
