#!/usr/bin/env python3
"""Check frozen external benchmark runner prerequisites without opening tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runners",
        type=Path,
        default=ROOT / "configs/nlp_benchmark_runners_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_benchmark_runner_preflight_v1.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.runners.read_text(encoding="utf-8"))
    executables = {
        name: shutil.which(name)
        for name in ("python", "podman")
    }
    evalplus_available = importlib.util.find_spec("evalplus") is not None
    runner_checks = {}
    for runner in payload["runners"]:
        command = runner["command"]
        kind = runner["runner_kind"]
        ready = (
            evalplus_available
            if kind == "evalplus"
            else executables["podman"] is not None
        )
        runner_checks[runner["benchmark_id"]] = {
            "kind": kind,
            "ready": ready,
            "command": command,
        }
    report = {
        "version": "nlp-benchmark-runner-preflight-v1",
        "status": (
            "ready"
            if all(row["ready"] for row in runner_checks.values())
            else "blocked"
        ),
        "executables": executables,
        "evalplus_available": evalplus_available,
        "runners": runner_checks,
        "network_execution_allowed": False,
        "benchmarks_opened": False,
        "evidence_boundary": "Runner installation preflight only; no benchmark cases were read or executed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
