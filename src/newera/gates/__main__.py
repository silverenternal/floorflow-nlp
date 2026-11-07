"""Run policy checks for local development and CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .checks import (
    check_directories,
    check_single_git_worktree,
    check_sources,
    check_versioned_paths,
)
from .evidence import check_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NewEra policy checks")
    parser.add_argument("--stage", choices=("explore", "research", "release"), default="explore")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--evidence", type=Path, help="JSON evidence manifest for research or release")
    parser.add_argument("--execute", nargs=argparse.REMAINDER, help="command and arguments to run after checks (no shell)")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(json.dumps({"stage": args.stage, "status": "failed", "qualified": False, "findings": [{"level": "error", "check": "layout", "message": f"root is not a directory: {root}"}]}), file=sys.stderr, flush=True)
        return 1
    findings = (
        check_directories(root)
        + check_versioned_paths(root)
        + check_single_git_worktree(root)
        + check_sources(root)
        + check_evidence(root, args.stage, args.evidence)
    )
    errors = [item for item in findings if item["level"] == "error"]
    warnings = [item for item in findings if item["level"] == "warning"]
    report = {"stage": args.stage, "status": "failed" if errors else "warnings" if warnings else "passed", "qualified": False if args.stage != "explore" else None, "findings": findings}
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr if args.execute is not None else sys.stdout, flush=True)
    if errors:
        return 1
    if args.execute is not None:
        if not args.execute:
            print("--execute requires a command", file=sys.stderr)
            return 2
        try:
            return subprocess.run(args.execute, cwd=root, check=False).returncode
        except OSError as exc:
            print(f"cannot execute command: {exc}", file=sys.stderr)
            return 127
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
