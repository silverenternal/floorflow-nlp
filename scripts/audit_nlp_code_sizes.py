#!/usr/bin/env python3
"""Enforce the repository's 599-line maximum for first-party code files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("src", "scripts", "tests", "rust")
MAX_LINES = 599


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def code_file_sizes(root: Path = ROOT) -> list[dict[str, int | str]]:
    rows = []
    for directory in SOURCE_ROOTS:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or path.suffix not in {".py", ".rs"}
                or "__pycache__" in path.parts
            ):
                continue
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "lines": line_count(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_code_size_audit_v1.json",
    )
    args = parser.parse_args()
    rows = code_file_sizes()
    violations = [row for row in rows if int(row["lines"]) > MAX_LINES]
    report = {
        "version": "nlp-code-size-audit-v1",
        "status": "pass" if not violations else "fail",
        "maximum_lines": MAX_LINES,
        "files_scanned": len(rows),
        "violations": violations,
        "largest_files": sorted(
            rows,
            key=lambda row: int(row["lines"]),
            reverse=True,
        )[:10],
        "evidence_boundary": "Engineering hygiene only; no model evidence.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
