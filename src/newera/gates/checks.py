"""Checks that can be evaluated without training data or accelerator access."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".DS_Store", ".data", "outputs", "checkpoints", "build", "dist"}
EXPLORATORY = ("src/newera/nlp_flow/components", "src/newera/nlp_flow/model")
VERSION_ITERATION = re.compile(r"(?i)(?:^|[._-])v\d+(?=$|[._-])")


def _ignored(name: str) -> bool:
    return name in IGNORED or name.endswith(".egg-info")


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def check_directories(root: Path) -> list[dict[str, str]]:
    findings = []
    directories = [root]
    for base in (root / "src", root / "project", root / ".github"):
        if base.is_dir():
            for current, subdirs, _ in os.walk(base, followlinks=False):
                subdirs[:] = [name for name in subdirs if not _ignored(name)]
                directories.append(Path(current))
    for directory in directories:
        entries = [
            p
            for p in directory.iterdir()
            if not _ignored(p.name)
            and not (directory == root and p.name == "uv.lock")
        ]
        if len(entries) <= 5:
            continue
        relative = directory.relative_to(root).as_posix()
        level = "warning" if any(_under(relative, prefix) for prefix in EXPLORATORY) else "error"
        findings.append({"level": level, "check": "directory_limit", "message": f"{relative}: {len(entries)} entries (limit 5)"})
    return findings


def check_versioned_paths(root: Path) -> list[dict[str, str]]:
    """Reject file and directory names used as parallel version snapshots."""
    findings = []
    for base in (root,):
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if any(_ignored(part) for part in relative.parts):
                continue
            if any(VERSION_ITERATION.search(part) for part in relative.parts):
                findings.append({
                    "level": "error",
                    "check": "git_is_the_version_history",
                    "message": f"versioned path name is prohibited; use Git history and a stable functional name: {relative.as_posix()}",
                })
    return findings


def check_single_git_worktree(root: Path) -> list[dict[str, str]]:
    """Keep one local checkout; Git history lives in commits/branches, not copies."""
    if not (root / ".git").exists():
        return []
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "worktree", "list", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return [{
            "level": "error",
            "check": "single_git_worktree",
            "message": f"cannot inspect Git worktrees: {exc}",
        }]
    worktrees = [line for line in result.stdout.splitlines() if line.startswith("worktree ")]
    if len(worktrees) <= 1:
        return []
    return [{
        "level": "error",
        "check": "single_git_worktree",
        "message": f"expected one local project worktree, found {len(worktrees)}",
    }]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def check_sources(root: Path) -> list[dict[str, str]]:
    findings = []
    for folder in (root / "src", root / "project"):
        if not folder.is_dir():
            findings.append({"level": "error", "check": "layout", "message": f"missing required directory: {folder}"})
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or any(_ignored(part) for part in path.relative_to(root).parts):
                continue
            if path.suffix not in (".py", ".json"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
                if path.suffix == ".py":
                    ast.parse(text, filename=str(path))
                else:
                    json.loads(text, object_pairs_hook=_unique_object)
            except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
                findings.append({"level": "error", "check": "syntax", "message": f"{path.relative_to(root)}: {exc}"})
    return findings
