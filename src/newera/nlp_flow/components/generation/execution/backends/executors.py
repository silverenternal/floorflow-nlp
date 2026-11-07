"""Deterministic isolated repository-patch execution."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class RepositoryCaseResult:
    patch_applied: bool
    commands: tuple[CommandResult, ...]

    @property
    def passed(self) -> bool:
        return self.patch_applied and all(command.passed for command in self.commands)


def run_command(
    command: tuple[str, ...] | list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> CommandResult:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("command must be a non-empty string list")
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONHASHSEED": "0"},
        )
        return CommandResult(
            tuple(command),
            completed.returncode,
            completed.stdout[-8000:],
            completed.stderr[-8000:],
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            tuple(command),
            -1,
            (error.stdout or "")[-8000:] if isinstance(error.stdout, str) else "",
            (error.stderr or "")[-8000:] if isinstance(error.stderr, str) else "",
            timed_out=True,
        )


def execute_repository_patch(
    repository: Path,
    patch_file: Path,
    commands: list[list[str]],
    *,
    command_timeout_seconds: float = 30.0,
) -> RepositoryCaseResult:
    """Apply and test a patch in a fresh copy without mutating the source repo."""

    if not repository.is_dir():
        raise FileNotFoundError(repository)
    if not patch_file.is_file():
        raise FileNotFoundError(patch_file)
    with tempfile.TemporaryDirectory(prefix="nlp-repo-case-") as temporary:
        workspace = Path(temporary) / repository.name
        shutil.copytree(repository, workspace)
        applied = run_command(
            ("git", "apply", "--whitespace=error-all", str(patch_file.resolve())),
            cwd=workspace,
            timeout_seconds=command_timeout_seconds,
        )
        if not applied.passed:
            return RepositoryCaseResult(False, (applied,))
        results = tuple(
            run_command(
                command,
                cwd=workspace,
                timeout_seconds=command_timeout_seconds,
            )
            for command in commands
        )
        return RepositoryCaseResult(True, (applied, *results))


__all__ = [
    "CommandResult",
    "RepositoryCaseResult",
    "execute_repository_patch",
    "run_command",
]
