from __future__ import annotations

import subprocess
from pathlib import Path

from floorflow.nlp_flow.executors import execute_repository_patch, run_command


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "value.txt").write_text("1\n", encoding="utf-8")
    (repo / "test_value.py").write_text(
        "from pathlib import Path\nassert Path('value.txt').read_text().strip() == '2'\n",
        encoding="utf-8",
    )
    subprocess.run(("git", "init"), cwd=repo, check=True, capture_output=True)
    return repo


def test_repository_patch_executor_passes_golden_patch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    patch = tmp_path / "golden.diff"
    patch.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "index 5d2a2fa..4e26cbd 100644\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-1\n"
        "+2\n",
        encoding="utf-8",
    )

    result = execute_repository_patch(
        repo,
        patch,
        [["python", "test_value.py"]],
    )

    assert result.passed
    assert (repo / "value.txt").read_text() == "1\n"


def test_repository_patch_executor_fails_broken_patch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    patch = tmp_path / "broken.diff"
    patch.write_text(
        "diff --git a/value.txt b/value.txt\n"
        "--- a/value.txt\n"
        "+++ b/value.txt\n"
        "@@ -1 +1 @@\n"
        "-missing\n"
        "+2\n",
        encoding="utf-8",
    )

    result = execute_repository_patch(repo, patch, [["python", "test_value.py"]])

    assert not result.passed
    assert not result.patch_applied


def test_command_timeout_fails_closed(tmp_path: Path) -> None:
    result = run_command(
        ("python", "-c", "import time; time.sleep(1)"),
        cwd=tmp_path,
        timeout_seconds=0.05,
    )

    assert result.timed_out
    assert not result.passed
