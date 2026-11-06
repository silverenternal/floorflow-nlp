#!/usr/bin/env python3
"""Verify one coding and one engineering executor from the P2M smoke stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from floorflow.nlp_flow.executors import execute_repository_patch
from floorflow.nlp_flow.triton_executor import score_linux_diagnostic

ROOT = Path(__file__).resolve().parents[1]


def _coding_case(root: Path) -> bool:
    repository = root / "repo"
    repository.mkdir()
    (repository / "value.txt").write_text("1\n", encoding="utf-8")
    (repository / "test_value.py").write_text(
        "from pathlib import Path\n"
        "assert Path('value.txt').read_text().strip() == '2'\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "init"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    patch = root / "change.diff"
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
    return execute_repository_patch(
        repository,
        patch,
        [["python", "test_value.py"]],
    ).passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_p2m_executor_integration_v1.json",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nlp-p2m-executor-") as temporary:
        coding_passed = _coding_case(Path(temporary))
    diagnostic = score_linux_diagnostic(
        {
            "diagnosis": "disk_exhaustion",
            "recommended_action": "rotate_logs_then_retry",
            "evidence": ["No space left on device", "df shows 100% use"],
        },
        expected_diagnosis="disk_exhaustion",
        expected_action="rotate_logs_then_retry",
    )
    report = {
        "version": "nlp-p2m-executor-integration-v1",
        "status": "pass" if coding_passed and diagnostic.passed else "fail",
        "coding_patch_executor_passed": coding_passed,
        "engineering_linux_diagnostic_passed": diagnostic.passed,
        "gpu_required": False,
        "hidden_retry": False,
        "evidence_boundary": (
            "Executor wiring smoke only; fixtures are not model-quality measurements."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
