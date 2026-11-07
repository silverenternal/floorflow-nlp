from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matched_validation_builds_complete_cpu_ledger_without_execution(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[5]
    report = tmp_path / "ledger.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "project/scripts/experiments/validation/run_nlp_matched_validation.py"),
            "--config",
            str(root / "project/configs/other/general/capability_open_matched/nlp_two_stage_matched.json"),
            "--report",
            str(report),
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(report.read_text())
    assert payload["status"] == "planned_not_executed"
    assert payload["execution_performed"] is False
    assert payload["gpu_used"] is False
    assert len(payload["ledger"]) == 12
    assert all(not row["measured"] and row["config_sha256"] for row in payload["ledger"])
