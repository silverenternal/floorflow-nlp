from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_structure_ablation_preflight_is_matched_and_keeps_test_closed(tmp_path) -> None:
    output = tmp_path / "preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_nlp_structure_ablation.py"),
            "--preflight",
            "--report",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [arm["id"] for arm in report["arms"]] == [
        "flat",
        "local_mask",
        "lexical_mask",
        "hierarchical",
    ]
    assert report["status"] == "ready"
    assert not report["gpu_used"]
    assert not report["test_content_read"]
    assert not report["winner_selection"]
    assert report["inputs"]["train_actions"]["rows"] == 403
    assert report["inputs"]["train_actions"]["split_scope"] == "train only"
    assert report["inputs"]["ast_corpus"]["test_rows_parsed"] == 0
    assert report["inputs"]["ast_corpus"]["validation_audit"]["status"] == "pass"
    for arm in report["arms"]:
        assert arm["registered_seeds"] == [17, 29, 43]
        assert arm["optimizer_steps_total"] == 612
