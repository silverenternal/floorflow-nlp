from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "project/scripts/experiments/validation"))

from run_nlp_strict_ast_validation import _trace_summary, load_validation_rows


def test_strict_validation_loader_skips_test_payload_before_decode(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"split":"validation","case_id":"val-1"}\n'
        '{"split":"test","payload": not-json}\n',
        encoding="utf-8",
    )

    rows = load_validation_rows(corpus, ["val-1"])

    assert rows == [{"split": "validation", "case_id": "val-1"}]


def test_live_trace_summary_keeps_decisions_but_drops_raw_logits() -> None:
    summary = _trace_summary(
        [
            {
                "step": 4,
                "action": {"type": "OpenNodeAction"},
                "choices": [
                    {"head": "action", "selected_id": 1, "allowed_ids": [1, 2], "logits": [0.1, 0.9]}
                ],
            },
            {"event": "generation_end", "error": "fixture failure"},
        ]
    )

    assert summary["decision_count"] == 1
    assert summary["terminal_error"] == "fixture failure"
    assert summary["decisions"][0]["heads"] == [
        {"head": "action", "selected_id": 1, "allowed_ids": [1, 2]}
    ]
