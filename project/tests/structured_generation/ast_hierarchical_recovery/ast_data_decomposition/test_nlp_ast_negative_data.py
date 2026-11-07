from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.corpus.ast_corpus import build_ast_corpus
from newera.nlp_flow.components.platform.data.splits.ast_negative_data import build_ast_negative_suite


def _training_corpus(tmp_path: Path) -> list[dict[str, object]]:
    source_root = tmp_path / ".data/code"
    source_root.mkdir(parents=True)
    (source_root / "sample.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    registry = {
        "sources": [
            {
                "id": "fixture",
                "repository": "fixture/repo",
                "revision": "abc",
                "source_url": "https://example.invalid",
                "license": "MIT",
                "license_url": "https://example.invalid/license",
                "usage": "training",
            }
        ]
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    config = {
        "root": ".data/code",
        "source_registry": "registry.json",
        "selection": {
            "rows_per_language": 1,
            "min_source_bytes": 1,
            "max_source_bytes": 1000,
            "min_ast_nodes": 1,
            "max_ast_nodes": 100,
            "max_depth": 100,
            "max_files_scanned_per_language": 10,
        },
        "sources": {
            "python": [
                {
                    "id": "fixture",
                    "directory": ".",
                    "extensions": [".py"],
                }
            ]
        },
        "exclude_path_parts": [],
        "split": {
            "train_modulus": 101,
            "validation_modulus": 101,
        },
    }
    rows, _summary = build_ast_corpus(config, root=tmp_path)
    return [row.as_dict() for row in rows]


def test_negative_suite_contains_required_kinds_from_train_only(
    tmp_path: Path,
) -> None:
    rows = _training_corpus(tmp_path)

    cases, report = build_ast_negative_suite(
        rows,
        grammar_cases_per_language=0,
    )

    assert report["status"] == "pass"
    assert report["all_training_only"]
    assert report["validation_test_provenance"] == []
    assert {case["split"] for case in cases} == {"train"}
    assert {case["kind"] for case in cases} >= {
        "illegal_root",
        "illegal_child",
        "illegal_edge",
        "illegal_scope",
        "parent_cycle",
        "premature_eos",
        "valid_but_semantically_wrong",
    }


def test_negative_suite_is_deterministic(tmp_path: Path) -> None:
    rows = _training_corpus(tmp_path)

    first, first_report = build_ast_negative_suite(
        rows,
        grammar_cases_per_language=0,
    )
    second, second_report = build_ast_negative_suite(
        rows,
        grammar_cases_per_language=0,
    )

    assert first == second
    assert first_report == second_report
