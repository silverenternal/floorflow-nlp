from __future__ import annotations

import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.platform.data.corpus.ast_corpus import (
    _constructs,
    build_ast_corpus,
    write_ast_corpus,
)


def test_ast_corpus_builder_rejects_parse_errors_and_keeps_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / ".data/code"
    source_root.mkdir(parents=True)
    (source_root / "good.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (source_root / "bad.py").write_text("def broken(\n", encoding="utf-8")
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
            "rows_per_language": 2,
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
            "train_modulus": 80,
            "validation_modulus": 90,
        },
    }

    rows, summary = build_ast_corpus(config, root=tmp_path)
    report = write_ast_corpus(
        rows,
        output=tmp_path / "corpus.jsonl",
        manifest=tmp_path / "manifest.json",
        summary=summary,
    )

    assert len(rows) == 1
    assert rows[0].license == "MIT"
    assert rows[0].revision == "abc"
    assert summary["rejected"]["parse_or_utf8"] == 1
    assert report["source_integrity"]


def test_ast_corpus_tracks_every_derive_target_for_holdout() -> None:
    source = (
        "#[derive(Clone, Debug, serde::Serialize)]\n"
        "#[repr(C)]\n"
        "pub struct Sample;\n"
    )
    graph = parse_ast_graph(source, "rust")

    constructs = set(_constructs(graph, source))

    assert {
        "attribute_name:Clone",
        "attribute_name:Debug",
        "attribute_name:Serialize",
        "attribute_name:repr",
    }.issubset(constructs)
