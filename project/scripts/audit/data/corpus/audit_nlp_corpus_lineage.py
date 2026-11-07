#!/usr/bin/env python3
"""Audit corpus lineage metadata without evaluating test examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.corpus.corpus_lineage import audit_lineage

ROOT = Path(__file__).resolve().parents[5]
PARENT_FIELDS = frozenset(
    {
        "case_id",
        "language",
        "split",
        "source_id",
        "source_path",
        "source_sha256",
        "repository",
        "revision",
        "license",
        "clone_family",
    }
)
FRAGMENT_FIELDS = frozenset({"case_id", "language", "split", "parent_case_id", "target_source_sha256"})


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path, fields: frozenset[str]) -> list[dict]:
    """Read only the allowlisted metadata into the audit, never source/target text."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError("corpus row must be an object")
            rows.append({key: item[key] for key in fields if key in item})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl")
    parser.add_argument("--fragments", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument("--registry", type=Path, default=ROOT / "project/configs/data/sources/nlp_source_registry.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_corpus_coverage_v11.json")
    args = parser.parse_args()
    report = audit_lineage(
        load_metadata(args.parents, PARENT_FIELDS),
        load_metadata(args.fragments, FRAGMENT_FIELDS),
        json.loads(args.registry.read_text(encoding="utf-8"))["sources"],
    )
    report.update(
        {
            "version": "nlp-corpus-lineage-v11",
            "status": "partial_metadata_only",
            "input_sha256": {
                "parents": file_hash(args.parents),
                "fragments": file_hash(args.fragments),
                "registry": file_hash(args.registry),
            },
            "test_content_evaluated": False,
            "near_duplicate_scan": "not_performed",
            "benchmark_contamination": "not_performed_or_fingerprinted",
            "multilingual_claim": "blocked_pending_supplement_and_audits",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "parent_rows",
                    "fragment_rows",
                    "parent_coverage",
                    "fragment_coverage",
                    "fragment_parent_id_inferred",
                    "fragment_parent_missing",
                    "parent_clone_family_missing",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
