#!/usr/bin/env python3
"""Build normalized source-disjoint mixed corpus shards before tokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from newera.nlp_flow.components.platform.data.corpus.corpus_build import normalize, sanitize

ROOT = Path(__file__).resolve().parents[4]


def read_seed(split: str) -> list[dict]:
    path = ROOT / f".data/nlp/tinystories_small_v1/{split}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(
                {
                    "id": str(payload["id"]),
                    "text": payload["text"],
                    "source_id": "tinystories",
                    "family": "general_language",
                    "language": "text",
                }
            )
    return rows


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for record in payload["files"]:
        source = (ROOT / record["path"]).resolve()
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rows.append(
            {
                "id": f"{record['source_id']}:{source.relative_to(ROOT)}",
                "text": text,
                "source_id": record["source_id"],
                "family": record["family"],
                "language": record["language"],
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            handle.write(line)
            digest.update(line)
    return {"path": str(path), "rows": len(rows), "sha256": digest.hexdigest()}


def counts(rows: list[dict]) -> dict:
    families: dict[str, int] = {}
    languages: dict[str, int] = {}
    for row in rows:
        length = len(row["text"])
        families[row["family"]] = families.get(row["family"], 0) + length
        languages[row["language"]] = languages.get(row["language"], 0) + length
    return {"characters_by_family": families, "characters_by_language": languages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1/corpus_v1",
    )
    args = parser.parse_args()
    sources = read_manifest(ROOT / ".data/nlp/mixed_v1/code_manifest.json")
    sources += read_manifest(ROOT / ".data/nlp/mixed_v1/docs_manifest.json")
    sources += read_manifest(ROOT / ".data/nlp/mixed_v1/local_manifest.json")
    sources += read_manifest(ROOT / ".data/nlp/mixed_v1/system_manifest.json")
    split_rows = {
        "train": read_seed("train") + sources,
        "validation": read_seed("validation"),
        "test": read_seed("test"),
    }
    manifests = {}
    stats = {}
    validation_hashes = {
        hashlib.sha256(normalize(row["text"]).encode()).hexdigest()
        for split in ("validation", "test")
        for row in split_rows[split]
    }
    for split, rows in split_rows.items():
        cleaned, cleaning = sanitize(rows)
        if split == "train":
            cleaned = [
                row
                for row in cleaned
                if hashlib.sha256(normalize(row["text"]).encode()).hexdigest()
                not in validation_hashes
            ]
        manifests[split] = write_jsonl(args.output_root / f"{split}.jsonl", cleaned)
        manifests[split]["cleaning"] = cleaning
        stats[split] = {**counts(cleaned), **manifests[split]}
    report = {
        "version": "nlp-mixed-corpus-v1",
        "splits": stats,
        "split_policy": "source-disjoint; TinyStories eval held out; repository files remain entirely in train",
        "training_use_exclusions": ["all benchmark rows"],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
