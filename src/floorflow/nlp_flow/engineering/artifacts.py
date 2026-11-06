"""Artifact hashing and atomic writes for nlp_flow runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.engineering.storage import write_json_atomic


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_file_record(path: str | Path) -> dict[str, object]:
    source = Path(path)
    rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {source}:{rows}")
            payload = json.loads(line)
            if not isinstance(payload, dict) or not {"id", "text"}.issubset(payload):
                raise ValueError(f"invalid JSONL row at {source}:{rows}")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "rows": rows,
        "sha256": sha256_file(source),
    }


def build_jsonl_manifest(split_paths: dict[str, str | Path]) -> dict[str, object]:
    """Hash train/validation/test JSONL files and reject missing/duplicate splits."""

    if set(split_paths) != {"train", "validation", "test"}:
        raise ValueError("manifest requires exactly train, validation, and test")
    records = {
        split: jsonl_file_record(path)
        for split, path in sorted(split_paths.items())
    }
    hashes = [record["sha256"] for record in records.values()]
    if len(hashes) != len(set(hashes)):
        raise ValueError("split files must be distinct")
    return {
        "version": "nlp-jsonl-manifest-v1",
        "splits": records,
    }


__all__ = ["build_jsonl_manifest", "jsonl_file_record", "sha256_file", "write_json_atomic"]
