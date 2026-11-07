#!/usr/bin/env python3
"""Audit fetched code/documentation files against the source registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1/code_manifest.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "project/configs/data/sources/nlp_source_registry.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    known = {source["id"]: source for source in registry["sources"]}
    errors = []
    language_bytes: Counter[str] = Counter()
    license_files: Counter[str] = Counter()
    for row in manifest["files"]:
        source = known.get(row["source_id"])
        if source is None:
            errors.append(f"unknown source: {row['source_id']}")
            continue
        for field in ("repository", "revision", "license", "language", "usage"):
            if row[field] != source[field]:
                errors.append(f"{row['path']} has mismatched {field}")
        path = ROOT / row["path"]
        if not path.is_file():
            errors.append(f"missing file: {row['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            errors.append(f"hash mismatch: {row['path']}")
        language_bytes[row["language"]] += int(row["bytes"])
        license_files[row["license"]] += 1
    report = {
        "version": "nlp-code-source-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "manifest": str(args.manifest.relative_to(ROOT)),
        "files": manifest["total_files"],
        "bytes": manifest["total_bytes"],
        "bytes_by_language": dict(sorted(language_bytes.items())),
        "files_by_license": dict(sorted(license_files.items())),
        "evidence_boundary": "Pinned source collection only; no cleaning or tokenization yet.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
