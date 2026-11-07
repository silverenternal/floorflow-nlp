#!/usr/bin/env python3
"""Fetch pinned Python, Rust, and C++ source/documentation files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from newera.nlp_flow.components.platform.data.sources.source_fetch import matches

ROOT = Path(__file__).resolve().parents[4]
MAX_FILE_BYTES = 2_000_000


def download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "-o",
            str(target),
            url,
        ],
        check=True,
    )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_source(source: dict, archive: Path, output_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                continue
            relative = PurePosixPath(member.name)
            parts = relative.parts
            if len(parts) < 2:
                continue
            candidate = str(PurePosixPath(*parts[1:]))
            if not matches(candidate, source):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                continue
            content = stream.read()
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            target = output_root / source["family"] / source["language"] / source["id"] / candidate
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append(
                {
                    "source_id": source["id"],
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "license": source["license"],
                    "usage": source["usage"],
                    "language": source["language"],
                    "family": source["family"],
                    "path": str(target),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "project/configs/data/sources/nlp_source_registry.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["code"],
        choices=("code", "model_architecture", "engineering_acceleration"),
    )
    parser.add_argument("--source-ids", nargs="+", default=None)
    parser.add_argument("--manifest-name", default="code_manifest.json")
    parser.add_argument("--merge-existing-manifest", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    archive_root = args.output_root / "_archives"
    records = []
    for source in registry["sources"]:
        if source["family"] not in args.families:
            continue
        if args.source_ids is not None and source["id"] not in args.source_ids:
            continue
        archive = archive_root / f"{source['id']}.tar.gz"
        url = f"https://codeload.github.com/{source['repository']}/tar.gz/{source['revision']}"
        if archive.is_file() and tarfile.is_tarfile(archive):
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        else:
            archive_sha = download(url, archive)
        extracted = extract_source(source, archive, args.output_root)
        if not extracted:
            raise RuntimeError(f"source produced no files: {source['id']}")
        records.extend(extracted)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "archive_sha256": archive_sha,
                    "files": len(extracted),
                    "bytes": sum(int(row["bytes"]) for row in extracted),
                }
            ),
            flush=True,
        )
    selected_ids = {row["source_id"] for row in records}
    existing_manifest = args.output_root / args.manifest_name
    if args.merge_existing_manifest and existing_manifest.is_file():
        existing_payload = json.loads(
            (args.output_root / args.manifest_name).read_text(encoding="utf-8")
        )
        existing_rows = [
            row
            for row in existing_payload.get("files", [])
            if row["source_id"] not in selected_ids
        ]
        records = existing_rows + records
    manifest = {
        "version": "nlp-code-source-fetch-v1",
        "files": records,
        "total_files": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
    }
    target = args.output_root / args.manifest_name
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
