#!/usr/bin/env python3
"""Fetch pinned Linux documentation through directory listing and raw files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from floorflow.nlp_flow.source_fetch import LINUX_DOCUMENTATION_DIRS

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "torvalds/linux"
REVISION = "cbbddb6b08cc9900a87cccaa9b22bbe06aa4f50a"


def github_tree() -> list[dict]:
    url = (
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/"
        f"{REVISION}:Documentation?recursive=1"
    )
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "nlp-flow"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if payload.get("truncated"):
        raise RuntimeError("Linux Documentation tree response was truncated")
    prefixes = tuple(path.removeprefix("Documentation/") for path in LINUX_DOCUMENTATION_DIRS)
    return [
        item
        for item in payload["tree"]
        if item["type"] == "blob"
        and item["path"].endswith(".rst")
        and item["path"].startswith(prefixes)
    ]


def download(path: str) -> bytes:
    encoded = urllib.parse.quote(path, safe="/")
    url = f"https://github.com/{REPOSITORY}/raw/{REVISION}/{encoded}"
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "nlp-flow-source-fetch"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2.0 ** attempt)
    raise AssertionError("unreachable")


def fetch_entry(entry: dict) -> tuple[dict, bytes]:
    return entry, download(f"Documentation/{entry['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/mixed_v1",
    )
    parser.add_argument("--per-prefix", type=int, default=20)
    args = parser.parse_args()
    target_root = (
        args.output_root
        / "engineering_acceleration"
        / "linux"
        / "linux_documentation"
    )
    records = []
    all_entries = sorted(github_tree(), key=lambda item: item["path"])
    entries = []
    for prefix in (path.removeprefix("Documentation/") for path in LINUX_DOCUMENTATION_DIRS):
        entries.extend(
            [item for item in all_entries if item["path"].startswith(prefix)][: args.per_prefix]
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        fetched = list(executor.map(fetch_entry, entries))
    for item, content in fetched:
        target = target_root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append(
            {
                "source_id": "linux_documentation",
                "repository": REPOSITORY,
                "revision": REVISION,
                "license": "GPL-2.0 WITH Linux-syscall-note",
                "usage": "training_conditional",
                "language": "linux",
                "family": "engineering_acceleration",
                "path": str(target),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError("Linux documentation fetch produced no files")
    docs_manifest = args.output_root / "docs_manifest.json"
    payload = json.loads(docs_manifest.read_text(encoding="utf-8"))
    existing = [
        row for row in payload["files"] if row["source_id"] != "linux_documentation"
    ]
    payload["files"] = existing + records
    payload["total_files"] = len(payload["files"])
    payload["total_bytes"] = sum(int(row["bytes"]) for row in payload["files"])
    docs_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "manifest": str(docs_manifest)}, indent=2))


if __name__ == "__main__":
    main()
