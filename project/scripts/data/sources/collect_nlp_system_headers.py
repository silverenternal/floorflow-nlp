#!/usr/bin/env python3
"""Collect local C/C++ system headers with explicit license partitions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MAX_FILE_BYTES = 2_000_000
HEADER_SOURCES = (
    {
        "id": "system_boost_headers",
        "root": Path("/usr/include/boost"),
        "patterns": ("*.hpp", "*.h"),
        "license": "BSL-1.0",
        "usage": "training",
        "language": "cpp",
        "family": "code",
    },
    {
        "id": "system_eigen_headers",
        "root": Path("/usr/include/eigen3"),
        "patterns": ("*.h",),
        "license": "MPL-2.0",
        "usage": "training_conditional",
        "language": "cpp",
        "family": "code",
    },
    {
        "id": "system_openssl_headers",
        "root": Path("/usr/include/openssl"),
        "patterns": ("*.h",),
        "license": "Apache-2.0",
        "usage": "training",
        "language": "c",
        "family": "engineering_acceleration",
    },
    {
        "id": "system_linux_uapi_headers",
        "root": Path("/usr/include/linux"),
        "patterns": ("*.h",),
        "license": "GPL-2.0 WITH Linux-syscall-note",
        "usage": "training_conditional",
        "language": "linux",
        "family": "engineering_acceleration",
    },
)


def main() -> None:
    output_root = ROOT / ".data/nlp/mixed_v1"
    records = []
    for source in HEADER_SOURCES:
        if not source["root"].is_dir():
            continue
        target_root = (
            output_root
            / source["family"]
            / source["language"]
            / source["id"]
        )
        files = sorted(
            path
            for pattern in source["patterns"]
            for path in source["root"].rglob(pattern)
        )
        for path in files:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            relative = path.relative_to(source["root"])
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            content = target.read_bytes()
            records.append(
                {
                    "source_id": source["id"],
                    "repository": str(source["root"]),
                    "revision": "local-system-2026-11-01",
                    "license": source["license"],
                    "usage": source["usage"],
                    "language": source["language"],
                    "family": source["family"],
                    "path": str(target),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    report = {
        "version": "nlp-system-header-source-v1",
        "files": records,
        "total_files": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
    }
    (output_root / "system_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": len(records), "bytes": report["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
