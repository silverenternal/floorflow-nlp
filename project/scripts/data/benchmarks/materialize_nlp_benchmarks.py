#!/usr/bin/env python3
"""Materialize benchmark rows for evaluation-only provenance and contamination checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ROWS_API = "https://datasets-server.huggingface.co/rows"


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
        return json.load(response)


def materialize(benchmark: dict, output: Path) -> dict[str, object]:
    rows: list[dict] = []
    offset = 0
    page_size = 100
    while True:
        page = fetch_page(
            benchmark["dataset"],
            benchmark["config"],
            benchmark["split"],
            offset,
            page_size,
        )
        rows.extend(page["rows"])
        offset += len(page["rows"])
        if offset >= int(page["num_rows_total"]):
            break
        if not page["rows"]:
            raise RuntimeError(f"benchmark stream stalled: {benchmark['id']}")
        time.sleep(0.05)
    target = output / f"{benchmark['id']}.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "id": f"{benchmark['id']}:{row['row_idx']}",
                "language": benchmark["language"],
                "fields": row["row"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "id": benchmark["id"],
        "path": str(target),
        "rows": len(rows),
        "sha256": digest,
        "dataset": benchmark["dataset"],
        "dataset_revision": benchmark["dataset_revision"],
        "config": benchmark["config"],
        "split": benchmark["split"],
        "usage": "evaluation_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "project/configs/evaluation/benchmarks/nlp_benchmark_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".data/nlp/benchmarks_v1",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = [
        materialize(benchmark, args.output_root)
        for benchmark in manifest["benchmarks"]
    ]
    report = {
        "version": "nlp-benchmark-materialization-v1",
        "training_use": "forbidden",
        "benchmarks": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
