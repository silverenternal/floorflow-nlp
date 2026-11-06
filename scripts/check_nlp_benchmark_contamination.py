#!/usr/bin/env python3
"""Fingerprint and scan benchmark fields after the external candidate freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.benchmark_contamination import (
    complete_single_scan,
    index_training,
    reserve_single_scan,
    scan_benchmark,
    validate_external_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError(f"input path is outside its allowed root: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nlp_benchmark_contamination_v2.json")
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/nlp_external_evaluation_freeze_v11.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_benchmark_contamination_v11.json")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output_path = args.output.resolve()
    marker_path = output_path.with_name(output_path.stem + ".started.json")
    if not output_path.is_relative_to(root) or not marker_path.is_relative_to(root):
        raise ValueError("scan report and marker must remain inside the repository")
    if marker_path.exists() or output_path.exists():
        raise ValueError("benchmark scan has already been reserved or completed; refusing to reopen test inputs")
    config_path = args.config.resolve(strict=True)
    freeze_path = args.freeze.resolve(strict=True)
    config_hash = sha256_file(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    benchmark_manifest_path = contained_path(root, config["benchmark_manifest"])
    benchmark_manifest_hash = sha256_file(benchmark_manifest_path)
    freeze_hash = sha256_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validate_external_freeze(
        freeze,
        benchmark_manifest_sha256=benchmark_manifest_hash,
        required_fields=tuple(config["required_freeze_identity"]),
    )
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if benchmark_manifest.get("training_use") != "forbidden":
        raise ValueError("benchmark manifest must declare evaluation-only use")
    benchmarks = benchmark_manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmark manifest has no benchmark records")
    width = int(config["shingle_characters"])
    threshold = float(config["near_overlap_threshold"])
    training_paths = [contained_path(root, value) for value in config["training_sources"]]
    training = index_training(training_paths, width)
    reserve_single_scan(
        marker_path,
        freeze_sha256=freeze_hash,
        benchmark_manifest_sha256=benchmark_manifest_hash,
    )
    scans = {}
    failures = []
    unassessed = []
    seen_benchmarks: set[str] = set()
    for record in benchmarks:
        benchmark_id = str(record.get("id") or "")
        if not benchmark_id or benchmark_id in seen_benchmarks:
            raise ValueError("benchmark IDs must be nonempty and unique")
        seen_benchmarks.add(benchmark_id)
        if record.get("usage") != "evaluation_only" or record.get("split") != "test":
            raise ValueError(f"benchmark is not frozen evaluation-only test material: {benchmark_id}")
        path = contained_path(root, str(record["path"]))
        result = scan_benchmark(
            path,
            training,
            near_threshold=threshold,
            minimum_substring_characters=int(config["minimum_substring_characters"]),
            expected_sha256=str(record["sha256"]),
            expected_rows=int(record["rows"]),
        )
        scans[benchmark_id] = result
        if len(result["exact_matches"]) > int(config["exact_overlap_limit"]):
            failures.append(f"{benchmark_id}:exact_document_overlap")
        if len(result["substring_matches"]) > int(config["substring_overlap_limit"]):
            failures.append(f"{benchmark_id}:exact_substring_overlap")
        if len(result["near_matches"]) > int(config["near_overlap_limit"]):
            failures.append(f"{benchmark_id}:near_overlap")
        if result["short_fields_unassessed_for_substring"] or result["short_fields_unassessed_for_near"]:
            unassessed.append(benchmark_id)
    post_training_hashes = {str(path): sha256_file(path) for path in training_paths}
    if post_training_hashes != training.input_sha256:
        raise ValueError("training corpus changed during benchmark contamination scan")
    if (
        sha256_file(config_path) != config_hash
        or sha256_file(benchmark_manifest_path) != benchmark_manifest_hash
        or sha256_file(freeze_path) != freeze_hash
    ):
        raise ValueError("configuration, benchmark manifest, or freeze changed during scan")
    report = {
        "version": "nlp-benchmark-contamination-audit-v11",
        "status": "fail" if failures else "partial" if unassessed else "pass",
        "errors": failures,
        "unassessed_benchmarks": unassessed,
        "training_rows": training.rows,
        "training_documents": len(training.normalized_documents),
        "training_shingle_count": len(training.shingle_documents),
        "benchmark_scans": scans,
        "frozen_candidate_id": freeze["candidate_id"],
        "single_scan_marker": str(marker_path.relative_to(root)),
        "input_sha256": {
            "config": config_hash,
            "benchmark_manifest": benchmark_manifest_hash,
            "external_freeze": freeze_hash,
            "training_sources": {
                str(Path(path).resolve().relative_to(root)): digest
                for path, digest in training.input_sha256.items()
            },
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/floorflow/nlp_flow/benchmark_contamination.py",
            )
        },
        "benchmark_test_content_scanned": True,
        "model_quality_evaluated": False,
        "evidence_boundary": (
            "Benchmark test fields are scanned only for contamination after identity freeze and are not used "
            "in training, prompt construction, model selection, or quality scoring. "
            "A scan with unassessed short fields is partial and cannot clear the contamination gate."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    complete_single_scan(marker_path, report_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest())
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_rows": report["training_rows"],
                "benchmarks": len(scans),
                "errors": len(failures),
                "unassessed_benchmarks": unassessed,
                "benchmark_test_content_scanned": True,
                "model_quality_evaluated": False,
            },
            indent=2,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
