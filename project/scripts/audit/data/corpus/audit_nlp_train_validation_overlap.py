#!/usr/bin/env python3
"""Run a fingerprinted train/validation source overlap audit on CPU."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from newera.nlp_flow.components.ast.diagnostics.tracing.ast_trace_provenance import file_sha256
from newera.nlp_flow.components.research.training.data.train_validation_overlap import audit_train_validation_overlap

ROOT = Path(__file__).resolve().parents[5]
_TRAIN_OR_VALIDATION = re.compile(r'"split"\s*:\s*"(train|validation)"')


def load_train_validation(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            # Skip test rows before decoding source fields.
            if _TRAIN_OR_VALIDATION.search(line) is None:
                continue
            row = json.loads(line)
            if row.get("split") not in {"train", "validation"}:
                raise ValueError("split metadata mismatch")
            rows.append({key: row[key] for key in ("case_id", "language", "split", "source", "source_sha256")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/evaluation/contamination/nlp_benchmark_contamination_base.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_train_validation_overlap_v11.json")
    args = parser.parse_args()
    config_before = file_sha256(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    before = file_sha256(args.corpus)
    report = audit_train_validation_overlap(
        load_train_validation(args.corpus),
        width=int(config["shingle_characters"]),
        threshold=float(config["near_overlap_threshold"]),
    )
    if file_sha256(args.corpus) != before or file_sha256(args.config) != config_before:
        raise ValueError("corpus or configuration changed during overlap audit")
    report["input_sha256"] = {"corpus": before, "config": config_before}
    report["implementation_sha256"] = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "src/newera/nlp_flow/components/research/training/data/train_validation_overlap.py",
            ROOT / "src/newera/nlp_flow/components/research/evaluation/benchmarks/benchmark_field_overlap.py",
            ROOT / "src/newera/nlp_flow/components/research/evaluation/benchmarks/benchmark_contamination.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_documents": report["training_documents"],
                "validation_documents": report["validation_documents"],
                "matches": len(report["matches"]),
                "short_fallback_matches": (
                    0 if report["short_fallback"] is None else len(report["short_fallback"]["matches"])
                ),
                "short_fields_unassessed": len(report["short_validation_fields_unassessed_for_near"]),
                "test_content_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
