#!/usr/bin/env python3
"""Audit supplement metadata without loading source or target text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.nlp_flow.multilingual_supplement import audit_multilingual_supplement

ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_TEXT_FIELDS = {"source", "target", "gold", "reference", "solution", "tests", "test_cases"}


def load_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"supplement metadata row {line_number} must be an object")
            forbidden = _FORBIDDEN_TEXT_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(f"supplement row {line_number} contains content fields: {sorted(forbidden)}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / ".data/nlp/multilingual_supplement_v11/metadata.jsonl",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs/nlp_multilingual_supplement_v11.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/nlp_multilingual_source_registry_v11.json",
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_multilingual_supplement_build_v11.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_multilingual_supplement_v11.json",
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        report = {
            "version": "nlp-multilingual-supplement-audit-v11",
            "status": "blocked_no_supplement_manifest",
            "metadata_rows": 0,
            "test_content_evaluated": False,
            "evidence_boundary": (
                "No supplement manifest exists; no source, target, test, or benchmark data were opened."
            ),
        }
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        report = audit_multilingual_supplement(load_metadata(args.manifest), registry["sources"], policy)
        if args.build_report.is_file():
            build = json.loads(args.build_report.read_text(encoding="utf-8"))
            report["materialization"] = {
                "status": build["status"],
                "source_revisions_verified": build["source_revisions_verified"],
                "train_action_rows": build["train_action_rows"],
                "train_action_contract": build["train_action_contract"],
                "validation_action_rows_written": build["validation_action_rows_written"],
                "training_contamination_scan": build["training_contamination_scan"],
                "action_rejection_count": len(build["action_rejections"]),
                "input_sha256": build["input_sha256"],
                "output_sha256": build["output_sha256"],
                "implementation_sha256": build["implementation_sha256"],
                "test_content_evaluated": False,
            }
            if build["status"] != "pass":
                report["status"] = "partial" if report["status"] == "pass" else report["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "metadata_rows", "test_content_evaluated")}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
