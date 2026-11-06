#!/usr/bin/env python3
"""Audit AST construct and complexity holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.ast_holdout import audit_ast_holdouts, load_ast_corpus
from floorflow.nlp_flow.ast_recovery import build_error_recovery_suite

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_ast_corpus_v1.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / ".data/nlp/ast_core_v1/corpus.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_construct_holdout_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = config["split"]
    rows = load_ast_corpus(args.corpus)
    report = audit_ast_holdouts(
        rows,
        heldout_constructs=split.get("heldout_constructs", ()),
        heldout_name_prefixes=split.get("heldout_name_prefixes", ()),
        heldout_name_modulus=int(split.get("heldout_name_modulus", 80)),
        structure_families=split.get("structure_holdout_families", ()),
        structure_cases_per_family=int(
            split.get("structure_holdout_cases_per_family", 16)
        ),
    )
    recovery_suite, recovery_report = build_error_recovery_suite(rows)
    recovery_path = ROOT / str(split["error_recovery_suite"])
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_payload = "".join(
        json.dumps(case, sort_keys=True) + "\n" for case in recovery_suite
    )
    recovery_path.write_text(recovery_payload, encoding="utf-8")
    recovery_report["suite_path"] = str(recovery_path.relative_to(ROOT))
    recovery_report["suite_sha256"] = hashlib.sha256(
        recovery_payload.encode("utf-8")
    ).hexdigest()
    report["error_recovery"] = recovery_report
    if (
        not recovery_report["all_recovered_to_expected_root"]
        or recovery_report["training_hash_overlap"]
    ):
        report["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
