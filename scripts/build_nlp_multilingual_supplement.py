#!/usr/bin/env python3
"""Build the pinned multilingual supplement and its train-only action data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.multilingual_supplement_build import build_multilingual_supplement

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_multilingual_supplement_sources_v11.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata, train_actions, report = build_multilingual_supplement(config, root=ROOT)
    metadata_path = ROOT / str(config["output_metadata"])
    actions_path = ROOT / str(config["output_train_actions"])
    report_path = ROOT / str(config["output_report"])
    _write_jsonl(metadata_path, metadata)
    _write_jsonl(actions_path, train_actions)
    report["metadata_path"] = str(metadata_path.relative_to(ROOT))
    report["train_actions_path"] = str(actions_path.relative_to(ROOT))
    report["input_sha256"] = {
        "source_config": _sha256_file(args.config.resolve()),
        "source_registry": _sha256_file(ROOT / str(config["registry"])),
        "coverage_policy": _sha256_file(ROOT / str(config["policy"])),
        "ast_train_corpus": _sha256_file(ROOT / str(config["existing_train_ast_corpus"])),
        "mixed_train_corpus": report["training_contamination_scan"]["training_source_sha256"],
        "transition_grammar": _sha256_file(ROOT / str(config["grammars"])),
    }
    report["output_sha256"] = {
        "metadata": _sha256_file(metadata_path),
        "train_actions": _sha256_file(actions_path),
    }
    report["implementation_sha256"] = {
        "builder": _sha256_file(Path(__file__).resolve()),
        "build_module": _sha256_file(ROOT / "src/floorflow/nlp_flow/multilingual_supplement_build.py"),
        "audit_module": _sha256_file(ROOT / "src/floorflow/nlp_flow/multilingual_supplement.py"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "metadata_rows", "train_action_rows")
            }
            | {"training_contamination": report["training_contamination_scan"]["status"]},
            indent=2,
        )
    )
    if report["status"] not in {
        "pass",
        "pass_with_explicit_rejections",
        "partial_unassessed_short_candidates",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
