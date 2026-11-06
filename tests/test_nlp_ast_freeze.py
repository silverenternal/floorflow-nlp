from __future__ import annotations

import hashlib
import json
from pathlib import Path

from floorflow.nlp_flow.ast_codec import parser_provenance
from floorflow.nlp_flow.ast_freeze import (
    artifact_inventory,
    build_ast_freeze_manifest,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, object]:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "rows": 1,
            "by_split": {"train": 1, "validation": 0, "test": 0},
            "corpus_sha256": corpus_hash,
            "source_integrity": True,
            "split_integrity": True,
        },
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"version":"1.0"}\n', encoding="utf-8")
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    tokenizer_report = tmp_path / "tokenizer_report.json"
    _write_json(
        tokenizer_report,
        {
            "sha256": tokenizer_hash,
            "fit_split": "train",
            "roundtrip_pass": True,
            "vocab_size": 8192,
        },
    )
    recovery = tmp_path / "recovery.jsonl"
    recovery.write_text("{}\n", encoding="utf-8")
    recovery_hash = hashlib.sha256(recovery.read_bytes()).hexdigest()
    holdout = tmp_path / "holdout.json"
    _write_json(
        holdout,
        {
            "status": "pass",
            "construct_isolated": True,
            "names_isolated": True,
            "complexity_covered": True,
            "structure_covered": True,
            "error_recovery": {
                "suite_path": "recovery.jsonl",
                "suite_sha256": recovery_hash,
            },
        },
    )
    negatives_suite = tmp_path / "negatives.jsonl"
    negatives_suite.write_text("{}\n", encoding="utf-8")
    negatives_hash = hashlib.sha256(negatives_suite.read_bytes()).hexdigest()
    negatives = tmp_path / "negatives.json"
    _write_json(
        negatives,
        {
            "status": "pass",
            "validation_test_provenance": [],
            "suite_path": "negatives.jsonl",
            "suite_sha256": negatives_hash,
        },
    )
    schema = tmp_path / "schema.json"
    _write_json(
        schema,
        {
            "parser_binding": {
                language: parser_provenance(language)
                for language in ("python", "rust", "cpp")
            }
        },
    )
    prompts = tmp_path / "prompts.json"
    _write_json(prompts, {"status": "frozen"})
    artifacts = [
        "corpus.jsonl",
        "manifest.json",
        "tokenizer.json",
        "tokenizer_report.json",
        "recovery.jsonl",
        "holdout.json",
        "negatives.jsonl",
        "negatives.json",
        "schema.json",
        "prompts.json",
    ]
    return {
        "artifacts": artifacts,
        "corpus": {
            "path": "corpus.jsonl",
            "manifest": "manifest.json",
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "report": "tokenizer_report.json",
        },
        "holdout_report": "holdout.json",
        "negative_report": "negatives.json",
        "parser_schema": "schema.json",
        "prompt_contract": "prompts.json",
        "parser_languages": ["python", "rust", "cpp"],
        "access_policy": {
            "test_read_during_selection": False,
            "test_predictions_read": False,
            "test_used_for_model_selection": False,
            "split_integrity_metadata_only": True,
            "architecture_freeze_required_after_selection": True,
        },
    }


def test_ast_freeze_manifest_hash_locks_every_required_artifact(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    manifest = build_ast_freeze_manifest(tmp_path, config)

    assert manifest["status"] == "pass"
    assert manifest["checks"]["test_not_used_during_selection"]
    assert all(manifest["artifact_hashes"].values())


def test_artifact_inventory_detects_post_freeze_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    before = artifact_inventory(tmp_path, ["artifact.txt"])
    artifact.write_text("changed\n", encoding="utf-8")
    after = artifact_inventory(tmp_path, ["artifact.txt"])

    assert before["artifact.txt"] != after["artifact.txt"]
