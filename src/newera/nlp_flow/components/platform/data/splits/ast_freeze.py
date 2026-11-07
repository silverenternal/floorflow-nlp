"""Hash-lock AST corpus, parser, tokenizer, prompt, and holdout artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parser_provenance


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_inventory(
    root: Path,
    artifacts: Sequence[str],
) -> dict[str, str | None]:
    return {
        path: sha256_file(root / path) if (root / path).is_file() else None
        for path in artifacts
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _report_suite_hash_matches(root: Path, report: Mapping[str, object]) -> bool:
    path = report.get("suite_path")
    expected = report.get("suite_sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        return False
    suite = root / path
    return suite.is_file() and sha256_file(suite) == expected


def build_ast_freeze_manifest(
    root: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    artifacts = [str(path) for path in config["artifacts"]]
    hashes = artifact_inventory(root, artifacts)
    corpus_config = config["corpus"]
    tokenizer_config = config["tokenizer"]
    corpus_path = root / str(corpus_config["path"])
    corpus_manifest = _load_json(root / str(corpus_config["manifest"]))
    tokenizer_path = root / str(tokenizer_config["path"])
    tokenizer_report = _load_json(root / str(tokenizer_config["report"]))
    holdout = _load_json(root / str(config["holdout_report"]))
    negatives = _load_json(root / str(config["negative_report"]))
    parser_schema = _load_json(root / str(config["parser_schema"]))
    prompt_contract = root / str(config["prompt_contract"])
    languages = [str(value) for value in config["parser_languages"]]
    parser_fingerprints = {
        language: parser_provenance(language) for language in languages
    }
    parser_binding = parser_schema.get("parser_binding", {})
    parser_hashes_match = all(
        isinstance(parser_binding.get(language), Mapping)
        and parser_binding[language].get("grammar_sha256")
        == parser_fingerprints[language]["grammar_sha256"]
        for language in languages
    )
    corpus_hash = sha256_file(corpus_path) if corpus_path.is_file() else None
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
    access_policy = dict(config["access_policy"])
    error_recovery = holdout.get("error_recovery", {})
    recovery_path = (
        error_recovery.get("suite_path")
        if isinstance(error_recovery, Mapping)
        else None
    )
    recovery_hash_matches = False
    if isinstance(recovery_path, str):
        recovery_file = root / recovery_path
        recovery_hash_matches = (
            recovery_file.is_file()
            and isinstance(error_recovery, Mapping)
            and sha256_file(recovery_file) == error_recovery.get("suite_sha256")
        )
    checks = {
        "all_artifacts_present": all(value is not None for value in hashes.values()),
        "corpus_hash_matches_manifest": (
            corpus_hash is not None
            and corpus_hash == corpus_manifest.get("corpus_sha256")
        ),
        "corpus_source_and_split_integrity": (
            corpus_manifest.get("source_integrity") is True
            and corpus_manifest.get("split_integrity") is True
        ),
        "tokenizer_hash_matches_report": (
            tokenizer_hash is not None
            and tokenizer_hash == tokenizer_report.get("sha256")
        ),
        "tokenizer_train_only_and_roundtrip": (
            tokenizer_report.get("fit_split") == "train"
            and tokenizer_report.get("roundtrip_pass") is True
        ),
        "parser_fingerprints_match_schema": parser_hashes_match,
        "holdout_suite_passed": (
            holdout.get("status") == "pass"
            and holdout.get("construct_isolated") is True
            and holdout.get("names_isolated") is True
            and holdout.get("complexity_covered") is True
            and holdout.get("structure_covered") is True
        ),
        "error_recovery_suite_hash_matches": recovery_hash_matches,
        "negative_suite_passed": (
            negatives.get("status") == "pass"
            and negatives.get("validation_test_provenance") == []
            and _report_suite_hash_matches(root, negatives)
        ),
        "prompt_contract_frozen": (
            prompt_contract.is_file()
            and _load_json(prompt_contract).get("status") == "frozen"
        ),
        "test_not_used_during_selection": (
            access_policy.get("test_read_during_selection") is False
            and access_policy.get("test_predictions_read") is False
            and access_policy.get("test_used_for_model_selection") is False
        ),
    }
    return {
        "version": "nlp-ast-corpus-freeze-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "artifact_hashes": hashes,
        "parser_fingerprints": parser_fingerprints,
        "corpus": {
            "path": str(corpus_config["path"]),
            "sha256": corpus_hash,
            "rows": corpus_manifest.get("rows"),
            "by_split": corpus_manifest.get("by_split"),
        },
        "tokenizer": {
            "path": str(tokenizer_config["path"]),
            "sha256": tokenizer_hash,
            "fit_split": tokenizer_report.get("fit_split"),
            "vocab_size": tokenizer_report.get("vocab_size"),
        },
        "holdout_report": str(config["holdout_report"]),
        "negative_report": str(config["negative_report"]),
        "prompt_contract": str(config["prompt_contract"]),
        "access_policy": access_policy,
        "checks": checks,
        "evidence_boundary": (
            "Artifact hash lock only. Split-integrity metadata was inspected; "
            "no test prediction, outcome, or model-selection result is included."
        ),
    }


__all__ = ["artifact_inventory", "build_ast_freeze_manifest", "sha256_file"]
