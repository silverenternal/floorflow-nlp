#!/usr/bin/env python3
"""Audit canonical AST round trips, parser provenance, and metric attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import (
    ast_metrics,
    deserialize_ast,
    parse_ast,
    parser_fingerprint,
    render_source,
    serialize_ast,
    validate_ast,
)

ROOT = Path(__file__).resolve().parents[5]
EXTENSION_LANGUAGE = {
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".py": "python",
    ".rs": "rust",
}
MAX_AUDIT_FILE_BYTES = 64 * 1024
MAX_FILES_PER_LANGUAGE = 8


def _candidate_files() -> list[Path]:
    candidates: list[Path] = []
    for manifest_path in (
        ROOT / ".data/nlp/mixed_v1/code_manifest.json",
        ROOT / ".data/nlp/mixed_v1/local_manifest.json",
    ):
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload.get("files", []):
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = ROOT / path
            if (
                path.suffix in EXTENSION_LANGUAGE
                and path.is_file()
                and int(row.get("bytes", path.stat().st_size)) <= MAX_AUDIT_FILE_BYTES
            ):
                candidates.append(path)
    return sorted(set(candidates))


def _roundtrip_audit() -> dict[str, object]:
    selected: dict[str, list[Path]] = {language: [] for language in EXTENSION_LANGUAGE.values()}
    for path in _candidate_files():
        language = EXTENSION_LANGUAGE[path.suffix]
        if len(selected[language]) < MAX_FILES_PER_LANGUAGE:
            selected[language].append(path)

    languages: dict[str, object] = {}
    errors: list[str] = []
    for language, paths in selected.items():
        if not paths:
            errors.append(f"no real {language} source selected")
            continue
        rows = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            try:
                graph = parse_ast(source, language)
            except ValueError:
                rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": len(source.encode("utf-8")),
                        "status": "parse_rejected",
                    }
                )
                continue
            restored = deserialize_ast(serialize_ast(graph))
            validate_ast(restored, source=source)
            valid = restored == graph and render_source(restored) == source
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": len(source.encode("utf-8")),
                    "status": "roundtrip_passed" if valid else "roundtrip_failed",
                    "nodes": _node_count(graph),
                    "scopes": len({node.scope_id for _path, node in _walk(graph)}),
                    "symbols": sum(
                        node.symbol_name is not None
                        for _path, node in _walk(graph)
                    ),
                    "roundtrip_equal": valid,
                }
            )
            if not valid:
                errors.append(f"{language} graph round trip failed: {path}")
        if not any(row["status"] == "roundtrip_passed" for row in rows):
            errors.append(f"no parseable real {language} source selected")
        languages[language] = {
            "files": rows,
            "passed": all(
                row["status"] in {"parse_rejected", "roundtrip_passed"}
                for row in rows
            )
            and any(row["status"] == "roundtrip_passed" for row in rows),
        }

    malformed = {
        "python": "def broken(:\n",
        "rust": "fn broken( {\n",
        "cpp": "int broken( {\n",
    }
    malformed_results = {}
    for language, source in malformed.items():
        try:
            parse_ast(source, language)
        except ValueError:
            malformed_results[language] = "rejected"
        else:
            malformed_results[language] = "accepted"
            errors.append(f"malformed {language} source passed AST parsing")

    parser_provenance = {
        language: _fingerprint_payload(parser_fingerprint(language))
        for language in ("python", "rust", "cpp")
    }
    schema = json.loads(
        (ROOT / "project/configs/structured_generation/ast_corpus_diagnostic/ast_prompts_schema/nlp_ast_schema.json").read_text(encoding="utf-8")
    )
    for language, binding in schema["parser_binding"].items():
        observed = parser_provenance[language]
        for key in ("version", "abi_version", "native_sha256", "grammar_sha256"):
            if binding[key] != observed[key]:
                errors.append(
                    f"{language} frozen parser {key} mismatch: "
                    f"{binding[key]} != {observed[key]}"
                )
    expected_metrics = set(schema["metrics"])
    observed_metrics = set(ast_metrics(
        parse_ast("def f():\n    return 1\n", "python"),
        parse_ast("def f():\n    return 1\n", "python"),
    ))
    if expected_metrics != observed_metrics:
        errors.append(
            f"frozen AST metrics mismatch: {sorted(expected_metrics)} != "
            f"{sorted(observed_metrics)}"
        )

    return {
        "version": "nlp-ast-roundtrip-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "parser_provenance": parser_provenance,
        "frozen_schema": str((ROOT / "project/configs/structured_generation/ast_corpus_diagnostic/ast_prompts_schema/nlp_ast_schema.json").relative_to(ROOT)),
        "frozen_metrics": sorted(expected_metrics),
        "languages": languages,
        "malformed_source_attacks": malformed_results,
        "serialization": {
            "format": "deterministic compact JSON",
            "preserves": [
                "node types",
                "typed child edges",
                "byte spans",
                "leaf lexical values",
                "scope paths",
                "symbol anchors",
                "exact source reconstruction",
            ],
        },
        "evidence_boundary": "Codec round-trip and parser rejection only; Stage-A generation quality is untested.",
    }


def _walk(node):
    yield (), node
    for index, (_field, child) in enumerate(node.children):
        yield from (
            (path, descendant)
            for path, descendant in _walk(child)
        )


def _node_count(node) -> int:
    return sum(1 for _path, _descendant in _walk(node))


def _fingerprint_payload(fingerprint) -> dict[str, object]:
    return {
        "language": fingerprint.language,
        "package": fingerprint.package,
        "version": fingerprint.package_version,
        "abi_version": fingerprint.abi_version,
        "semantic_version": (
            None
            if fingerprint.semantic_version is None
            else list(fingerprint.semantic_version)
        ),
        "native_sha256": fingerprint.native_sha256,
        "grammar_sha256": fingerprint.grammar_sha256,
    }


def _metric_attack_audit() -> dict[str, object]:
    cases = [
        (
            "python",
            "def add(a, b):\n    return a + b\n",
            "def sum_values(a, b):\n    if a:\n        return a * b\n    return b\n",
        ),
        (
            "rust",
            "fn add(a: i32, b: i32) -> i32 { a + b }\n",
            "fn sum_values(a: i32, b: i32) -> i32 { if a > 0 { a * b } else { b } }\n",
        ),
        (
            "cpp",
            "int add(int a, int b) { return a + b; }\n",
            "int sum_values(int a, int b) { if (a > 0) { return a * b; } return b; }\n",
        ),
    ]
    attacks = {}
    errors = []
    for language, reference_source, prediction_source in cases:
        reference = parse_ast(reference_source, language)
        prediction = parse_ast(prediction_source, language)
        identity = ast_metrics(reference, reference)
        attacked = ast_metrics(prediction, reference)
        checks = {
            "identity_node_f1": identity["node_type_f1"] == 1.0,
            "identity_lexical_f1": identity["lexical_f1"] == 1.0,
            "identity_edge_f1": identity["typed_edge_f1"] == 1.0,
            "identity_scope_accuracy": identity["scope_accuracy"] == 1.0,
            "identity_parse_validity": identity["parse_validity"] is True,
            "identity_tree_edit_zero": identity["normalized_tree_edit_distance"] == 0.0,
            "attack_node_f1_drops": attacked["node_type_f1"] < 1.0,
            "attack_lexical_f1_drops": attacked["lexical_f1"] < 1.0,
            "attack_edge_f1_drops": attacked["typed_edge_f1"] < 1.0,
            "attack_scope_drops": attacked["scope_accuracy"] < 1.0,
            "attack_tree_edit_positive": attacked["normalized_tree_edit_distance"] > 0.0,
        }
        attacks[language] = {
            "identity": identity,
            "mutated": attacked,
            "checks": checks,
            "passed": all(checks.values()),
        }
        errors.extend(
            f"{language} metric attack failed: {name}"
            for name, passed in checks.items()
            if not passed
        )

    return {
        "version": "nlp-ast-metric-attack-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "attacks": attacks,
        "evidence_boundary": "Deterministic structural metric attacks only; no semantic-equivalence claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roundtrip-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_roundtrip_v1.json",
    )
    parser.add_argument(
        "--metric-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_ast_metric_attack_v1.json",
    )
    args = parser.parse_args()

    reports = (_roundtrip_audit(), _metric_attack_audit())
    for path, report in zip(
        (args.roundtrip_output, args.metric_output),
        reports,
        strict=True,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(report, indent=2))
    if any(report["status"] != "pass" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
