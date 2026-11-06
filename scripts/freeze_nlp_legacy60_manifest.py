#!/usr/bin/env python3
"""Fingerprint the unchanged 60-case validation suite and its evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.ast_codec import parse_ast_graph, parser_fingerprint
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.real_corpus_policy_data import action_from_record
from floorflow.nlp_flow.tree_action_decoder import decode_actions, render_tree_structure

ROOT = Path(__file__).resolve().parents[1]
_VALIDATION_LINE = re.compile(r'"split"\s*:\s*"validation"')
_LANGUAGES = ("python", "rust", "cpp")
_CODE_FILES = {
    "canonicalizer": (
        "src/floorflow/nlp_flow/ast_codec.py",
        "src/floorflow/nlp_flow/ast_codec_io.py",
        "src/floorflow/nlp_flow/ast_codec_parser.py",
    ),
    "metrics": (
        "src/floorflow/nlp_flow/ast_codec_metrics.py",
        "src/floorflow/nlp_flow/ast_quality_gate.py",
    ),
    "teacher_projection": (
        "src/floorflow/nlp_flow/ast_grammar.py",
        "src/floorflow/nlp_flow/real_corpus_policy_data.py",
        "src/floorflow/nlp_flow/tree_action_decoder.py",
    ),
    "prompt_contract": (
        "src/floorflow/nlp_flow/prompt_provenance.py",
        "configs/nlp_task_tracks_v11.json",
    ),
    "legacy_adapter": ("src/floorflow/nlp_flow/legacy60_adapter.py",),
}
_THRESHOLDS = {
    "minimum_node_type_f1": 0.95,
    "minimum_typed_edge_f1": 0.95,
    "minimum_scope_relation_f1": 0.90,
    "minimum_lexical_f1": 0.95,
    "maximum_normalized_tree_edit_distance": 0.05,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def load_validation_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if _VALIDATION_LINE.search(line) is None:
                continue
            row = json.loads(line)
            if row.get("split") != "validation":
                raise ValueError("split metadata does not match selected validation row")
            rows.append(row)
    return rows


def freeze_manifest(rows: list[dict[str, Any]], *, grammar_path: Path) -> dict[str, object]:
    if len(rows) != 60:
        raise ValueError(f"the original recovery suite must contain exactly 60 rows, found {len(rows)}")
    grammars = {language: ASTTransitionGrammar.from_file(grammar_path, language) for language in _LANGUAGES}
    cases = []
    language_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        language = row.get("language")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("validation case IDs must be nonempty and unique")
        if language not in _LANGUAGES:
            raise ValueError(f"unsupported language for frozen case {case_id}")
        seen.add(case_id)
        prompt = row.get("prompt")
        metadata = row.get("prompt_metadata")
        prompt_source = row.get("prompt_source")
        target_hash = row.get("target_source_sha256")
        actions = row.get("actions")
        if not isinstance(prompt, str) or not isinstance(metadata, dict) or not isinstance(prompt_source, str):
            raise TypeError(f"case {case_id} has incomplete prompt provenance")
        if not isinstance(target_hash, str) or len(target_hash) != 64 or not isinstance(actions, list):
            raise ValueError(f"case {case_id} has incomplete gold provenance")
        teacher_actions = [action_from_record(action) for action in actions]
        structure = decode_actions(grammars[language], teacher_actions)
        gold_source = render_tree_structure(structure)
        observed_source_hash = sha256_bytes(gold_source.encode("utf-8"))
        if observed_source_hash != target_hash:
            raise ValueError(f"teacher replay differs from frozen source hash for {case_id}")
        graph = parse_ast_graph(gold_source, language)
        parser = parser_fingerprint(language)
        if row.get("parser_grammar_sha256") != parser.grammar_sha256:
            raise ValueError(f"parser grammar fingerprint mismatch for {case_id}")
        graph_hash = sha256_bytes(serialize_ast_graph(graph).encode("utf-8"))
        cases.append(
            {
                "case_id": case_id,
                "language": language,
                "prompt_source": prompt_source,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "prompt_metadata_sha256": stable_hash(metadata),
                "gold_source_sha256": observed_source_hash,
                "gold_actions_sha256": stable_hash(actions),
                "gold_graph_sha256": graph_hash,
                "parser_grammar_sha256": parser.grammar_sha256,
            }
        )
        language_counts[language] += 1
    parser_fingerprints = {}
    for language in _LANGUAGES:
        fingerprint = parser_fingerprint(language)
        parser_fingerprints[language] = {
            "package": fingerprint.package.replace("_", "-"),
            "version": fingerprint.package_version,
            "abi_version": fingerprint.abi_version,
            "semantic_version": None if fingerprint.semantic_version is None else list(fingerprint.semantic_version),
            "native_sha256": fingerprint.native_sha256,
            "grammar_sha256": fingerprint.grammar_sha256,
        }
    code_hashes = {
        group: {name: sha256_file(ROOT / name) for name in files}
        for group, files in _CODE_FILES.items()
    }
    code_hashes["grammar_config"] = {
        "configs/nlp_ast_transition_grammar_v1.json": sha256_file(grammar_path)
    }
    code_hashes["canonical_ir_config"] = {
        "configs/nlp_canonical_ir_v11.json": sha256_file(ROOT / "configs/nlp_canonical_ir_v11.json")
    }
    code_hashes["parser_schema"] = {
        "configs/nlp_ast_schema_v1.json": sha256_file(ROOT / "configs/nlp_ast_schema_v1.json")
    }
    code_hashes["manifest_freezer"] = {
        "scripts/freeze_nlp_legacy60_manifest.py": sha256_file(ROOT / "scripts/freeze_nlp_legacy60_manifest.py")
    }
    code_hashes["manifest_verifier"] = {
        "scripts/verify_nlp_legacy60_manifest.py": sha256_file(ROOT / "scripts/verify_nlp_legacy60_manifest.py")
    }
    return {
        "version": "nlp-legacy60-evaluator-v11",
        "suite_id": "frozen_tree_action_validation_60_v1",
        "source_corpus": ".data/nlp/tree_action_v1/corpus.jsonl",
        "split": "validation",
        "case_count": len(cases),
        "ordered_case_ids": [case["case_id"] for case in cases],
        "language_counts": {language: language_counts[language] for language in _LANGUAGES},
        "cases": cases,
        "parser_fingerprints": parser_fingerprints,
        "implementation_sha256": code_hashes,
        "thresholds": _THRESHOLDS,
        "aggregate_rule": (
            "Every case must independently pass all thresholds; aggregate averages cannot compensate for a failure."
        ),
        "test_content_evaluated": False,
        "evidence_boundary": (
            "Frozen validation prompts/gold hashes and evaluator identity; no model quality is established "
            "by manifest creation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument(
        "--grammar",
        type=Path,
        default=ROOT / "configs/nlp_ast_transition_grammar_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/nlp_legacy60_eval_manifest_v11.json",
    )
    args = parser.parse_args()
    rows = load_validation_rows(args.corpus)
    manifest = freeze_manifest(rows, grammar_path=args.grammar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "frozen",
                "case_count": manifest["case_count"],
                "language_counts": manifest["language_counts"],
                "test_content_evaluated": manifest["test_content_evaluated"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
