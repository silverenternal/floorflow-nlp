#!/usr/bin/env python3
"""Verify pinned supplement lineage and train-only action replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph, render_source
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.platform.data.sources.multilingual_supplement import audit_multilingual_supplement
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import action_from_record
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import decode_actions, render_tree_structure

ROOT = Path(__file__).resolve().parents[5]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def verify_supplement(config: dict[str, object], *, root: Path) -> dict[str, object]:
    registry_payload = json.loads((root / str(config["registry"])).read_text(encoding="utf-8"))
    sources = {str(row["id"]): row for row in registry_payload["sources"]}
    source_roots = {
        str(row["source_id"]): root / str(config["source_root"]) / str(row["directory"])
        for row in config["validation_sources"]
    }
    train_parents: dict[str, bytes] = {}
    train_corpus = root / str(config["existing_train_ast_corpus"])
    if _sha256(train_corpus.read_bytes()) != config["expected_ast_corpus_sha256"]:
        raise ValueError("AST train corpus differs from its pinned hash")
    with train_corpus.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "train":
                payload = str(row["source"]).encode("utf-8")
                if _sha256(payload) != row["source_sha256"]:
                    raise ValueError(f"train parent source hash mismatch: {row['case_id']}")
                train_parents[str(row["case_id"])] = payload

    metadata_path = root / str(config["output_metadata"])
    metadata = _read_jsonl(metadata_path)
    fragments: dict[str, str] = {}
    checked_validation_blobs: set[tuple[str, str]] = set()
    for row in metadata:
        source = sources[str(row["source_id"])]
        if any(row[field] != source[field] for field in ("repository", "revision", "license", "clone_family")):
            raise ValueError(f"source identity mismatch: {row['case_id']}")
        if row["split"] != source["planned_split"]:
            raise ValueError(f"source split mismatch: {row['case_id']}")
        path = str(row["source_path"])
        expected_url = f"{str(source['source_url']).rstrip('/')}/blob/{row['revision']}/{quote(path, safe='/')}"
        if row["source_url"] != expected_url:
            raise ValueError(f"pinned source URL mismatch: {row['case_id']}")
        if row["split"] == "train":
            parent = train_parents.get(str(row["parent_case_id"]))
            if parent is None:
                raise ValueError(f"missing train parent: {row['case_id']}")
        else:
            repository = source_roots[str(row["source_id"])]
            parent_path = repository / path
            parent = parent_path.read_bytes()
            if (str(row["source_id"]), path) not in checked_validation_blobs:
                if _git(repository, "rev-parse", "HEAD") != row["revision"]:
                    raise ValueError(f"validation checkout pin mismatch: {row['source_id']}")
                blob = _git(repository, "rev-parse", f"HEAD:{path}")
                if blob != _git(repository, "hash-object", str(parent_path)):
                    raise ValueError(f"validation source blob mismatch: {row['source_id']}:{path}")
                checked_validation_blobs.add((str(row["source_id"]), path))
        if _sha256(parent) != row["parent_source_sha256"]:
            raise ValueError(f"parent byte hash mismatch: {row['case_id']}")
        start, end = int(row["fragment_start_byte"]), int(row["fragment_end_byte"])
        if end > len(parent):
            raise ValueError(f"fragment span exceeds parent: {row['case_id']}")
        fragment = parent[start:end].decode("utf-8").rstrip() + "\n"
        if _sha256(fragment.encode("utf-8")) != row["source_sha256"]:
            raise ValueError(f"fragment byte hash mismatch: {row['case_id']}")
        fragments[str(row["case_id"])] = fragment

    policy = json.loads((root / str(config["policy"])).read_text(encoding="utf-8"))
    admission = audit_multilingual_supplement(metadata, registry_payload["sources"], policy)
    if admission["status"] != "pass":
        raise ValueError("supplement metadata admission audit failed")

    grammar_path = root / str(config["grammars"])
    grammars = {
        language: load_ast_grammar(grammar_path, language)
        for language in ("python", "rust", "cpp")
    }
    train_actions = _read_jsonl(root / str(config["output_train_actions"]))
    metadata_by_id = {str(row["case_id"]): row for row in metadata}
    replayed = 0
    for row in train_actions:
        case_id = str(row["case_id"])
        record = metadata_by_id[case_id]
        if row["split"] != "train" or record["split"] != "train":
            raise ValueError(f"non-train action row emitted: {case_id}")
        if row["target_source_sha256"] != record["source_sha256"]:
            raise ValueError(f"action target hash mismatch: {case_id}")
        actions = [action_from_record(value) for value in row["actions"]]
        if len(actions) != int(row["action_count"]) or len(actions) != int(record["action_count"]):
            raise ValueError(f"action count mismatch: {case_id}")
        structure = decode_actions(grammars[str(row["language"])], actions)
        rendered = render_tree_structure(structure)
        if rendered != fragments[case_id]:
            raise ValueError(f"train action source replay mismatch: {case_id}")
        if render_source(parse_ast_graph(rendered, str(row["language"])).root) != rendered:
            raise ValueError(f"train action parser replay mismatch: {case_id}")
        replayed += 1

    if any(row["split"] != "train" for row in train_actions):
        raise ValueError("validation content entered the train action output")
    unassessed = sum(
        row.get("near_contamination_status") == "unassessed_below_64_characters"
        for row in metadata
    )
    return {
        "version": "nlp-multilingual-supplement-verification-v11",
        "status": "partial" if unassessed else "pass",
        "lineage_status": "pass",
        "metadata_admission_status": admission["status"],
        "training_action_replay_status": "pass",
        "metadata_rows": len(metadata),
        "validation_counts": admission["validation_counts"],
        "train_action_rows_replayed": replayed,
        "validation_action_rows_written": 0,
        "validation_source_blobs_verified": len(checked_validation_blobs),
        "near_overlap_short_rows_unassessed": unassessed,
        "test_content_evaluated": False,
        "evidence_boundary": (
            "Pinned source lineage, supplement metadata coverage, and train-only teacher-action replay; "
            "short-row near overlap and frozen external benchmark scan remain incomplete."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/data/sources/nlp_multilingual_supplement_sources.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = verify_supplement(config, root=ROOT)
    output = args.output or ROOT / str(config["output_verification_report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
