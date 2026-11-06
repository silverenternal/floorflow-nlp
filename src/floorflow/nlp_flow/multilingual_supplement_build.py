"""Materialize a repo-disjoint multilingual supplement from pinned repositories."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from floorflow.nlp_flow.ast_codec import ASTNode, parse_ast_graph, render_source
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_corpus import _tree_shape
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.benchmark_contamination import index_training, scan_benchmark
from floorflow.nlp_flow.multilingual_supplement import audit_multilingual_supplement
from floorflow.nlp_flow.real_corpus_action_data import build_real_corpus_action_dataset


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _walk(node: ASTNode) -> Sequence[ASTNode]:
    return (node, *(child for _field, child_node in node.children for child in _walk(child_node)))


def _constructs(graph_root: ASTNode) -> list[str]:
    names = {node.node_type for node in _walk(graph_root)}
    values = set(names)
    if names.intersection({"integer", "float", "string", "character", "boolean", "true", "false", "null"}) or any(
        name.endswith("_literal") for name in names
    ):
        values.add("literal")
    if names.intersection({"class_specifier", "struct_specifier", "enum_specifier", "struct_item", "enum_item", "type_item"}):
        values.add("type_declaration")
    if names.intersection({"call", "call_expression", "macro_invocation"}):
        values.add("call")
    if names.intersection({"return_statement", "return_expression"}):
        values.add("return")
    return sorted(values)


def _candidate_nodes(
    root: ASTNode,
    declaration_types: set[str],
    language: str,
) -> list[ASTNode]:
    output = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.node_type in declaration_types:
            output.append(node)
        stack.extend(child for _field, child in reversed(node.children))
    priorities = {
        "cpp": {
            "class_specifier": 0,
            "struct_specifier": 0,
            "enum_specifier": 0,
            "function_definition": 1,
            "template_declaration": 2,
            "declaration": 3,
        },
        "rust": {
            "struct_item": 0,
            "enum_item": 0,
            "trait_item": 0,
            "type_item": 0,
            "function_item": 1,
            "impl_item": 2,
        },
        "python": {
            "class_definition": 0,
            "decorated_definition": 1,
            "function_definition": 2,
        },
    }
    language_priorities = priorities[language]
    return sorted(
        output,
        key=lambda node: (
            language_priorities.get(node.node_type, 5),
            node.end_byte - node.start_byte,
            node.start_byte,
            node.node_type,
        ),
    )


def _scan_train_overlap(
    fragments: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    validation_rows = [row for row in fragments if row["split"] == "validation"]
    training_paths = [root / str(value) for value in config["contamination_training_sources"]]
    scan_config = config["contamination_scan"]
    training = index_training(training_paths, int(scan_config["shingle_characters"]))
    observed_training_hashes = set(training.input_sha256.values())
    if observed_training_hashes != {str(config["expected_training_corpus_sha256"])}:
        raise ValueError("contamination training corpus hash differs from its pinned manifest")
    with tempfile.TemporaryDirectory(prefix="nlp-supplement-scan-") as directory:
        candidate_path = Path(directory) / "validation_fragments.jsonl"
        with candidate_path.open("w", encoding="utf-8") as handle:
            for row in validation_rows:
                handle.write(
                    json.dumps(
                        {"id": row["case_id"], "fields": {"source": row["_source"]}},
                        sort_keys=True,
                    )
                    + "\n"
                )
        result = scan_benchmark(
            candidate_path,
            training,
            near_threshold=float(scan_config["near_overlap_threshold"]),
            minimum_substring_characters=int(scan_config["minimum_substring_characters"]),
        )
    if any(_sha256_file(path) != training.input_sha256[str(path)] for path in training_paths):
        raise ValueError("training source changed during the supplement overlap scan")
    exclusion_reasons: dict[str, set[str]] = {}
    for reason, values in (
        ("exact_training_overlap", result["exact_matches"]),
        ("substring_training_overlap", result["substring_matches"]),
        ("near_training_overlap", [row["id"] for row in result["near_matches"]]),
    ):
        for case_id in values:
            exclusion_reasons.setdefault(str(case_id), set()).add(reason)
    return {
        "status": "partial_unassessed_short_candidates"
        if result["short_fields_unassessed_for_near"]
        else "pass_after_excluding_flagged_candidates",
        "candidate_scan_status": "fail" if any(
            result[key] for key in ("exact_matches", "substring_matches", "near_matches")
        ) else "partial" if result["short_fields_unassessed_for_near"] else "pass",
        "training_rows": training.rows,
        "training_documents": len(training.document_ids),
        "training_source_sha256": training.input_sha256,
        "validation_rows": result["rows"],
        "exact_matches": result["exact_matches"],
        "substring_matches": result["substring_matches"],
        "near_matches": result["near_matches"],
        "short_fields_unassessed_for_substring": result["short_fields_unassessed_for_substring"],
        "short_fields_unassessed_for_near": result["short_fields_unassessed_for_near"],
        "excluded_candidate_cases": [
            {"case_id": case_id, "reasons": sorted(reasons)}
            for case_id, reasons in sorted(exclusion_reasons.items())
        ],
        "unassessed_short_candidate_ids": sorted(
            {str(row["id"]) for row in result["short_fields_unassessed_for_near"]}
        ),
        "evidence_boundary": result["evidence_boundary"],
    }


def _make_fragments(
    parent: Mapping[str, object],
    *,
    source_id: str,
    repository: Path,
    repository_name: str,
    language: str,
    split: str,
    policy: Mapping[str, object],
    selection: Mapping[str, object],
    declaration_types: Mapping[str, object],
    verify_git_blobs: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    source = str(parent["source"])
    encoded = source.encode("utf-8")
    parent_digest = hashlib.sha256(encoded).hexdigest()
    if parent.get("source_sha256") != parent_digest:
        raise ValueError(f"parent source hash mismatch: {parent.get('case_id')}")
    graph = parse_ast_graph(source, language)
    if render_source(graph.root) != source:
        raise ValueError(f"parent source replay mismatch: {parent.get('case_id')}")
    node_count, _edges, depth = _tree_shape(graph.root)
    if node_count > int(selection["max_ast_nodes"]) or depth > int(selection["max_depth"]):
        return [], {"parent_ast_too_large": 1}
    if verify_git_blobs:
        relative = str(parent["source_path"])
        blob = _git(repository, "rev-parse", f"HEAD:{relative}")
        actual = _git(repository, "hash-object", str(repository / relative))
        if blob != actual:
            raise ValueError(f"pinned Git blob mismatch: {repository_name}:{relative}")

    result = []
    rejected: dict[str, int] = {}
    seen: set[str] = set()
    fragment_limit = int(selection["max_fragments_per_source"])
    candidates = _candidate_nodes(graph.root, set(declaration_types[language]), language)
    for index, node in enumerate(candidates):
        if len(result) >= fragment_limit:
            break
        if node.end_byte - node.start_byte > int(selection["max_fragment_bytes"]):
            rejected["fragment_bytes"] = rejected.get("fragment_bytes", 0) + 1
            continue
        fragment_start = node.start_byte
        fragment_end = node.end_byte
        if node.node_type in {"class_specifier", "struct_specifier", "enum_specifier"}:
            if encoded[node.end_byte : node.end_byte + 1] != b";":
                rejected["missing_type_declaration_terminator"] = (
                    rejected.get("missing_type_declaration_terminator", 0) + 1
                )
                continue
            fragment_end += 1
        fragment = encoded[fragment_start:fragment_end].decode("utf-8").rstrip() + "\n"
        digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
        if digest in seen:
            rejected["duplicate_fragment"] = rejected.get("duplicate_fragment", 0) + 1
            continue
        seen.add(digest)
        try:
            fragment_graph = parse_ast_graph(fragment, language)
        except (TypeError, ValueError, UnicodeError):
            rejected["fragment_parse"] = rejected.get("fragment_parse", 0) + 1
            continue
        nodes, _fragment_edges, fragment_depth = _tree_shape(fragment_graph.root)
        if nodes > int(selection["max_fragment_nodes"]):
            rejected["fragment_ast_nodes"] = rejected.get("fragment_ast_nodes", 0) + 1
            continue
        source_path = PurePosixPath(str(parent["source_path"]))
        source_url = (
            f"{str(parent['source_url']).rstrip('/')}/blob/{parent['revision']}/"
            f"{quote(source_path.as_posix(), safe='/')}"
        )
        result.append(
            {
                "case_id": f"supp-{language}-{digest[:20]}",
                "language": language,
                "split": split,
                "source_id": source_id,
                "repository": repository_name,
                "revision": parent["revision"],
                "license": parent["license"],
                "source_path": source_path.as_posix(),
                "source_url": source_url,
                "source_sha256": digest,
                "parent_source_sha256": parent_digest,
                "parent_case_id": str(parent["case_id"]),
                "fragment_start_byte": fragment_start,
                "fragment_end_byte": fragment_end,
                "clone_family": parent["clone_family"],
                "constructs": _constructs(fragment_graph.root),
                "ast_node_count": nodes,
                "ast_depth": fragment_depth,
                "action_count": 1,
                "_source": fragment,
                "_ast": serialize_ast_graph(fragment_graph),
            }
        )
    return result, rejected


def build_multilingual_supplement(
    config: Mapping[str, object],
    *,
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build metadata and train actions; validation targets never leave this call."""
    registry_payload = json.loads((root / str(config["registry"])).read_text())
    policy = json.loads((root / str(config["policy"])).read_text())
    sources = {str(row["id"]): row for row in registry_payload["sources"]}
    ast_path = root / str(config["existing_train_ast_corpus"])
    if _sha256_file(ast_path) != str(config["expected_ast_corpus_sha256"]):
        raise ValueError("existing AST train corpus hash differs from its pinned manifest")
    source_root = root / str(config["source_root"])
    selection = config["selection"]
    prompt_config = json.loads((root / str(config["prompt_contract"])).read_text())
    prompt_policy = prompt_config["prompt"]
    languages = ("python", "rust", "cpp")
    fragments: list[dict[str, object]] = []
    rejected: dict[str, int] = {}

    with ast_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            source_id = str(row["source_id"])
            registry_row = sources.get(source_id)
            if registry_row is None or registry_row.get("planned_split") != "train":
                continue
            for key in ("repository", "revision", "license"):
                if row.get(key) != registry_row.get(key):
                    raise ValueError(f"training registry mismatch at AST line {line_number}: {key}")
            parent = dict(row)
            parent["clone_family"] = registry_row["clone_family"]
            local_directory = PurePosixPath(str(registry_row["corpus_directory"]))
            source_root_relative = PurePosixPath(str(row["source_path"])).relative_to(
                PurePosixPath(".data/nlp/mixed_v1/code")
            )
            upstream_path = source_root_relative.relative_to(local_directory)
            parent["source_path"] = upstream_path.as_posix()
            parent["source_url"] = registry_row["source_url"]
            made, rejects = _make_fragments(
                parent,
                source_id=source_id,
                repository=source_root / source_id,
                repository_name=str(registry_row["repository"]),
                language=str(row["language"]),
                split="train",
                policy=config,
                selection=selection,
                declaration_types=config["declaration_types"],
                verify_git_blobs=False,
            )
            fragments.extend(made)
            for reason, count in rejects.items():
                rejected[reason] = rejected.get(reason, 0) + count

    for source in config["validation_sources"]:
        source_id = str(source["source_id"])
        registry_row = sources[source_id]
        if registry_row.get("planned_split") != "validation":
            raise ValueError(f"validation source is not held out: {source_id}")
        repository = source_root / str(source["directory"])
        expected_revision = str(registry_row["revision"])
        actual_revision = _git(repository, "rev-parse", "HEAD")
        if actual_revision != expected_revision:
            raise ValueError(f"repository revision mismatch for {source_id}: {actual_revision}")
        language = str(source["language"])
        for include in source["include_directories"]:
            base = repository / str(include)
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix not in source["extensions"]:
                    continue
                if set(path.parts).intersection(selection["excluded_path_parts"]):
                    continue
                if any(marker in path.name for marker in selection.get("excluded_file_markers", [])):
                    continue
                raw = path.read_bytes()
                if not int(selection["min_source_bytes"]) <= len(raw) <= int(selection["max_source_bytes"]):
                    rejected["validation_source_size"] = rejected.get("validation_source_size", 0) + 1
                    continue
                try:
                    text = raw.decode("utf-8")
                    graph = parse_ast_graph(text, language)
                    if render_source(graph.root) != text:
                        raise ValueError("lossy_render")
                    nodes, _edges, depth = _tree_shape(graph.root)
                    if nodes < 8 or nodes > int(selection["max_ast_nodes"]) or depth > int(selection["max_depth"]):
                        rejected["validation_ast_size"] = rejected.get("validation_ast_size", 0) + 1
                        continue
                    repository_path = path.relative_to(repository).as_posix()
                    commit_blob = _git(repository, "rev-parse", f"HEAD:{repository_path}")
                    actual_blob = _git(repository, "hash-object", str(path))
                    if commit_blob != actual_blob:
                        raise ValueError(f"pinned Git blob mismatch: {source_id}:{repository_path}")
                    parent = {
                        "case_id": f"parent-{language}-{hashlib.sha256(raw).hexdigest()[:16]}",
                        "source": text,
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_path": repository_path,
                        "source_url": registry_row["source_url"],
                        "revision": expected_revision,
                        "license": registry_row["license"],
                        "clone_family": registry_row["clone_family"],
                    }
                    made, rejects = _make_fragments(
                        parent,
                        source_id=source_id,
                        repository=repository,
                        repository_name=str(registry_row["repository"]),
                        language=language,
                        split="validation",
                        policy=config,
                        selection=selection,
                        declaration_types=config["declaration_types"],
                        verify_git_blobs=False,
                    )
                    fragments.extend(made)
                    for reason, count in rejects.items():
                        rejected[reason] = rejected.get(reason, 0) + count
                except (UnicodeError, ValueError) as error:
                    key = f"validation_parse_or_contract:{str(error).splitlines()[0][:80]}"
                    rejected[key] = rejected.get(key, 0) + 1

    unique_fragments = []
    seen_fragment_hashes: set[str] = set()
    for fragment in fragments:
        digest = str(fragment["source_sha256"])
        if digest in seen_fragment_hashes:
            rejected["duplicate_fragment_across_sources_or_splits"] = (
                rejected.get("duplicate_fragment_across_sources_or_splits", 0) + 1
            )
            continue
        seen_fragment_hashes.add(digest)
        unique_fragments.append(fragment)
    fragments = unique_fragments

    contamination = _scan_train_overlap(fragments, config=config, root=root)
    candidates_by_id = {str(row["case_id"]): row for row in fragments}
    contamination["excluded_candidate_cases"] = [
        {
            **excluded,
            **{
                key: candidates_by_id[str(excluded["case_id"])][key]
                for key in ("source_id", "repository", "source_path", "source_url")
            },
        }
        for excluded in contamination["excluded_candidate_cases"]
    ]
    contamination["unassessed_short_candidate_cases"] = [
        {
            "case_id": case_id,
            "reason": "near_overlap_unassessed_under_64_characters",
            **{
                key: candidates_by_id[case_id][key]
                for key in ("source_id", "repository", "source_path", "source_url")
            },
        }
        for case_id in contamination["unassessed_short_candidate_ids"]
    ]
    excluded_by_id = {
        str(row["case_id"])
        for row in contamination["excluded_candidate_cases"]
    }
    fragments = [row for row in fragments if str(row["case_id"]) not in excluded_by_id]
    unassessed_short_ids = set(contamination["unassessed_short_candidate_ids"])
    fragment_inputs = [
        {
            "case_id": row["case_id"],
            "language": row["language"],
            "split": row["split"],
            "source": row["_source"],
            "source_sha256": row["source_sha256"],
            "ast": row["_ast"],
        }
        for row in fragments
    ]
    grammar = {
        language: load_ast_grammar(root / str(config["grammars"]), language)
        for language in languages
    }
    action_rows, action_report = build_real_corpus_action_dataset(
        fragment_inputs,
        grammar,
        prompt_policy,
    )
    action_counts = {str(row.case_id): row.action_count for row in action_rows}
    metadata = []
    for row in fragments:
        if row["case_id"] not in action_counts:
            continue
        row["action_count"] = action_counts[str(row["case_id"])]
        if str(row["case_id"]) in unassessed_short_ids:
            row["near_contamination_status"] = "unassessed_below_64_characters"
        metadata.append({key: value for key, value in row.items() if not key.startswith("_")})
    audit = audit_multilingual_supplement(metadata, registry_payload["sources"], policy)
    fragment_by_id = {str(row["case_id"]): row for row in fragments}
    action_failures = []
    for failure in action_report["failures"]:
        case = fragment_by_id.get(str(failure["case_id"]), {})
        action_failures.append(
            {
                "case_id": failure["case_id"],
                "split": case.get("split", "unknown"),
                "source_id": case.get("source_id", "unknown"),
                "repository": case.get("repository", "unknown"),
                "source_path": case.get("source_path", "unknown"),
                "error": failure["error"],
            }
        )
    train_action_failures = [item for item in action_failures if item["split"] == "train"]
    train_action_gate = not train_action_failures and not action_report["prompt_leakage"]
    accepted_status = (
        "partial_unassessed_short_candidates"
        if audit["status"] == "pass"
        and contamination["status"] == "partial_unassessed_short_candidates"
        and train_action_gate
        else
        "pass_with_explicit_rejections"
        if audit["status"] == "pass"
        and contamination["status"] == "pass"
        and train_action_gate
        and action_failures
        else "pass"
        if audit["status"] == "pass"
        and contamination["status"] == "pass"
        and train_action_gate
        else "blocked_or_failed"
    )
    report = {
        "version": "nlp-multilingual-supplement-build-v11",
        "status": accepted_status,
        "source_revisions_verified": {
            str(row["source_id"]): str(sources[str(row["source_id"])]["revision"])
            for row in config["validation_sources"]
        },
        "metadata_rows": len(metadata),
        "train_action_rows": sum(row.split == "train" for row in action_rows),
        "validation_action_rows_written": 0,
        "audit": audit,
        "training_contamination_scan": contamination,
        "action_generation": action_report,
        "action_rejections": action_failures,
        "train_action_contract": "pass" if train_action_gate else "fail",
        "rejected": rejected,
        "test_content_evaluated": False,
        "evidence_boundary": (
            "Pinned source parsing, fragment replay, prompt non-leakage and metadata coverage; no benchmark scan, "
            "model training or capability claim."
        ),
    }
    return metadata, [row.as_dict() for row in action_rows if row.split == "train"], report


__all__ = ["build_multilingual_supplement"]
