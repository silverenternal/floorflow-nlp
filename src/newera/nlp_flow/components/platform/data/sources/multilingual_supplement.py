"""Metadata gate for a source-disjoint multilingual corpus supplement."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_LANGUAGES = ("python", "rust", "cpp")
_SPLITS = frozenset({"train", "validation", "test"})
_FIELDS = (
    "case_id",
    "language",
    "split",
    "source_id",
    "repository",
    "revision",
    "license",
    "source_path",
    "source_url",
    "source_sha256",
    "parent_source_sha256",
    "parent_case_id",
    "fragment_start_byte",
    "fragment_end_byte",
    "clone_family",
    "constructs",
    "ast_node_count",
    "ast_depth",
    "action_count",
)


def _source_mismatches(case: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]]) -> list[str]:
    source = registry.get(str(case.get("source_id") or ""))
    if source is None:
        return ["unregistered_source"]
    failures = []
    for case_field, registry_field in (
        ("repository", "repository"),
        ("revision", "revision"),
        ("license", "license"),
        ("clone_family", "clone_family"),
        ("split", "planned_split"),
    ):
        expected = source.get(registry_field)
        if not isinstance(expected, str) or not expected.strip():
            failures.append(f"source_registry_missing_{registry_field}")
        elif case.get(case_field) != expected:
            failures.append(f"source_{registry_field}_mismatch")
    revision = source.get("revision")
    if not isinstance(revision, str) or not _GIT_COMMIT.fullmatch(revision):
        failures.append("source_revision_not_immutable_commit")
    for field in ("source_url", "license_url", "clone_family_evidence_url"):
        parsed = urlparse(str(source.get(field) or ""))
        if parsed.scheme != "https" or not parsed.netloc:
            failures.append(f"source_registry_invalid_{field}")
    source_url = source.get("source_url")
    source_path = case.get("source_path")
    if isinstance(source_url, str) and isinstance(source_path, str) and revision:
        expected_file_url = (
            f"{source_url.rstrip('/')}/blob/{revision}/{quote(source_path, safe='/')}"
        )
        if case.get("source_url") != expected_file_url:
            failures.append("source_file_url_mismatch")
    if source.get("redistribution") not in {
        "permitted_with_license",
        "do_not_redistribute_source",
    }:
        failures.append("source_registry_invalid_redistribution")
    source_language = source.get("language")
    if source_language is not None and source_language != case.get("language"):
        failures.append("source_language_mismatch")
    if source.get("usage") not in {"training", "training_conditional"}:
        failures.append("source_not_approved_for_training")
    if source.get("usage") == "training_conditional" and source.get("redistribution") != "do_not_redistribute_source":
        failures.append("conditional_source_redistribution_mismatch")
    return failures


def audit_multilingual_supplement(
    cases: Sequence[Mapping[str, Any]],
    source_registry: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, object]:
    """Validate metadata, provenance, split isolation and predeclared coverage.

    The audit consumes metadata only; it never opens source, target, test, or
    benchmark fields. Clone families must be established upstream.
    """
    source_by_id = {str(row.get("id") or ""): row for row in source_registry}
    if len(source_by_id) != len(source_registry) or "" in source_by_id:
        raise ValueError("source registry IDs must be nonempty and unique")
    failures: list[dict[str, str]] = []
    ids: set[str] = set()
    repository_splits: dict[str, set[str]] = defaultdict(set)
    files_by_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    clone_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    coverage: Counter[tuple[str, str, str]] = Counter()
    construct_coverage: Counter[tuple[str, str, str]] = Counter()
    valid_rows: list[Mapping[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in ids:
            failures.append({"case_id": case_id or f"row-{index}", "reason": "missing_or_duplicate_case_id"})
            continue
        ids.add(case_id)
        missing = [field for field in _FIELDS if field not in case]
        if missing:
            failures.append({"case_id": case_id, "reason": "missing_fields:" + ",".join(missing)})
            continue
        language, split = case["language"], case["split"]
        if language not in _LANGUAGES or split not in _SPLITS:
            failures.append({"case_id": case_id, "reason": "invalid_language_or_split"})
            continue
        for mismatch in _source_mismatches(case, source_by_id):
            failures.append({"case_id": case_id, "reason": mismatch})
        path = PurePosixPath(str(case["source_path"]))
        if not case["source_path"] or path.is_absolute() or ".." in path.parts:
            failures.append({"case_id": case_id, "reason": "invalid_source_path"})
        if not _SHA256.fullmatch(str(case["source_sha256"])):
            failures.append({"case_id": case_id, "reason": "invalid_source_sha256"})
        if not _SHA256.fullmatch(str(case["parent_source_sha256"])):
            failures.append({"case_id": case_id, "reason": "invalid_parent_source_sha256"})
        if not isinstance(case["parent_case_id"], str) or not case["parent_case_id"].strip():
            failures.append({"case_id": case_id, "reason": "invalid_parent_case_id"})
        start, end = case["fragment_start_byte"], case["fragment_end_byte"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
        ):
            failures.append({"case_id": case_id, "reason": "invalid_fragment_byte_span"})
        if not isinstance(case["clone_family"], str) or not case["clone_family"].strip():
            failures.append({"case_id": case_id, "reason": "missing_clone_family"})
        constructs = case["constructs"]
        if (
            not isinstance(constructs, list)
            or not constructs
            or any(not isinstance(value, str) or not value for value in constructs)
        ):
            failures.append({"case_id": case_id, "reason": "invalid_constructs"})
        numbers = (case["ast_node_count"], case["ast_depth"], case["action_count"])
        if any(not isinstance(value, int) or value < 1 for value in numbers):
            failures.append({"case_id": case_id, "reason": "invalid_complexity_or_action_count"})
            continue
        nodes, depth, action_count = numbers
        if nodes <= 40 and depth <= 5:
            complexity = "basic"
        elif nodes <= 150 and depth <= 10:
            complexity = "intermediate"
        else:
            complexity = "advanced"
        action_bin = "short" if action_count <= 128 else "medium" if action_count <= 512 else "long"
        repository_splits[str(case["repository"])].add(split)
        files_by_split[(str(case["repository"]), str(case["source_path"]))].add(split)
        clone_splits[str(case["clone_family"])].add(split)
        hash_splits[str(case["source_sha256"])].add(split)
        coverage[(str(language), split, complexity)] += 1
        coverage[(str(language), split, f"action_{action_bin}")] += 1
        if isinstance(constructs, list):
            for construct in set(constructs):
                if isinstance(construct, str) and construct:
                    construct_coverage[(str(language), split, construct)] += 1
        valid_rows.append(case)

    for group_splits, label in (
        (repository_splits, "repository"),
        (clone_splits, "clone_family"),
        (hash_splits, "source_sha256"),
    ):
        for group, splits in group_splits.items():
            if len(splits) > 1:
                failures.append({"case_id": group, "reason": f"{label}_cross_split_conflict"})
    for (repository, path), splits in files_by_split.items():
        if len(splits) > 1:
            failures.append({"case_id": f"{repository}:{path}", "reason": "file_cross_split_conflict"})

    minimum = int(policy.get("minimum_validation_cases_per_language", 20))
    required_constructs = policy.get("required_constructs_by_language", {})
    required_complexity = policy.get("required_complexity_bins", [])
    required_action = policy.get("required_action_length_bins", [])
    counts = Counter((str(row.get("language")), str(row.get("split"))) for row in valid_rows)
    for language in _LANGUAGES:
        if counts[(language, "validation")] < minimum:
            failures.append(
                {
                    "case_id": language,
                    "reason": f"validation_below_minimum:{counts[(language, 'validation')]}/{minimum}",
                }
            )
        expected_constructs = required_constructs.get(language, []) if isinstance(required_constructs, Mapping) else []
        for construct in expected_constructs:
            if construct_coverage[(language, "validation", str(construct))] == 0:
                failures.append({"case_id": language, "reason": f"validation_construct_missing:{construct}"})
        for category in (*required_complexity, *(f"action_{value}" for value in required_action)):
            if coverage[(language, "validation", str(category))] == 0:
                failures.append({"case_id": language, "reason": f"validation_coverage_missing:{category}"})
    return {
        "version": "nlp-multilingual-supplement-audit-v11",
        "status": "pass" if valid_rows and not failures else "blocked_or_failed",
        "metadata_rows": len(cases),
        "validated_rows": len(valid_rows),
        "validation_counts": {language: counts[(language, "validation")] for language in _LANGUAGES},
        "language_split_complexity_action_coverage": [
            {"language": language, "split": split, "category": category, "rows": count}
            for (language, split, category), count in sorted(coverage.items())
        ],
        "language_split_construct_coverage": [
            {"language": language, "split": split, "construct": construct, "rows": count}
            for (language, split, construct), count in sorted(construct_coverage.items())
        ],
        "failures": failures,
        "test_content_evaluated": False,
        "evidence_boundary": (
            "Metadata lineage and predeclared coverage only; source bytes, parser replay and benchmark "
            "contamination require separate fingerprinted audits."
        ),
    }


__all__ = ["audit_multilingual_supplement"]
