"""Metadata-only lineage and split audit for frozen NLP corpora."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FRAGMENT = re.compile(r"^(?P<parent>.+)-fragment-(?P<ordinal>[0-9]+)-(?P<digest>[0-9a-f]{8})$")
_SPLITS = frozenset({"train", "validation", "test"})


def _split_conflicts(rows: Sequence[Mapping[str, Any]], fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        values = tuple(str(row.get(field) or "") for field in fields)
        if all(values) and row.get("split") in _SPLITS:
            groups[values][row["split"]].append(str(row["case_id"]))
    return [
        {"key": dict(zip(fields, values, strict=True)), "case_ids_by_split": dict(sorted(by_split.items()))}
        for values, by_split in sorted(groups.items())
        if len(by_split) > 1
    ]


def audit_lineage(
    parents: Sequence[Mapping[str, Any]],
    fragments: Sequence[Mapping[str, Any]],
    registry: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Audit metadata without opening validation/test source text or predictions.

    Legacy fragment parents inferred from an ID are marked as inferred, not as
    independently verified lineage. Missing clone labels remain unknown.
    """
    source_by_id = {str(item["id"]): item for item in registry}
    if len(source_by_id) != len(registry):
        raise ValueError("duplicate source registry id")
    parent_by_id: dict[str, Mapping[str, Any]] = {}
    violations: list[dict[str, str]] = []
    coverage: Counter[tuple[str, str]] = Counter()
    missing_clone_family = 0
    for row in parents:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in parent_by_id:
            raise ValueError("missing or duplicate parent case id")
        parent_by_id[case_id] = row
        language, split = str(row.get("language") or ""), str(row.get("split") or "")
        coverage[(language, split)] += 1
        if split not in _SPLITS or language not in {"python", "rust", "cpp"}:
            violations.append({"case_id": case_id, "reason": "invalid_language_or_split"})
        source = source_by_id.get(str(row.get("source_id") or ""))
        if source is None:
            violations.append({"case_id": case_id, "reason": "unregistered_source"})
        else:
            for field in ("repository", "revision", "license"):
                if not row.get(field) or row[field] != source.get(field):
                    violations.append({"case_id": case_id, "reason": f"registry_{field}_mismatch"})
        digest = row.get("source_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            violations.append({"case_id": case_id, "reason": "invalid_source_sha256"})
        source_path = row.get("source_path")
        if (
            not isinstance(source_path, str)
            or not source_path
            or PurePosixPath(source_path).is_absolute()
            or (".." in PurePosixPath(source_path).parts)
        ):
            violations.append({"case_id": case_id, "reason": "invalid_source_path"})
        if not row.get("clone_family"):
            missing_clone_family += 1
    seen_fragments: set[str] = set()
    inferred = 0
    missing_parent = 0
    fragment_coverage: Counter[tuple[str, str]] = Counter()
    for row in fragments:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in seen_fragments:
            raise ValueError("missing or duplicate fragment case id")
        seen_fragments.add(case_id)
        fragment_coverage[(str(row.get("language") or ""), str(row.get("split") or ""))] += 1
        match = _FRAGMENT.fullmatch(case_id)
        parent_id = row.get("parent_case_id")
        if not parent_id and match:
            parent_id = match.group("parent")
            inferred += 1
        if match and parent_id and str(parent_id) != match.group("parent"):
            violations.append({"case_id": case_id, "reason": "fragment_parent_id_mismatch"})
        digest = row.get("target_source_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            violations.append({"case_id": case_id, "reason": "invalid_fragment_sha256"})
        elif match and not digest.startswith(match.group("digest")):
            violations.append({"case_id": case_id, "reason": "fragment_hash_prefix_mismatch"})
        parent = parent_by_id.get(str(parent_id)) if parent_id else None
        if parent is None:
            missing_parent += 1
        elif row.get("split") != parent.get("split") or row.get("language") != parent.get("language"):
            violations.append({"case_id": case_id, "reason": "fragment_parent_split_or_language_mismatch"})

    def rows_for(counter: Counter[tuple[str, str]]) -> list[dict[str, object]]:
        return [
            {"language": language, "split": split, "rows": count}
            for (language, split), count in sorted(counter.items())
        ]

    return {
        "parent_rows": len(parents),
        "fragment_rows": len(fragments),
        "parent_coverage": rows_for(coverage),
        "fragment_coverage": rows_for(fragment_coverage),
        "fragment_parent_id_inferred": inferred,
        "fragment_parent_missing": missing_parent,
        "parent_clone_family_missing": missing_clone_family,
        "registry_and_lineage_violations": violations,
        "repository_conflicts": _split_conflicts(parents, ("repository",)),
        "file_conflicts": _split_conflicts(parents, ("repository", "source_path")),
        "source_hash_conflicts": _split_conflicts(parents, ("source_sha256",)),
        "clone_family_conflicts": _split_conflicts(parents, ("clone_family",)),
        "evidence_boundary": (
            "Split metadata and pinned registry comparisons only; inferred fragment IDs are not independent proof, "
            "missing clone families are unknown; bytes, near duplicates and benchmark contamination are unverified."
        ),
    }
