from copy import deepcopy

import pytest

from newera.nlp_flow.components.platform.data.corpus.corpus_lineage import audit_lineage


def _parent(case_id: str, split: str, *, path: str, digest: str) -> dict:
    return {
        "case_id": case_id,
        "language": "python",
        "split": split,
        "source_id": "source",
        "source_path": path,
        "source_sha256": digest,
        "repository": "owner/repo",
        "revision": "revision",
        "license": "MIT",
    }


def test_lineage_detects_cross_split_groups_and_unverified_legacy_links() -> None:
    registry = [{"id": "source", "repository": "owner/repo", "revision": "revision", "license": "MIT"}]
    first = _parent("parent1", "train", path="file.py", digest="a" * 64)
    second = _parent("parent2", "validation", path="file.py", digest="a" * 64)
    fragments = [
        {
            "case_id": "parent1-fragment-00-bbbbbbbb",
            "language": "python",
            "split": "train",
            "target_source_sha256": "b" * 64,
        }
    ]
    report = audit_lineage([first, second], fragments, registry)
    assert len(report["repository_conflicts"]) == 1
    assert len(report["file_conflicts"]) == 1
    assert len(report["source_hash_conflicts"]) == 1
    assert report["fragment_parent_id_inferred"] == 1
    assert report["fragment_parent_missing"] == 0
    assert report["parent_clone_family_missing"] == 2
    assert "not independent proof" in report["evidence_boundary"]


def test_lineage_flags_changed_registry_fragment_and_missing_parent() -> None:
    registry = [{"id": "source", "repository": "owner/repo", "revision": "revision", "license": "MIT"}]
    parent = _parent("parent1", "train", path="../escape.py", digest="a" * 64)
    parent["revision"] = "unexpected"
    fragments = [
        {
            "case_id": "parent1-fragment-00-bbbbbbbb",
            "language": "python",
            "split": "validation",
            "target_source_sha256": "c" * 64,
        },
        {
            "case_id": "missing-fragment-00-dddddddd",
            "language": "python",
            "split": "test",
            "target_source_sha256": "d" * 64,
        },
    ]
    report = audit_lineage([parent], fragments, registry)
    reasons = {row["reason"] for row in report["registry_and_lineage_violations"]}
    assert {
        "registry_revision_mismatch",
        "invalid_source_path",
        "fragment_hash_prefix_mismatch",
        "fragment_parent_split_or_language_mismatch",
    } <= reasons
    assert report["fragment_parent_missing"] == 1
    with pytest.raises(ValueError, match="duplicate parent"):
        audit_lineage([parent, deepcopy(parent)], fragments, registry)
