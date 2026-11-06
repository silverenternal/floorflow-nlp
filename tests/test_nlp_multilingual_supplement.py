import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.multilingual_supplement import audit_multilingual_supplement

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "configs/nlp_multilingual_supplement_v11.json").read_text())


def _registry() -> list[dict]:
    rows = []
    for language in ("python", "rust", "cpp"):
        for split in ("train", "validation"):
            rows.append(
                {
                    "id": f"{language}_{split}",
                    "repository": f"org/{language}-{split}",
                    "revision": hashlib.sha256(f"revision-{language}-{split}".encode()).hexdigest()[:40],
                    "license": "MIT",
                    "license_url": "https://example.org/license",
                    "source_url": f"https://github.com/org/{language}-{split}",
                    "clone_family": f"repo:{language}-{split}",
                    "clone_family_evidence_url": f"https://github.com/org/{language}-{split}/network/members",
                    "planned_split": split,
                    "usage": "training",
                    "redistribution": "permitted_with_license",
                }
            )
    return rows


def _cases() -> list[dict]:
    rows = []
    registry = {row["id"]: row for row in _registry()}
    for language in ("python", "rust", "cpp"):
        constructs = POLICY["required_constructs_by_language"][language]
        for split in ("train", "validation"):
            source_id = f"{language}_{split}"
            source = registry[source_id]
            count = 20 if split == "validation" else 1
            for index in range(count):
                case_id = f"{language}-{split}-{index}"
                rows.append(
                    {
                        "case_id": case_id,
                        "language": language,
                        "split": split,
                        "source_id": source_id,
                        "repository": source["repository"],
                        "revision": source["revision"],
                        "license": source["license"],
                        "source_path": f"src/{case_id}.txt",
                        "source_url": f"{source['source_url']}/blob/{source['revision']}/src/{case_id}.txt",
                        "source_sha256": hashlib.sha256(case_id.encode()).hexdigest(),
                        "parent_source_sha256": hashlib.sha256(f"parent-{case_id}".encode()).hexdigest(),
                        "parent_case_id": f"parent-{case_id}",
                        "fragment_start_byte": 0,
                        "fragment_end_byte": 12,
                        "clone_family": source["clone_family"],
                        "constructs": [constructs[index % len(constructs)]],
                        "ast_node_count": (10, 80, 200)[index % 3],
                        "ast_depth": (3, 8, 12)[index % 3],
                        "action_count": (32, 256, 768)[index % 3],
                    }
                )
    return rows


def test_metadata_contract_accepts_source_disjoint_predeclared_coverage() -> None:
    result = audit_multilingual_supplement(_cases(), _registry(), POLICY)
    assert result["status"] == "pass"
    assert result["validation_counts"] == {"python": 20, "rust": 20, "cpp": 20}
    assert result["test_content_evaluated"] is False
    assert result["failures"] == []


def test_repository_clone_family_hash_and_file_must_be_split_disjoint() -> None:
    rows = _cases()
    train = next(row for row in rows if row["case_id"] == "python-train-0")
    validation = next(row for row in rows if row["case_id"] == "python-validation-0")
    validation.update(
        repository=train["repository"],
        source_path=train["source_path"],
        clone_family=train["clone_family"],
        source_sha256=train["source_sha256"],
    )
    result = audit_multilingual_supplement(rows, _registry(), POLICY)
    reasons = {row["reason"] for row in result["failures"]}
    assert "repository_cross_split_conflict" in reasons
    assert "file_cross_split_conflict" in reasons
    assert "clone_family_cross_split_conflict" in reasons
    assert "source_sha256_cross_split_conflict" in reasons


def test_missing_construct_and_short_validation_are_explicit() -> None:
    rows = [row for row in _cases() if row["case_id"] != "cpp-validation-19"]
    for row in rows:
        if row["language"] == "cpp" and row["split"] == "validation":
            row["constructs"] = ["function_definition"]
            row["ast_node_count"] = 20
            row["ast_depth"] = 3
            row["action_count"] = 32
    result = audit_multilingual_supplement(rows, _registry(), POLICY)
    reasons = [row["reason"] for row in result["failures"]]
    assert "validation_below_minimum:19/20" in reasons
    assert "validation_construct_missing:declaration" in reasons
    assert any(reason.startswith("validation_coverage_missing:") for reason in reasons)


def test_registry_revision_license_and_path_are_verified() -> None:
    rows = _cases()
    row = next(item for item in rows if item["case_id"] == "rust-validation-0")
    row.update(revision="other", license="GPL", source_path="../outside.rs", source_sha256="bad")
    result = audit_multilingual_supplement(rows, _registry(), POLICY)
    reasons = {entry["reason"] for entry in result["failures"]}
    assert "source_revision_mismatch" in reasons
    assert "source_license_mismatch" in reasons
    assert "invalid_source_path" in reasons
    assert "invalid_source_sha256" in reasons


def test_registry_requires_full_commit_and_pinned_fragment_url() -> None:
    rows = _cases()
    registry = _registry()
    row = next(item for item in rows if item["case_id"] == "cpp-validation-0")
    source = next(item for item in registry if item["id"] == "cpp_validation")
    source["revision"] = "main"
    source["clone_family"] = ""
    result = audit_multilingual_supplement(rows, registry, POLICY)
    reasons = {entry["reason"] for entry in result["failures"]}
    assert "source_revision_not_immutable_commit" in reasons
    assert "source_registry_missing_clone_family" in reasons
    assert "source_file_url_mismatch" in reasons
    assert row["split"] == source["planned_split"]


def test_metadata_loader_rejects_source_target_and_test_content(tmp_path: Path) -> None:
    script = ROOT / "scripts/audit_nlp_multilingual_supplement.py"
    spec = importlib.util.spec_from_file_location("supplement_audit", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "metadata.jsonl"
    path.write_text(json.dumps({"case_id": "x", "source": "hidden"}) + "\n")
    with pytest.raises(ValueError, match="content fields"):
        module.load_metadata(path)
