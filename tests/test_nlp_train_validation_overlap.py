import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.train_validation_overlap import audit_train_validation_overlap

ROOT = Path(__file__).resolve().parents[1]


def _row(case_id: str, split: str, source: str, language: str = "python") -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "language": language,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def test_overlap_reports_exact_matches_without_returning_source() -> None:
    source = "distinctive_function = alpha + beta"
    result = audit_train_validation_overlap(
        [_row("train", "train", source), _row("valid", "validation", source), _row("short", "validation", "x")],
        width=8,
        threshold=0.8,
    )
    assert result["matches"][0]["exact_document_ids"] == ["train"]
    assert result["short_validation_fields_unassessed_for_near"] == [{"case_id": "short", "field": "source"}]
    assert result["status"] == "partial"
    assert source not in json.dumps(result)
    assert result["test_content_evaluated"] is False


def test_overlap_rejects_test_content_duplicate_ids_and_tampering() -> None:
    train = _row("train", "train", "alpha")
    validation = _row("valid", "validation", "beta")
    with pytest.raises(ValueError, match="only train and validation"):
        audit_train_validation_overlap([train, validation, _row("hidden", "test", "secret")], width=4, threshold=0.8)
    with pytest.raises(ValueError, match="unique"):
        audit_train_validation_overlap([train, {**validation, "case_id": "train"}], width=4, threshold=0.8)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        audit_train_validation_overlap([train, {**validation, "source": "altered"}], width=4, threshold=0.8)


def test_short_validation_sources_receive_a_separate_smaller_shingle_scan() -> None:
    result = audit_train_validation_overlap(
        [_row("train", "train", "abcdefghijk"), _row("valid", "validation", "abcdefghijk")],
        width=16,
        threshold=0.8,
    )
    assert result["status"] == "scanned_with_short_fallback"
    assert result["short_fallback"]["width"] == 4
    assert result["short_fallback"]["matches"][0]["exact_document_ids"] == ["train"]
    assert result["short_validation_fields_unassessed_for_near"] == []


def test_alignment_shifted_validation_source_is_found_as_exact_substring() -> None:
    embedded = "def distinctive():\n    return project_specific_value + 19\n"
    train_source = "# different prefix shifts every sampled window\n" + embedded + "# suffix\n"
    result = audit_train_validation_overlap(
        [_row("train", "train", train_source), _row("valid", "validation", embedded)],
        width=32,
        threshold=0.8,
    )
    match = result["matches"][0]
    assert match["exact_document_ids"] == []
    assert match["exact_substring_document_ids"] == ["train"]
    assert embedded not in json.dumps(result)


def test_substrings_are_checked_within_documents_without_cross_file_joining() -> None:
    validation = "project_specific_token_123"
    result = audit_train_validation_overlap(
        [
            _row("left", "train", "prefix project_specific_"),
            _row("right", "train", "token_123 suffix"),
            _row("valid", "validation", validation),
        ],
        width=8,
        threshold=0.8,
    )
    assert result["matches"] == []


def test_loader_skips_test_rows_before_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "train_validation_audit", ROOT / "scripts/audit_nlp_train_validation_overlap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(_row("a", "train", "foo"), separators=(",", ":"))
        + "\n"
        + json.dumps(_row("b", "validation", "bar"))
        + "\n"
        + json.dumps(_row("c", "test", "hidden"))
        + "\n"
    )
    original_loads = json.loads
    decoded = []

    def tracking_loads(data):
        decoded.append(data)
        return original_loads(data)

    monkeypatch.setattr(module.json, "loads", tracking_loads)
    assert [row["case_id"] for row in module.load_train_validation(path)] == ["a", "b"]
    assert len(decoded) == 2
    assert all('"hidden"' not in row for row in decoded)
