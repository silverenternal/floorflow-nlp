import pytest

from newera.nlp_flow.components.research.evaluation.benchmarks.benchmark_field_overlap import scan_field_overlap


def test_scans_fields_separately_and_short_fields_remain_unassessed() -> None:
    result = scan_field_overlap(
        [{"id": "train", "text": "return alpha + beta"}],
        [
            {"case_id": "case", "field": "solution", "text": "return alpha + beta"},
            {"case_id": "case", "field": "test", "text": "x"},
        ],
        width=8,
        threshold=0.8,
    )
    assert result["matches"][0]["field"] == "solution"
    assert result["matches"][0]["exact_document_ids"] == ["train"]
    assert result["short_fields_unassessed_for_near"] == [{"case_id": "case", "field": "test"}]
    assert result["status"] == "partial"


def test_shingle_overlap_is_not_aggregated_across_training_documents() -> None:
    result = scan_field_overlap(
        [{"id": "left", "text": "abcdefghijkl"}, {"id": "right", "text": "ijklmnopqrst"}],
        [{"case_id": "case", "field": "prompt", "text": "abcdefghijklmnopqrst"}],
        width=4,
        threshold=0.9,
    )
    assert result["matches"] == []


def test_rejects_duplicate_ids_and_empty_scans_are_partial() -> None:
    with pytest.raises(ValueError, match="distinct"):
        scan_field_overlap([{"id": "x", "text": "a"}, {"id": "x", "text": "b"}], [], width=4, threshold=0.8)
    assert scan_field_overlap([], [], width=4, threshold=0.8)["status"] == "partial"
