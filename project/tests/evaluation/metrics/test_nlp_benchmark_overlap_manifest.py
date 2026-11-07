import hashlib
import json
from pathlib import Path

import pytest

from newera.nlp_flow.components.research.evaluation.manifests.benchmark_overlap_manifest import audit_fingerprinted_fixture_overlap


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n")


def test_overlap_scan_binds_exact_fixture_bytes_and_detects_changes(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    _write(train, {"id": "doc", "text": "return alpha + beta", "split": "fixture", "role": "training_fixture"})
    _write(
        benchmark,
        {
            "case_id": "one",
            "field": "solution",
            "text": "return alpha + beta",
            "split": "fixture",
            "role": "benchmark_fixture",
        },
    )
    first = audit_fingerprinted_fixture_overlap(train, benchmark, width=8, threshold=0.8)
    assert first["training_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert first["benchmark_sha256"] == hashlib.sha256(benchmark.read_bytes()).hexdigest()
    assert first["matches"][0]["exact_document_ids"] == ["doc"]
    _write(train, {"id": "doc", "text": "return different", "split": "fixture", "role": "training_fixture"})
    second = audit_fingerprinted_fixture_overlap(train, benchmark, width=8, threshold=0.8)
    assert first["training_sha256"] != second["training_sha256"]
    assert second["matches"] == []


def test_implementation_scan_rejects_test_split_or_wrong_role(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    _write(train, {"id": "doc", "text": "source", "split": "train", "role": "training_fixture"})
    _write(benchmark, {"case_id": "one", "field": "test", "text": "expected", "split": "test"})
    with pytest.raises(ValueError, match="fixture rows only"):
        audit_fingerprinted_fixture_overlap(train, benchmark, width=4, threshold=0.8)
    _write(train, {"id": "doc", "text": "source", "split": "fixture", "role": "wrong"})
    _write(benchmark, {"case_id": "one", "field": "test", "text": "expected", "split": "fixture"})
    with pytest.raises(ValueError, match="invalid fixture role"):
        audit_fingerprinted_fixture_overlap(train, benchmark, width=4, threshold=0.8)


def test_test_split_is_rejected_before_decoding_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import newera.nlp_flow.components.research.evaluation.manifests.benchmark_overlap_manifest as module

    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    _write(train, {"id": "doc", "text": "source", "split": "fixture", "role": "training_fixture"})
    _write(benchmark, {"case_id": "one", "field": "test", "text": "hidden", "split": "test"})
    original_loads = json.loads
    decoded = []

    def tracking_loads(data):
        decoded.append(data)
        return original_loads(data)

    monkeypatch.setattr(module.json, "loads", tracking_loads)
    with pytest.raises(ValueError, match="fixture rows only"):
        audit_fingerprinted_fixture_overlap(train, benchmark, width=4, threshold=0.8)
    assert len(decoded) == 1
    assert b'"training_fixture"' in decoded[0]
