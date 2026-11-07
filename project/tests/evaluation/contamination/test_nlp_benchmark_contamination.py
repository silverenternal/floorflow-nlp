from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from newera.nlp_flow.components.research.evaluation.benchmarks.benchmark_contamination import (
    complete_single_scan,
    index_training,
    normalize_text,
    reserve_single_scan,
    scan_benchmark,
    shingles,
    validate_external_freeze,
)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    value = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(value, encoding="utf-8")
    return hashlib.sha256(value.encode()).hexdigest()


def test_contamination_scanner_reports_exact_and_near_fields_per_document(tmp_path: Path) -> None:
    shared = "def f(x): return x + 1 " * 20
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"id": "train-1", "text": shared}])
    benchmark_hash = write_jsonl(
        benchmark,
        [
            {"id": "exact", "fields": {"prompt": shared}},
            {"id": "near", "fields": {"prompt": shared + "changed"}},
            {"id": "clean", "fields": {"prompt": "unrelated story about a bridge with enough words to scan"}},
        ],
    )

    index = index_training([train], 32)
    result = scan_benchmark(
        benchmark,
        index,
        near_threshold=0.8,
        minimum_substring_characters=16,
        expected_sha256=benchmark_hash,
        expected_rows=3,
    )

    assert index.rows == 1
    assert len(index.input_sha256[str(train)]) == 64
    assert result["exact_matches"] == ["exact"]
    assert result["substring_matches"] == ["exact"]
    assert [row["id"] for row in result["near_matches"]] == ["exact", "near"]
    assert result["status"] == "scanned"
    assert result["sha256"] == benchmark_hash


def test_exact_substring_search_is_alignment_invariant(tmp_path: Path) -> None:
    embedded = "distinctive_function_name(value): return project_specific_constant + 927"
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"text": "x" * 137 + embedded + " # suffix"}])
    write_jsonl(benchmark, [{"id": "shifted", "fields": {"prompt": embedded}}])
    result = scan_benchmark(
        benchmark,
        index_training([train], 32),
        near_threshold=0.95,
        minimum_substring_characters=32,
    )
    assert result["substring_matches"] == ["shifted"]
    assert result["match_details"][0]["substring_fields"][0]["training_document_ids"] == ["source-000-row-0"]


def test_near_shingles_cannot_be_combined_across_training_documents(tmp_path: Path) -> None:
    left = "alpha_beta_gamma_delta_epsilon_zeta_eta_theta"
    right = "iota_kappa_lambda_mu_nu_xi_omicron_pi_rho"
    query = left + "_separator_" + right
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"text": left}, {"text": right}])
    write_jsonl(benchmark, [{"id": "joined", "fields": {"prompt": query}}])
    result = scan_benchmark(
        benchmark,
        index_training([train], 8),
        near_threshold=0.8,
        minimum_substring_characters=8,
    )
    assert result["near_matches"] == []
    assert result["substring_matches"] == []


def test_field_boundaries_are_not_concatenated_and_shortness_is_reported(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"text": "prefixsuffix"}])
    write_jsonl(benchmark, [{"id": "separate", "fields": {"a": "prefix", "b": "suffix"}}])
    result = scan_benchmark(
        benchmark,
        index_training([train], 8),
        near_threshold=0.8,
        minimum_substring_characters=20,
    )
    assert result["exact_matches"] == []
    assert result["substring_matches"] == []
    assert len(result["short_fields_unassessed_for_near"]) == 2
    assert result["status"] == "partial"


def test_scan_rejects_changed_benchmark_hash_and_row_count(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    benchmark = tmp_path / "bench.jsonl"
    write_jsonl(train, [{"text": "training source string"}])
    write_jsonl(benchmark, [{"id": "one", "fields": {"prompt": "long enough benchmark field"}}])
    index = index_training([train], 8)
    with pytest.raises(ValueError, match="hash differs"):
        scan_benchmark(benchmark, index, near_threshold=0.8, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="row count"):
        scan_benchmark(benchmark, index, near_threshold=0.8, expected_rows=2)


def test_external_scan_requires_frozen_all_identity_and_dataset_hashes() -> None:
    payload = {
        "status": "frozen",
        "candidate_id": "candidate-A",
        **{field: hashlib.sha256(field.encode()).hexdigest() for field in (
            "checkpoint_sha256",
            "tokenizer_sha256",
            "training_corpus_sha256",
            "parser_suite_sha256",
            "evaluator_sha256",
            "claim_sha256",
        )},
        "benchmark_manifest_sha256": "a" * 64,
    }
    validate_external_freeze(payload, benchmark_manifest_sha256="a" * 64)
    with pytest.raises(ValueError, match="not frozen"):
        validate_external_freeze({**payload, "status": "pending"}, benchmark_manifest_sha256="a" * 64)
    with pytest.raises(ValueError, match="does not bind"):
        validate_external_freeze(payload, benchmark_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="missing identity"):
        validate_external_freeze({"status": "frozen"}, benchmark_manifest_sha256="a" * 64)


def test_single_scan_marker_prevents_reopening_after_success_or_interruption(tmp_path: Path) -> None:
    marker = tmp_path / "scan.started.json"
    reserve_single_scan(marker, freeze_sha256="f" * 64, benchmark_manifest_sha256="b" * 64)
    assert json.loads(marker.read_text())["status"] == "scan_started"
    with pytest.raises(ValueError, match="refusing to reopen"):
        reserve_single_scan(marker, freeze_sha256="f" * 64, benchmark_manifest_sha256="b" * 64)
    complete_single_scan(marker, report_sha256="r" * 64)
    finished = json.loads(marker.read_text())
    assert finished["status"] == "scan_completed"
    assert finished["report_sha256"] == "r" * 64
    with pytest.raises(ValueError, match="refusing to reopen"):
        reserve_single_scan(marker, freeze_sha256="f" * 64, benchmark_manifest_sha256="b" * 64)


def test_normalization_and_short_shingles_are_stable() -> None:
    assert normalize_text(" A\n B\tC ") == "a b c"
    assert len(shingles("short", 32)) == 1
