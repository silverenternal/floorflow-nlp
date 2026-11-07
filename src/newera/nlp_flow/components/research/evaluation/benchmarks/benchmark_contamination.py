"""Per-document benchmark overlap scans with input identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256 = frozenset("0123456789abcdef")
_FREEZE_FIELDS = (
    "candidate_id",
    "checkpoint_sha256",
    "tokenizer_sha256",
    "training_corpus_sha256",
    "parser_suite_sha256",
    "evaluator_sha256",
    "claim_sha256",
    "benchmark_manifest_sha256",
)


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shingles_from_normalized(value: str, width: int) -> set[str]:
    if not value:
        return set()
    if len(value) <= width:
        return {digest(value)}
    stride = max(1, width // 4)
    return {digest(value[index : index + width]) for index in range(0, len(value) - width + 1, stride)}


def _shingle_key(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest(), "big")


def shingles(value: str, width: int) -> set[str]:
    """Hash deterministic character shingles from normalized text."""
    if width < 1:
        raise ValueError("shingle width must be positive")
    return _shingles_from_normalized(normalize_text(value), width)


@dataclass(frozen=True)
class TrainingCorpusIndex:
    document_ids: list[str]
    normalized_documents: list[str]
    exact_documents: dict[str, list[int]]
    shingle_documents: dict[int, list[int]]
    document_shingle_counts: list[int]
    rows: int
    input_sha256: dict[str, str]
    shingle_width: int


def index_training(paths: list[Path], width: int) -> TrainingCorpusIndex:
    """Index every training text separately and fingerprint source files."""
    if not paths or width < 1:
        raise ValueError("training paths and a positive shingle width are required")
    document_ids: list[str] = []
    normalized_documents: list[str] = []
    exact_documents: dict[str, list[int]] = defaultdict(list)
    shingle_documents: dict[int, list[int]] = defaultdict(list)
    shingle_counts: list[int] = []
    input_hashes: dict[str, str] = {}
    rows = 0
    for file_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        row_number = 0
        source_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for raw_line in handle:
                source_digest.update(raw_line)
                payload = json.loads(raw_line.decode("utf-8"))
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"training row {path.name}:{row_number + 1} has no text")
                normalized = normalize_text(text)
                if not normalized:
                    raise ValueError(f"training row {path.name}:{row_number + 1} normalizes to empty")
                document_id = f"source-{file_index:03d}-row-{row_number}"
                document_index = len(document_ids)
                document_ids.append(document_id)
                normalized_documents.append(normalized)
                exact_documents[digest(normalized)].append(document_index)
                pieces = _shingles_from_normalized(normalized, width)
                shingle_counts.append(len(pieces))
                for piece in pieces:
                    shingle_documents[_shingle_key(piece)].append(document_index)
                rows += 1
                row_number += 1
        input_hashes[str(path)] = source_digest.hexdigest()
    return TrainingCorpusIndex(
        document_ids=document_ids,
        normalized_documents=normalized_documents,
        exact_documents=dict(exact_documents),
        shingle_documents=dict(shingle_documents),
        document_shingle_counts=shingle_counts,
        rows=rows,
        input_sha256=input_hashes,
        shingle_width=width,
    )


def _field_texts(value: Any, path: str = "fields") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [
            item
            for key, nested in value.items()
            for item in _field_texts(nested, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [item for index, nested in enumerate(value) for item in _field_texts(nested, f"{path}[{index}]")]
    return []


def scan_benchmark(
    path: Path,
    training: TrainingCorpusIndex,
    *,
    near_threshold: float,
    minimum_substring_characters: int = 64,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Scan fields independently against individual train files, reading input once."""
    if not 0 < near_threshold <= 1 or minimum_substring_characters < 1:
        raise ValueError("overlap threshold and minimum substring length are invalid")
    exact_rows: set[str] = set()
    substring_rows: set[str] = set()
    near_rows: dict[str, float] = {}
    short_substrings = []
    short_near = []
    details = []
    source_digest = hashlib.sha256()
    row_count = 0
    field_count = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            source_digest.update(raw_line)
            payload = json.loads(raw_line.decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), (str, int)):
                raise TypeError(f"invalid benchmark record in {path.name}")
            row_id = str(payload["id"])
            field_payload = payload.get("fields")
            if field_payload is None:
                field_payload = {key: value for key, value in payload.items() if key != "id"}
            fields = _field_texts(field_payload)
            if not fields:
                raise ValueError(f"benchmark row {row_id} contains no text fields")
            row_detail = {"id": row_id, "exact_fields": [], "substring_fields": [], "near_fields": []}
            for field_path, raw_text in fields:
                normalized = normalize_text(raw_text)
                if not normalized:
                    continue
                field_count += 1
                exact_ids: set[str] = set()
                if digest(normalized) in training.exact_documents:
                    exact_ids = training.exact_documents[digest(normalized)]
                    row_detail["exact_fields"].append(
                        {
                            "field": field_path,
                            "training_document_ids": sorted(training.document_ids[index] for index in exact_ids),
                        }
                    )
                    exact_rows.add(row_id)
                if len(normalized) < minimum_substring_characters:
                    short_substrings.append({"id": row_id, "field": field_path, "characters": len(normalized)})
                else:
                    substring_ids = [
                        training.document_ids[index]
                        for index, source in enumerate(training.normalized_documents)
                        if len(source) >= len(normalized) and normalized in source
                    ]
                    if substring_ids:
                        row_detail["substring_fields"].append(
                            {"field": field_path, "training_document_ids": sorted(substring_ids)}
                        )
                        substring_rows.add(row_id)
                pieces = _shingles_from_normalized(normalized, training.shingle_width)
                if len(normalized) < training.shingle_width:
                    short_near.append({"id": row_id, "field": field_path, "characters": len(normalized)})
                    continue
                if not pieces:
                    continue
                overlap_counts: Counter[int] = Counter()
                for piece in pieces:
                    overlap_counts.update(training.shingle_documents.get(_shingle_key(piece), ()))
                for document_index, count in overlap_counts.items():
                    overlap = count / len(pieces)
                    if overlap >= near_threshold:
                        row_detail["near_fields"].append(
                            {
                                "field": field_path,
                                "training_document_id": training.document_ids[document_index],
                                "overlap": overlap,
                            }
                        )
                        near_rows[row_id] = max(near_rows.get(row_id, 0.0), overlap)
            if any(row_detail[key] for key in ("exact_fields", "substring_fields", "near_fields")):
                details.append(row_detail)
            row_count += 1
    observed_hash = source_digest.hexdigest()
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise ValueError("benchmark file hash differs from its frozen manifest")
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError("benchmark row count differs from its frozen manifest")
    short_substrings.sort(key=lambda item: (item["id"], item["field"]))
    short_near.sort(key=lambda item: (item["id"], item["field"]))
    details.sort(key=lambda item: item["id"])
    return {
        "rows": row_count,
        "fields": field_count,
        "sha256": observed_hash,
        "exact_matches": sorted(exact_rows),
        "substring_matches": sorted(substring_rows),
        "near_matches": [{"id": row_id, "overlap": near_rows[row_id]} for row_id in sorted(near_rows)],
        "short_fields_unassessed_for_substring": short_substrings,
        "short_fields_unassessed_for_near": short_near,
        "match_details": details,
        "status": "partial" if short_substrings or short_near else "scanned",
        "evidence_boundary": (
            "Fields are scanned independently against individual training documents for normalized exact documents, "
            "alignment-invariant substrings above the configured minimum, and per-document strided shingles."
        ),
    }


def validate_external_freeze(
    payload: Any,
    *,
    benchmark_manifest_sha256: str,
    required_fields: tuple[str, ...] = _FREEZE_FIELDS,
) -> None:
    """Require a frozen candidate, claim, evaluator and exact evaluation manifest."""
    if set(required_fields) != set(_FREEZE_FIELDS):
        raise ValueError("external freeze identity policy must include every bound candidate and benchmark field")
    if not isinstance(payload, dict) or payload.get("status") != "frozen":
        raise ValueError("external evaluation freeze is missing or not frozen")
    missing = [field for field in required_fields if not isinstance(payload.get(field), str) or not payload[field]]
    if missing:
        raise ValueError(f"external evaluation freeze is missing identity fields: {missing}")
    if payload["benchmark_manifest_sha256"] != benchmark_manifest_sha256:
        raise ValueError("external evaluation freeze does not bind the benchmark manifest")
    digest_fields = [field for field in required_fields if field.endswith("_sha256")]
    for field in digest_fields:
        value = payload[field]
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError(f"external evaluation freeze has invalid hash: {field}")


def reserve_single_scan(marker_path: Path, *, freeze_sha256: str, benchmark_manifest_sha256: str) -> None:
    """Atomically reserve the only permitted benchmark scan for a frozen run."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "scan_started",
        "freeze_sha256": freeze_sha256,
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
    }
    try:
        descriptor = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("benchmark scan marker already exists; refusing to reopen test inputs") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def complete_single_scan(marker_path: Path, *, report_sha256: str) -> None:
    """Mark the reserved benchmark scan complete without removing its lock."""
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("status") != "scan_started":
        raise ValueError("benchmark scan marker is absent or already completed")
    payload["status"] = "scan_completed"
    payload["report_sha256"] = report_sha256
    marker_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "TrainingCorpusIndex",
    "complete_single_scan",
    "index_training",
    "normalize_text",
    "reserve_single_scan",
    "scan_benchmark",
    "shingles",
    "validate_external_freeze",
]
