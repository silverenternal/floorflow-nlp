"""Per-document, per-field benchmark overlap diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from floorflow.nlp_flow.benchmark_contamination import digest, normalize_text, shingles


def scan_field_overlap(
    training: Sequence[Mapping[str, Any]],
    benchmark: Sequence[Mapping[str, Any]],
    *,
    width: int,
    threshold: float,
) -> dict[str, object]:
    """Find normalized exact matches and per-document shingle overlap.

    Matching is confined to each training document. It includes normalized
    exact documents, exact substrings, and strided shingle overlap; cross-
    language clones are not ruled out by a clean result.
    """
    if width < 1 or not 0 < threshold <= 1:
        raise ValueError("width must be positive and threshold must be in (0, 1]")
    documents: dict[str, set[str]] = defaultdict(set)
    normalized_documents: dict[str, str] = {}
    index: dict[str, set[str]] = defaultdict(set)
    seen_docs: set[str] = set()
    for row in training:
        doc_id, text = row.get("id"), row.get("text")
        if not isinstance(doc_id, str) or not doc_id or doc_id in seen_docs or not isinstance(text, str):
            raise ValueError("training documents require distinct nonempty ids and text")
        seen_docs.add(doc_id)
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("empty training text is not scannable")
        documents[digest(normalized)].add(doc_id)
        normalized_documents[doc_id] = normalized
        for piece in shingles(normalized, width):
            index[piece].add(doc_id)
    matches = []
    short_fields = []
    seen_fields: set[tuple[str, str]] = set()
    for row in benchmark:
        case_id, field, text = row.get("case_id"), row.get("field"), row.get("text")
        if not all(isinstance(value, str) and value for value in (case_id, field, text)):
            raise ValueError("benchmark fields require case id, field name and nonempty text")
        key = (case_id, field)
        if key in seen_fields:
            raise ValueError("duplicate benchmark case/field")
        seen_fields.add(key)
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("empty benchmark text is not scannable")
        pieces = shingles(normalized, width)
        exact = documents.get(digest(normalized), set())
        substring_ids = sorted(
            doc_id for doc_id, training_text in normalized_documents.items() if normalized in training_text
        )
        overlaps: dict[str, int] = defaultdict(int)
        if len(normalized) < width:
            short_fields.append({"case_id": case_id, "field": field})
        else:
            for piece in pieces:
                for doc_id in index.get(piece, ()):
                    overlaps[doc_id] += 1
        near = [
            {"document_id": doc_id, "overlap": count / len(pieces)}
            for doc_id, count in sorted(overlaps.items())
            if count / len(pieces) >= threshold
        ]
        if exact or substring_ids or near:
            matches.append(
                {
                    "case_id": case_id,
                    "field": field,
                    "exact_document_ids": sorted(exact),
                    "exact_substring_document_ids": substring_ids,
                    "near": near,
                }
            )
    return {
        "training_documents": len(seen_docs),
        "benchmark_fields": len(seen_fields),
        "width": width,
        "threshold": threshold,
        "matches": matches,
        "short_fields_unassessed_for_near": short_fields,
        "status": "partial" if short_fields or not benchmark or not training else "scanned",
        "evidence_boundary": (
            "Per-field normalized exact-document, exact-substring and strided shingle overlap within individual "
            "training documents; cross-language clones and fields absent from inputs are not excluded. "
            "Results require fingerprinted inputs."
        ),
    }
