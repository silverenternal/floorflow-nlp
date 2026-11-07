"""Training/validation source overlap; test content is outside this audit."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from newera.nlp_flow.components.research.evaluation.benchmarks.benchmark_field_overlap import scan_field_overlap


def audit_train_validation_overlap(
    rows: Sequence[Mapping[str, Any]], *, width: int, threshold: float
) -> dict[str, object]:
    """Compare pinned train and validation source texts, returning IDs only."""
    ids: set[str] = set()
    training = []
    validation = []
    language: dict[str, str] = {}
    for row in rows:
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("only train and validation source rows are allowed")
        case_id, source, source_hash = row.get("case_id"), row.get("source"), row.get("source_sha256")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case ids must be nonempty and unique across splits")
        if not isinstance(source, str) or hashlib.sha256(source.encode("utf-8")).hexdigest() != source_hash:
            raise ValueError(f"source fingerprint mismatch for {case_id}")
        if row.get("language") not in {"python", "rust", "cpp"}:
            raise ValueError("unknown source language")
        ids.add(case_id)
        language[case_id] = row["language"]
        if split == "train":
            training.append({"id": case_id, "text": source})
        else:
            validation.append({"case_id": case_id, "field": "source", "text": source})
    if not training or not validation:
        raise ValueError("both training and validation sources are required")
    scan = scan_field_overlap(training, validation, width=width, threshold=threshold)

    def annotate(matches: list[dict]) -> list[dict]:
        return [
            {
                **match,
                "validation_language": language[match["case_id"]],
                "training_languages": {
                    match_id: language[match_id]
                    for match_id in sorted(
                        set(match["exact_document_ids"]) | {item["document_id"] for item in match["near"]}
                    )
                },
            }
            for match in matches
        ]

    initial_short = scan["short_fields_unassessed_for_near"]
    short_ids = {item["case_id"] for item in initial_short}
    fallback_width = max(1, min(64, width // 4))
    fallback = (
        scan_field_overlap(
            training,
            [row for row in validation if row["case_id"] in short_ids],
            width=fallback_width,
            threshold=threshold,
        )
        if short_ids
        else None
    )
    unassessed = [] if fallback is None else fallback["short_fields_unassessed_for_near"]
    return {
        "version": "nlp-train-validation-overlap-v11",
        "status": "partial" if unassessed else "scanned_with_short_fallback" if fallback else scan["status"],
        "splits": ["train", "validation"],
        "test_content_evaluated": False,
        "training_documents": len(training),
        "validation_documents": len(validation),
        "validation_languages": dict(Counter(language[row["case_id"]] for row in validation)),
        "width": width,
        "threshold": threshold,
        "matches": annotate(scan["matches"]),
        "primary_short_validation_fields": initial_short,
        "short_fallback": None
        if fallback is None
        else {
            "width": fallback_width,
            "threshold": threshold,
            "documents": len(short_ids),
            "matches": annotate(fallback["matches"]),
        },
        "short_validation_fields_unassessed_for_near": unassessed,
        "evidence_boundary": (
            "Normalized exact-document, alignment-invariant exact-substring, and strided shingle comparisons "
            "against individual train documents only; semantic/cross-language clones, held-out test and external "
            "benchmarks are unassessed."
        ),
    }
