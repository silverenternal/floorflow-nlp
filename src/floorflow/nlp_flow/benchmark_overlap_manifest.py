"""Fingerprint and audit prepared, explicitly scoped overlap inputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.benchmark_field_overlap import scan_field_overlap

_FIXTURE_SPLIT = re.compile(rb'"split"\s*:\s*"fixture"')


def _read_jsonl(path: Path, expected_role: str) -> tuple[list[dict[str, Any]], str]:
    """Check fixture metadata before decoding fields; hash the evaluated bytes."""
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if _FIXTURE_SPLIT.search(raw_line) is None:
                raise ValueError("implementation preflight accepts fixture rows only")
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise TypeError("overlap input row must be an object")
            if row.get("split") != "fixture":
                raise ValueError("implementation preflight accepts fixture rows only")
            if row.get("role") != expected_role:
                raise ValueError(f"{expected_role} input has an invalid fixture role")
            rows.append(row)
    return rows, digest.hexdigest()


def audit_fingerprinted_fixture_overlap(
    training_path: Path, benchmark_path: Path, *, width: int, threshold: float
) -> dict[str, object]:
    """Scan explicitly labeled fixtures; real benchmark content stays closed.

    This implementation preflight deliberately accepts synthetic fixture rows
    only. A future frozen evaluation workflow must independently scope and
    fingerprint any quarantined benchmark audit.
    """
    training, training_sha = _read_jsonl(training_path, "training_fixture")
    benchmark, benchmark_sha = _read_jsonl(benchmark_path, "benchmark_fixture")
    result = scan_field_overlap(training, benchmark, width=width, threshold=threshold)
    return {
        **result,
        "version": "nlp-fingerprinted-overlap-fixture-v11",
        "training_sha256": training_sha,
        "benchmark_sha256": benchmark_sha,
        "split": "synthetic_fixture_only",
        "evidence_boundary": (
            "Synthetic input and per-field scanner contract only; no external benchmark content, "
            "dataset contamination, clone coverage or model quality is certified."
        ),
    }
