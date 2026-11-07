"""Bind diagnostic traces to explicit, independently checked input identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from newera.nlp_flow.components.ast.diagnostics.validation.ast_failure_taxonomy import inspect_generation_trace
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar, load_ast_grammar

_HASH = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = (
    "corpus_sha256",
    "prompt_sha256",
    "checkpoint_sha256",
    "grammar_sha256",
    "runtime_sha256",
    "config_sha256",
)
_IDENTITY_FIELDS = frozenset((*_HASH_FIELDS, "case_id", "language", "split", "policy"))


def _json_value(value: object) -> object:
    """Make nonfinite model logits valid JSON without changing finite values."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("trace object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    """Produce stable, standards-compliant UTF-8 JSON for hashing."""
    return json.dumps(
        _json_value(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    """Hash files incrementally, including large checkpoints and corpora."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_expected_identity(
    *,
    case_id: str,
    language: str,
    policy: str,
    prompt: str,
    corpus: Path,
    checkpoint: Path,
    grammar: Path,
    runtime_sources: Mapping[str, Path],
    config: Mapping[str, object],
) -> dict[str, object]:
    """Build expected identity from inputs, not from an imported trace."""
    if not runtime_sources or any(not isinstance(name, str) or not name for name in runtime_sources):
        raise ValueError("named runtime sources are required")
    runtime_manifest = {name: file_sha256(path) for name, path in sorted(runtime_sources.items())}
    return _identity(
        {
            "case_id": case_id,
            "language": language,
            "policy": policy,
            "split": "validation",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "corpus_sha256": file_sha256(corpus),
            "checkpoint_sha256": file_sha256(checkpoint),
            "grammar_sha256": file_sha256(grammar),
            "runtime_sha256": hashlib.sha256(canonical_bytes(runtime_manifest)).hexdigest(),
            "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
        }
    )


def _identity(identity: Mapping[str, object]) -> dict[str, object]:
    if set(identity) != _IDENTITY_FIELDS:
        raise ValueError("trace identity has missing or unexpected fields")
    if identity["split"] != "validation":
        raise ValueError("trace identity must use validation split")
    if identity["language"] not in {"python", "rust", "cpp"}:
        raise ValueError("unsupported trace language")
    if identity["policy"] not in {"prefix_transformer", "event_mlp"}:
        raise ValueError("unsupported trace policy")
    if not isinstance(identity["case_id"], str) or not identity["case_id"]:
        raise ValueError("trace case id must be nonempty")
    for key in _HASH_FIELDS:
        value = identity[key]
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"{key} must be a lowercase SHA-256")
    return dict(identity)


def make_trace_envelope(identity: Mapping[str, object], trace: list[Mapping[str, Any]]) -> dict[str, object]:
    """Create a hash-bound record, without asserting execution authenticity."""
    clean_trace = _json_value(trace)
    body = {"version": "nlp-ast-trace-v1", "identity": _identity(identity), "trace": clean_trace}
    return {**body, "sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def inspect_bound_trace(
    envelope: Mapping[str, Any],
    expected_identity: Mapping[str, object],
    grammar: ASTTransitionGrammar,
    *,
    grammar_path: Path,
    expected_error: str | None,
) -> dict[str, object]:
    """Check the envelope against an independently constructed identity and replay it.

    Matching hashes establish consistency, not authentication or proof of model execution.
    """
    if set(envelope) != {"version", "identity", "trace", "sha256"} or envelope["version"] != "nlp-ast-trace-v1":
        raise ValueError("unsupported trace envelope")
    expected = _identity(expected_identity)
    actual = _identity(envelope["identity"])
    if actual != expected or actual["language"] != grammar.language:
        raise ValueError("trace identity mismatch")
    if file_sha256(grammar_path) != expected["grammar_sha256"]:
        raise ValueError("trace grammar file hash mismatch")
    if load_ast_grammar(grammar_path, str(actual["language"])) != grammar:
        raise ValueError("replay grammar does not match the pinned grammar file")
    body = {key: envelope[key] for key in ("version", "identity", "trace")}
    digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if envelope["sha256"] != digest:
        raise ValueError("trace content hash mismatch")
    if not isinstance(envelope["trace"], list):
        raise TypeError("trace must be a list")
    result = inspect_generation_trace(envelope["trace"], grammar, expected_error=expected_error)
    return {
        **result,
        "identity": actual,
        "trace_sha256": digest,
        "identity_check": "matches_supplied_expected_identity",
        "execution_authenticity": "not_authenticated",
    }


def compare_historical_result(
    replay: Mapping[str, Any], identity: Mapping[str, object], historical_case: Mapping[str, Any]
) -> dict[str, object]:
    """Compare observable metadata without asserting a missing historical trajectory.

    A matching case, action count and terminal error do not establish that two
    action sequences, model executions, or generated outputs were identical.
    """
    expected = _identity(identity)
    if replay.get("trace_status") != "replayed_structure_only" or replay.get("identity_check") != (
        "matches_supplied_expected_identity"
    ):
        raise ValueError("historical comparison requires a structurally replayed bound trace")
    if replay.get("identity") != expected:
        raise ValueError("replay identity does not match expected case")
    if historical_case.get("case_id") != expected["case_id"] or historical_case.get("language") != expected["language"]:
        raise ValueError("historical case identity mismatch")
    historical_count = historical_case.get("actions")
    if isinstance(historical_count, bool) or not isinstance(historical_count, int) or historical_count < 0:
        raise ValueError("invalid historical action count")
    if replay.get("action_count") != historical_count:
        raise ValueError("historical action count mismatch")
    observed = replay.get("observed_error")
    if observed is not None and not isinstance(observed, Mapping):
        raise TypeError("invalid replay error record")
    replay_error = None if observed is None else observed.get("error")
    if replay_error != historical_case.get("generation_error"):
        raise ValueError("historical generation error mismatch")
    return {
        "case_id": expected["case_id"],
        "comparison": "metadata_consistent_only",
        "matched_fields": ["case_id", "language", "action_count", "generation_error"],
        "historical_action_sequence": "unavailable",
        "trajectory_equivalence": "not_established",
        "primary_cause": None,
    }
