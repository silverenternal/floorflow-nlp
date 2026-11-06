"""Construct task prompts from independently registered evidence fields."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Origin = Literal["user_available", "target_derived", "training_only", "evaluation_only"]
Track = Literal["user_task_generation", "reconstruction_recovery"]
_ORIGINS = frozenset({"user_available", "target_derived", "training_only", "evaluation_only"})
_TRACKS = frozenset({"user_task_generation", "reconstruction_recovery"})


@dataclass(frozen=True)
class Evidence:
    """An externally classified source; origin cannot come from a prompt field."""

    evidence_id: str
    origin: Origin
    content: str
    sha256: str


@dataclass(frozen=True)
class PromptField:
    """A verbatim field referencing an independently registered source."""

    name: str
    value: str
    evidence_id: str


def content_sha256(value: str) -> str:
    """Return the SHA-256 of the exact UTF-8 source text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compile_prompt(
    track: Track,
    fields: Sequence[PromptField],
    trusted_sources: Mapping[str, Evidence],
) -> dict[str, object]:
    """Validate field origin and emit a fingerprinted prompt bundle.

    The caller must construct ``trusted_sources`` from an independently audited
    source registry. This function does not prove that a caller's classification
    is truthful. Only verbatim substrings of registered source content are
    admitted, avoiding unaudited transformations and invented provenance.
    """
    if track not in _TRACKS:
        raise ValueError("unknown task track")
    if not fields:
        raise ValueError("prompt must have at least one field")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        if not field.name or field.name in seen or not field.value or not field.evidence_id:
            raise ValueError("prompt field names must be unique and field values and evidence ids nonempty")
        seen.add(field.name)
        evidence = trusted_sources.get(field.evidence_id)
        if evidence is None or evidence.evidence_id != field.evidence_id:
            raise ValueError("prompt field references unregistered evidence")
        if evidence.origin not in _ORIGINS or evidence.sha256 != content_sha256(evidence.content):
            raise ValueError("source origin or content fingerprint is invalid")
        if evidence.origin == "evaluation_only":
            raise ValueError("evaluation-only evidence cannot enter a prompt")
        if track == "user_task_generation" and evidence.origin != "user_available":
            raise ValueError("user task prompt contains non-user evidence")
        if field.value not in evidence.content:
            raise ValueError("prompt field is not present in its registered evidence")
        records.append(
            {
                "name": field.name,
                "value": field.value,
                "evidence_id": evidence.evidence_id,
                "origin": evidence.origin,
                "source_sha256": evidence.sha256,
            }
        )
    payload = {"version": "nlp-prompt-provenance-v11", "track": track, "fields": records}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def verify_prompt_bundle(bundle: Mapping[str, object], trusted_sources: Mapping[str, Evidence]) -> None:
    """Revalidate field bindings and fingerprint against independent sources."""
    if set(bundle) != {"version", "track", "fields", "sha256"} or bundle["version"] != "nlp-prompt-provenance-v11":
        raise ValueError("unsupported prompt bundle")
    fields = bundle["fields"]
    if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
        raise TypeError("prompt bundle fields must be a list of objects")
    try:
        requested = [PromptField(field["name"], field["value"], field["evidence_id"]) for field in fields]
        expected = compile_prompt(bundle["track"], requested, trusted_sources)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid prompt field record") from error
    if expected != bundle:
        raise ValueError("prompt bundle metadata or fingerprint mismatch")
