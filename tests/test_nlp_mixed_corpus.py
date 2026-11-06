from __future__ import annotations

from floorflow.nlp_flow.corpus_build import normalize, sanitize


def test_normalize_and_sanitize_reject_duplicates_and_secrets() -> None:
    rows = [
        {"id": "a", "text": " line one \r\nline two ", "family": "code", "language": "python", "source_id": "a"},
        {"id": "b", "text": "line one\nline two", "family": "code", "language": "python", "source_id": "b"},
        {"id": "c", "text": "-----BEGIN " + "PRIVATE KEY-----\nsecret", "family": "code", "language": "python", "source_id": "c"},
    ]

    cleaned, stats = sanitize(rows)

    assert normalize(rows[0]["text"]) == "line one\nline two"
    assert len(cleaned) == 1
    assert stats == {"duplicate_documents": 1, "secret_rejections": 1}
