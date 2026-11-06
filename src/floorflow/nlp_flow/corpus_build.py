"""Text normalization, chunking, and safety screening for mixed corpora."""

from __future__ import annotations

import hashlib
import re

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def chunks(text: str, size: int = 32_000) -> list[str]:
    if len(text) <= size:
        return [text]
    output = []
    for start in range(0, len(text), size):
        part = text[start : start + size].strip()
        if part:
            output.append(part)
    return output


def sanitize(rows: list[dict], *, deduplicate: bool = True) -> tuple[list[dict], dict]:
    seen: set[str] = set()
    output = []
    rejected_secret = 0
    duplicates = 0
    for row in rows:
        text = normalize(str(row["text"]))
        if not text:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            rejected_secret += 1
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if deduplicate and key in seen:
            duplicates += 1
            continue
        seen.add(key)
        for index, part in enumerate(chunks(text)):
            output.append({**row, "id": f"{row['id']}#{index}", "text": part})
    return output, {
        "duplicate_documents": duplicates,
        "secret_rejections": rejected_secret,
    }


__all__ = ["SECRET_PATTERNS", "chunks", "normalize", "sanitize"]
