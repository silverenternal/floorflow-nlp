"""Stable task-family and language identifiers for model input metadata."""

from __future__ import annotations

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import (
    CODE_LANGUAGE_NAMES,
    CORPUS_FAMILY_NAMES,
)

FAMILY_TO_ID = {
    name: index for index, name in enumerate(CORPUS_FAMILY_NAMES)
}
LANGUAGE_TO_ID = {
    name: index + 1 for index, name in enumerate(CODE_LANGUAGE_NAMES)
}
LANGUAGE_TO_ID["text"] = 0
DEFAULT_FAMILY_ID = FAMILY_TO_ID["general_language"]
DEFAULT_LANGUAGE_ID = LANGUAGE_TO_ID["text"]


def family_id(value: object) -> int:
    return FAMILY_TO_ID.get(str(value), DEFAULT_FAMILY_ID)


def language_id(value: object) -> int:
    return LANGUAGE_TO_ID.get(str(value), DEFAULT_LANGUAGE_ID)


__all__ = [
    "DEFAULT_FAMILY_ID",
    "DEFAULT_LANGUAGE_ID",
    "FAMILY_TO_ID",
    "LANGUAGE_TO_ID",
    "family_id",
    "language_id",
]
