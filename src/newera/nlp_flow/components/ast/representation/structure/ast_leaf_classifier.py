"""Typed lexical classes used by AST leaf actions."""

from __future__ import annotations


def classify_leaf(node_type: str, lexical: str) -> str:
    lowered = node_type.lower()
    if not lexical.strip():
        return "whitespace"
    if "comment" in lowered:
        return "comment"
    if "string" in lowered or "char_literal" in lowered:
        return "string_literal"
    if any(token in lowered for token in ("integer", "float", "number")):
        return "numeric_literal"
    if "identifier" in lowered:
        return "identifier"
    if lexical.isidentifier():
        return "identifier"
    if all(character in "()[]{};,.:" for character in lexical):
        return "punctuation"
    if all(character in "+-*/%<>=!&|^~@?" for character in lexical):
        return "operator"
    return "lexical_text"


__all__ = ["classify_leaf"]
