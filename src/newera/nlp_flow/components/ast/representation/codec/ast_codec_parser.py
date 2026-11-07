"""Parser fingerprint and provenance helpers for AST codecs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from newera.nlp_flow.components.ast.representation.codec.ast_codec import (
    _MODULES,
    ParserFingerprint,
    language_for,
)


def parser_fingerprint(name: str) -> ParserFingerprint:
    """Return a reproducible fingerprint for the parser and grammar tables."""

    if name not in _MODULES:
        raise ValueError(f"unsupported AST language: {name}")
    language = language_for(name)
    module = _MODULES[name]
    module_spec = importlib.util.find_spec(module.__name__)
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError(f"cannot locate parser module {module.__name__}")
    module_dir = Path(module_spec.origin).parent
    native_candidates = sorted(module_dir.glob("_binding*.so"))
    if len(native_candidates) != 1:
        raise RuntimeError(
            f"expected one parser binding for {name}, found {native_candidates}"
        )
    native_sha256 = hashlib.sha256(native_candidates[0].read_bytes()).hexdigest()
    fields = tuple(
        language.field_name_for_id(index)
        for index in range(1, language.field_count + 1)
    )
    node_kinds = tuple(
        {
            "id": index,
            "name": language.node_kind_for_id(index),
            "named": language.node_kind_is_named(index),
            "visible": language.node_kind_is_visible(index),
            "supertype": language.node_kind_is_supertype(index),
        }
        for index in range(language.node_kind_count)
    )
    grammar_payload = json.dumps(
        {
            "abi_version": language.abi_version,
            "semantic_version": language.semantic_version,
            "fields": fields,
            "node_kinds": node_kinds,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return ParserFingerprint(
        language=name,
        package=module.__name__,
        package_version=importlib.metadata.version(module.__name__.replace("_", "-")),
        abi_version=int(language.abi_version),
        semantic_version=(
            None
            if language.semantic_version is None
            else tuple(int(value) for value in language.semantic_version)
        ),
        native_sha256=native_sha256,
        grammar_sha256=hashlib.sha256(grammar_payload).hexdigest(),
    )


def parser_provenance(name: str) -> dict[str, object]:
    """Return the frozen parser metadata consumed by the AST audit."""

    fingerprint = parser_fingerprint(name)
    language = language_for(name)
    return {
        "package": fingerprint.package.replace("_", "-"),
        "version": fingerprint.package_version,
        "abi_version": fingerprint.abi_version,
        "grammar_semantic_version": (
            list(fingerprint.semantic_version)
            if fingerprint.semantic_version is not None
            else None
        ),
        "node_kind_count": language.node_kind_count,
        "grammar_sha256": fingerprint.grammar_sha256,
    }


__all__ = ["parser_fingerprint", "parser_provenance"]
