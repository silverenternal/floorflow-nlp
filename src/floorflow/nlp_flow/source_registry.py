"""License and source-family registry contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_FAMILIES = {
    "code",
    "model_architecture",
    "engineering_acceleration",
}
REQUIRED_LANGUAGES = {"python", "rust", "cpp", "text", "triton", "linux"}
REQUIRED_FIELDS = (
    "id",
    "repository",
    "revision",
    "source_url",
    "family",
    "language",
    "license",
    "license_url",
    "include_globs",
    "exclude_globs",
    "usage",
    "redistribution",
)


def validate_source_registry(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        return tuple(errors)
    ids: set[str] = set()
    families: set[str] = set()
    languages: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            errors.append(f"sources[{index}] must be an object")
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        if missing:
            errors.append(f"sources[{index}] is missing {missing}")
            continue
        source_id = str(source["id"])
        if not source_id or source_id in ids:
            errors.append(f"sources[{index}] has a missing or duplicate id")
        ids.add(source_id)
        for field in ("repository", "revision", "source_url", "license", "license_url"):
            if not isinstance(source[field], str) or not source[field]:
                errors.append(f"{source_id}.{field} must be non-empty")
        if not isinstance(source["include_globs"], list) or not source["include_globs"]:
            errors.append(f"{source_id}.include_globs must be non-empty")
        if not isinstance(source["exclude_globs"], list):
            errors.append(f"{source_id}.exclude_globs must be a list")
        if source["usage"] not in {"training", "training_conditional", "evaluation_only"}:
            errors.append(f"{source_id} has unknown usage")
        if source["usage"] == "training_conditional" and source["redistribution"] != "do_not_redistribute_source":
            errors.append(f"{source_id} restricts redistribution but source use is not isolated")
        families.add(str(source["family"]))
        languages.add(str(source["language"]))
    if families != REQUIRED_FAMILIES:
        errors.append(f"source families must be exactly {sorted(REQUIRED_FAMILIES)}")
    if not REQUIRED_LANGUAGES.issubset(languages):
        errors.append(f"source languages are missing {sorted(REQUIRED_LANGUAGES - languages)}")
    return tuple(errors)


__all__ = [
    "REQUIRED_FAMILIES",
    "REQUIRED_LANGUAGES",
    "validate_source_registry",
]
