"""Native strict-JSON validity contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPECTED_HEADS = {
    "object_open_close",
    "property_key",
    "array_open_close",
    "string_value",
    "number_value",
    "boolean_value",
    "null_value",
    "schema_reference",
}


def validate_json_contract(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all JSON hard-validity contract violations."""

    errors: list[str] = []
    validity = payload.get("validity_contract")
    if not isinstance(validity, Mapping):
        errors.append("validity_contract must be an object")
    else:
        for field in (
            "strict_json_parse_validity",
            "schema_compliance",
        ):
            if validity.get(field) != 1.0:
                errors.append(f"{field} must equal 1.0")
        if validity.get("duplicate_key_count") != 0:
            errors.append("duplicate_key_count must be zero")
        if validity.get("nonfinite_number_count") != 0:
            errors.append("nonfinite_number_count must be zero")
        if validity.get("guarantee") != "valid_by_construction":
            errors.append("JSON must be valid_by_construction")
        if validity.get("guarantee_level") != "model_structure":
            errors.append("JSON guarantee must come from model structure")
        if validity.get("training_only_guarantee_possible") is not False:
            errors.append("unconstrained training alone cannot guarantee 100% JSON validity")

    native = payload.get("native_generation")
    if not isinstance(native, Mapping):
        errors.append("native_generation must be an object")
    else:
        heads = set(native.get("typed_heads", []))
        if heads != EXPECTED_HEADS:
            errors.append(f"typed heads must be exactly {sorted(EXPECTED_HEADS)}")
        if native.get("training_native") is not True:
            errors.append("JSON generation must be training-native")
        if native.get("posthoc_repair") is not False:
            errors.append("post-hoc JSON repair is forbidden")
        if native.get("retry") is not False:
            errors.append("JSON retries are forbidden")
        if "structurally zero probability" not in str(native.get("transition_rule", "")):
            errors.append("invalid JSON transitions must have structural zero probability")

    objectives = " ".join(payload.get("training_objectives", [])).lower()
    for objective in ("transition", "escaping", "finite number", "duplicate-key"):
        if objective not in objectives:
            errors.append(f"training objectives are missing {objective}")

    forbidden = " ".join(payload.get("forbidden_implementations", [])).lower()
    for shortcut in ("regex extraction", "brace", "parser fallback", "retry", "best-of-n"):
        if shortcut not in forbidden:
            errors.append(f"forbidden implementations must include {shortcut}")

    gates = payload.get("hard_gates")
    if not isinstance(gates, Mapping):
        errors.append("hard_gates must be an object")
    else:
        for split in ("train", "validation", "test", "engineering"):
            value = gates.get(f"{split}_strict_json_validity")
            if value != 1.0:
                errors.append(f"{split} JSON validity must equal 1.0")
        if gates.get("posthoc_repair_count") != 0 or gates.get("retry_count") != 0:
            errors.append("repair and retry counts must be zero")
        if "blocks C1-C4" not in str(gates.get("claim_rule", "")):
            errors.append("JSON failure must block C1-C4 claims")
    return tuple(errors)


__all__ = ["EXPECTED_HEADS", "validate_json_contract"]
