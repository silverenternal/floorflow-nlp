"""Construct and complexity holdout audit for the AST corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def load_ast_corpus(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"AST corpus row {line_number} must be an object")
            required = {
                "case_id",
                "language",
                "source_sha256",
                "split",
                "node_count",
                "max_depth",
                "constructs",
            }
            if not required.issubset(payload):
                raise ValueError(
                    f"AST corpus row {line_number} is missing "
                    f"{sorted(required - set(payload))}"
                )
            rows.append(payload)
    return rows


def _names(
    rows: Sequence[Mapping[str, object]],
    prefix: str,
) -> dict[str, set[str]]:
    output = {"train": set(), "validation": set(), "test": set()}
    for row in rows:
        split = str(row["split"])
        for value in row["constructs"]:
            if str(value).startswith(prefix):
                output[split].add(str(value).removeprefix(prefix))
    return output


def _heldout_names(
    names: Mapping[str, set[str]],
    *,
    prefix: str,
    modulus: int,
) -> set[str]:
    return {
        value
        for value in names["test"]
        if construct_name_digest(f"{prefix}{value}") >= modulus
    }


def construct_name_digest(construct: str) -> int:
    """Return the shared holdout digest bucket for a fully qualified construct."""
    return int(hashlib.sha256(construct.encode("utf-8")).hexdigest()[:8], 16) % 100


def _matches_structure_family(
    row: Mapping[str, object],
    family: str,
) -> bool:
    constructs = {str(value) for value in row["constructs"]}
    if family == "functions":
        return "function" in constructs
    if family == "control_flow":
        return "control_flow" in constructs
    if family == "classes_impl_traits":
        return bool(constructs & {"class", "impl_trait"})
    if family == "nested_scopes":
        return int(row["scope_count"]) >= 2
    if family == "macros_attributes":
        return "macro" in constructs or any(
            value.startswith(("macro_name:", "attribute_name:"))
            for value in constructs
        )
    if family == "cpp_templates":
        return row["language"] == "cpp" and "template" in constructs
    raise ValueError(f"unknown AST structure family: {family}")


def audit_structure_holdouts(
    rows: Sequence[Mapping[str, object]],
    *,
    families: Sequence[str],
    cases_per_family: int,
) -> dict[str, object]:
    excluded_hashes = {
        str(row["source_sha256"])
        for row in rows
        if row["split"] in {"train", "validation"}
    }
    checks = {}
    for family in families:
        matching = [
            row for row in rows if _matches_structure_family(row, family)
        ]
        test_rows = sorted(
            (row for row in matching if row["split"] == "test"),
            key=lambda row: (str(row["source_sha256"]), str(row["case_id"])),
        )
        selected = test_rows[:cases_per_family]
        leaked = sorted(
            str(row["source_sha256"])
            for row in selected
            if row["source_sha256"] in excluded_hashes
        )
        checks[family] = {
            "split_counts": {
                split: sum(row["split"] == split for row in matching)
                for split in ("train", "validation", "test")
            },
            "selected_case_ids": [str(row["case_id"]) for row in selected],
            "selected_source_hashes": [
                str(row["source_sha256"]) for row in selected
            ],
            "source_hash_overlap": leaked,
            "covered": bool(selected) and not leaked,
        }
    return {
        "families": dict(checks),
        "covered": all(bool(check["covered"]) for check in checks.values()),
    }


def audit_ast_holdouts(
    rows: Sequence[Mapping[str, object]],
    *,
    heldout_constructs: Sequence[str],
    heldout_name_prefixes: Sequence[str],
    heldout_name_modulus: int,
    structure_families: Sequence[str] = (),
    structure_cases_per_family: int = 16,
) -> dict[str, object]:
    construct_checks = {}
    for construct in heldout_constructs:
        construct_checks[construct] = {
            "train_rows": sum(
                row["split"] == "train"
                and construct in row["constructs"]
                for row in rows
            ),
            "validation_rows": sum(
                row["split"] == "validation"
                and construct in row["constructs"]
                for row in rows
            ),
            "test_rows": sum(
                row["split"] == "test"
                and construct in row["constructs"]
                for row in rows
            ),
        }
    name_checks = {}
    for prefix in heldout_name_prefixes:
        names = _names(rows, prefix)
        heldout = _heldout_names(
            names,
            prefix=prefix,
            modulus=heldout_name_modulus,
        )
        name_checks[prefix] = {
            "heldout_names": sorted(heldout),
            "train_overlap": sorted(names["train"] & heldout),
            "validation_overlap": sorted(names["validation"] & heldout),
            "test_coverage": sorted(names["test"] & heldout),
        }
    ordered = sorted(rows, key=lambda row: int(row["node_count"]))
    quartiles = []
    size = max(1, len(ordered) // 4)
    for index in range(4):
        start = index * size
        end = len(ordered) if index == 3 else (index + 1) * size
        bucket = ordered[start:end]
        quartiles.append(
            {
                "quartile": index + 1,
                "minimum_nodes": min(int(row["node_count"]) for row in bucket)
                if bucket
                else 0,
                "maximum_nodes": max(int(row["node_count"]) for row in bucket)
                if bucket
                else 0,
                "split_counts": {
                    split: sum(row["split"] == split for row in bucket)
                    for split in ("train", "validation", "test")
                },
            }
        )
    construct_pass = all(
        check["train_rows"] == 0 and check["test_rows"] > 0
        for check in construct_checks.values()
    )
    name_pass = all(
        not check["train_overlap"]
        and not check["validation_overlap"]
        and bool(check["test_coverage"])
        for check in name_checks.values()
    )
    complexity_pass = all(
        row["split_counts"]["validation"] > 0
        and row["split_counts"]["test"] > 0
        for row in quartiles
    )
    structure = audit_structure_holdouts(
        rows,
        families=structure_families,
        cases_per_family=structure_cases_per_family,
    )
    return {
        "version": "nlp-ast-construct-holdout-v1",
        "status": (
            "pass"
            if construct_pass
            and name_pass
            and complexity_pass
            and structure["covered"]
            else "fail"
        ),
        "rows": len(rows),
        "construct_checks": construct_checks,
        "name_checks": name_checks,
        "complexity_quartiles": quartiles,
        "structure_checks": structure["families"],
        "construct_isolated": construct_pass,
        "names_isolated": name_pass,
        "complexity_covered": complexity_pass,
        "structure_covered": structure["covered"],
        "evidence_boundary": (
            "Corpus split audit only; no model capability or parser synthesis result."
        ),
    }


__all__ = [
    "audit_ast_holdouts",
    "audit_structure_holdouts",
    "construct_name_digest",
    "load_ast_corpus",
]
