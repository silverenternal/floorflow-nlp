from __future__ import annotations

from newera.nlp_flow.components.platform.data.splits.ast_holdout import (
    audit_ast_holdouts,
    construct_name_digest,
)


def _row(split: str, nodes: int, constructs: list[str]) -> dict[str, object]:
    return {
        "case_id": f"{split}-{nodes}",
        "language": "python",
        "source_sha256": f"{split}-{nodes}",
        "split": split,
        "node_count": nodes,
        "scope_count": 2,
        "max_depth": 3,
        "constructs": constructs,
    }


def test_ast_holdout_audit_rejects_construct_and_name_leakage() -> None:
    rows = [
        _row("train", 10, ["macro_name:A"]),
        _row("validation", 20, ["macro_name:B"]),
        _row("test", 30, ["macro_name:A"]),
    ]

    report = audit_ast_holdouts(
        rows,
        heldout_constructs=[],
        heldout_name_prefixes=["macro_name:"],
        heldout_name_modulus=0,
    )

    assert report["status"] == "fail"
    assert report["name_checks"]["macro_name:"]["train_overlap"]


def test_name_holdout_hashing_uses_the_fully_qualified_construct() -> None:
    prefix = "attribute_name:"
    name = next(
        f"name_{index}"
        for index in range(1000)
        if construct_name_digest(f"name_{index}") < 80
        <= construct_name_digest(f"{prefix}name_{index}")
    )
    rows = [
        _row("train", 10, [f"{prefix}{name}"]),
        _row("test", 20, [f"{prefix}{name}"]),
    ]

    report = audit_ast_holdouts(
        rows,
        heldout_constructs=[],
        heldout_name_prefixes=[prefix],
        heldout_name_modulus=80,
    )

    assert report["name_checks"][prefix]["heldout_names"] == [name]
    assert report["name_checks"][prefix]["train_overlap"] == [name]


def test_structure_holdout_selects_only_test_cases() -> None:
    rows = [
        _row("train", 10, ["function", "control_flow", "class"]),
        _row("validation", 20, ["impl_trait", "macro"]),
        _row("test", 30, ["function", "control_flow", "class", "macro"]),
        _row("test", 40, ["impl_trait", "template"]),
    ]
    rows[-1]["language"] = "cpp"

    report = audit_ast_holdouts(
        rows,
        heldout_constructs=[],
        heldout_name_prefixes=[],
        heldout_name_modulus=80,
        structure_families=[
            "functions",
            "control_flow",
            "classes_impl_traits",
            "nested_scopes",
            "macros_attributes",
            "cpp_templates",
        ],
    )

    assert report["structure_covered"]
    assert all(
        case_id.startswith("test-")
        for check in report["structure_checks"].values()
        for case_id in check["selected_case_ids"]
    )
