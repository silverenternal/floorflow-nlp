"""Evaluation-only parser recovery cases for the AST-first pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from newera.nlp_flow.components.ast.diagnostics.recovery.ast_recovery_probe import inspect_parse_recovery

ERROR_RECOVERY_CASES: tuple[dict[str, str], ...] = (
    {
        "case_id": "python-function-missing-colon",
        "language": "python",
        "family": "function",
        "source": "def recover(value)\n    return value\n",
    },
    {
        "case_id": "python-unclosed-call",
        "language": "python",
        "family": "nested_scopes",
        "source": "def outer():\n    def inner():\n        return sum([1, 2]\n",
    },
    {
        "case_id": "python-class-indentation",
        "language": "python",
        "family": "class",
        "source": "class Box\n    def value(self):\n        return 1\n",
    },
    {
        "case_id": "rust-missing-semicolon",
        "language": "rust",
        "family": "control_flow",
        "source": "fn choose(x: bool) -> i32 {\n    let y = 1\n    y\n}\n",
    },
    {
        "case_id": "rust-unclosed-impl",
        "language": "rust",
        "family": "impl_trait",
        "source": "struct Box;\nimpl Box {\n    fn value(&self) -> i32 { 1 }\n",
    },
    {
        "case_id": "rust-partial-match",
        "language": "rust",
        "family": "nested_scopes",
        "source": (
            "fn outer(x: i32) -> i32 {\n"
            "    let f = |x: i32| { x + 1 ;\n"
            "    f(x)\n"
            "}\n"
        ),
    },
    {
        "case_id": "cpp-class-missing-semicolon",
        "language": "cpp",
        "family": "class",
        "source": "class Box {\n public:\n  int value() const { return 1; }\n}\n",
    },
    {
        "case_id": "cpp-unclosed-template",
        "language": "cpp",
        "family": "template",
        "source": "template <typename T\nT identity(T value) { return value; }\n",
    },
    {
        "case_id": "cpp-unclosed-control",
        "language": "cpp",
        "family": "control_flow",
        "source": "int choose(int value) {\n  if (value > 0) {\n    return value;\n}\n",
    },
)


def build_error_recovery_suite(
    corpus_rows: list[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    excluded_hashes = {
        str(row["source_sha256"])
        for row in corpus_rows
        if row["split"] in {"train", "validation"}
    }
    suite = []
    for case in ERROR_RECOVERY_CASES:
        probe = inspect_parse_recovery(case["source"], case["language"])
        source_sha256 = hashlib.sha256(case["source"].encode("utf-8")).hexdigest()
        suite.append(
            {
                **case,
                "source_sha256": source_sha256,
                "access": "evaluation_only",
                "training_allowed": False,
                "parser_recovery": probe,
            }
        )
    language_counts: dict[str, int] = {}
    for case in suite:
        language = str(case["language"])
        language_counts[language] = language_counts.get(language, 0) + 1
    report = {
        "version": "nlp-ast-error-recovery-suite-v1",
        "cases": len(suite),
        "language_counts": language_counts,
        "all_recovered_to_expected_root": all(
            bool(case["parser_recovery"]["has_error"])
            and case["parser_recovery"]["root_type"]
            == case["parser_recovery"]["expected_root_type"]
            for case in suite
        ),
        "training_hash_overlap": sorted(
            str(case["source_sha256"])
            for case in suite
            if case["source_sha256"] in excluded_hashes
        ),
        "training_allowed": False,
        "evidence_boundary": (
            "Parser recovery coverage only; malformed source is evaluation-only and "
            "cannot establish model generation validity or repair capability."
        ),
    }
    return suite, report


__all__ = ["ERROR_RECOVERY_CASES", "build_error_recovery_suite"]
