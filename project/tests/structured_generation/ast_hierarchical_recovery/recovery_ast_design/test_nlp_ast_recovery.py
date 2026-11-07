from __future__ import annotations

from newera.nlp_flow.components.ast.diagnostics.recovery.ast_recovery import build_error_recovery_suite
from newera.nlp_flow.components.ast.diagnostics.recovery.ast_recovery_probe import inspect_parse_recovery


def test_parser_recovery_probe_reports_error_nodes_without_repairing() -> None:
    probe = inspect_parse_recovery(
        "def broken(value)\n    return value\n",
        "python",
    )

    assert probe["root_type"] == "module"
    assert probe["has_error"]
    assert probe["error_node_count"] > 0


def test_error_recovery_suite_is_evaluation_only_and_recoverable() -> None:
    suite, report = build_error_recovery_suite([])

    assert report["all_recovered_to_expected_root"]
    assert not report["training_allowed"]
    assert all(case["access"] == "evaluation_only" for case in suite)
    assert all(not case["training_allowed"] for case in suite)
