from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from newera.nlp_flow.components.platform.contracts.protocol._generated_protocol import JSON_ACTION_NAMES, OUTPUT_CONTRACT_ROWS, OUTPUT_MODE_NAMES
from newera.nlp_flow.model.components.tasks.output.json_grammar import ACTIONS
from newera.nlp_flow.components.generation.outputs.contracts.native_json_contract import (
    InvalidNativeJSONDecision,
    commit_native_json_decisions,
    output_schema,
)
from newera.nlp_flow.components.generation.outputs.contracts.native_json_contract import (
    NativeJSONDecision as D,
)
from newera.nlp_flow.components.generation.trees.typed.typed_tree import TREE_DOMAINS


def _tool_actions() -> list[D]:
    return [
        D("start_object"),
        D("key", key="schema_version"), D("colon"), D("string", value="nlp-json-output-v1"),
        D("comma"), D("key", key="kind"), D("colon"), D("string", value="tool_action"),
        D("comma"), D("key", key="payload"), D("colon"), D("start_object"),
        D("key", key="tool"), D("colon"), D("string", value="run_tests"),
        D("comma"), D("key", key="arguments"), D("colon"), D("start_object"),
        D("key", key="command"), D("colon"), D("start_array"),
        D("string", value="pytest"), D("end_array"),
        D("comma"), D("key", key="timeout_seconds"), D("colon"), D("number", value=30),
        D("end_object"), D("end_object"), D("end_object"),
    ]


def test_nested_tool_decisions_commit_without_serializer_added_fields() -> None:
    committed = commit_native_json_decisions("tool_action", _tool_actions())

    assert committed.value == {
        "schema_version": "nlp-json-output-v1",
        "kind": "tool_action",
        "payload": {
            "tool": "run_tests",
            "arguments": {"command": ["pytest"], "timeout_seconds": 30},
        },
    }
    assert committed.text == (
        '{"schema_version":"nlp-json-output-v1","kind":"tool_action","payload":'
        '{"tool":"run_tests","arguments":{"command":["pytest"],"timeout_seconds":30}}}'
    )
    assert committed.structured.validity.schema


def test_schema_rejects_duplicate_missing_wrong_type_and_wrong_tool_arguments() -> None:
    actions = _tool_actions()
    duplicate = actions.copy()
    duplicate.insert(4, D("key", key="schema_version"))
    with pytest.raises(InvalidNativeJSONDecision, match="duplicate schema key"):
        commit_native_json_decisions("tool_action", duplicate)

    missing_payload = [
        D("start_object"), D("key", key="schema_version"), D("colon"),
        D("string", value="nlp-json-output-v1"), D("comma"),
        D("key", key="kind"), D("colon"), D("string", value="tool_action"),
        D("end_object"),
    ]
    with pytest.raises(InvalidNativeJSONDecision, match="missing required fields"):
        commit_native_json_decisions("tool_action", missing_payload)

    wrong_type = actions.copy()
    wrong_type[7] = D("number", value=3)
    with pytest.raises(InvalidNativeJSONDecision, match="schema expected string"):
        commit_native_json_decisions("tool_action", wrong_type)

    wrong_args = actions.copy()
    command_array = next(i for i, item in enumerate(wrong_args) if item.action == "start_array")
    wrong_args[command_array] = D("string", value="pytest")
    with pytest.raises(InvalidNativeJSONDecision, match="schema expected array"):
        commit_native_json_decisions("tool_action", wrong_args)


def test_schema_enum_bounds_and_tool_registry_reject_uncommitted_values() -> None:
    actions = _tool_actions()
    wrong_tool = actions.copy()
    wrong_tool[14] = D("string", value="delete_repository")
    with pytest.raises(InvalidNativeJSONDecision, match="outside the committed schema enum"):
        commit_native_json_decisions("tool_action", wrong_tool)

    wrong_timeout = actions.copy()
    timeout_index = next(i for i, item in enumerate(wrong_timeout) if item.action == "number")
    wrong_timeout[timeout_index] = D("number", value=1000)
    with pytest.raises(InvalidNativeJSONDecision, match="above the schema maximum"):
        commit_native_json_decisions("tool_action", wrong_timeout)

    with pytest.raises(ValueError, match="unsupported tool schema"):
        output_schema("tool_action", tool_name="delete_repository")


@pytest.mark.skip(reason="Requires historical generated JSON/Rust contract snapshots removed from the active project layout.")
def test_json_actions_domains_and_output_modes_match_rust_generated_contract() -> None:
    root = Path(__file__).resolve().parents[5]
    subprocess.run(
        [
            "cargo", "run", "--quiet", "--manifest-path",
            str(root / "rust/nlp_flow_codegen/Cargo.toml"), "--",
            "--format", "python", "--check", str(root / "src/newera/nlp_flow/components/platform/contracts/protocol/_generated_protocol.py"),
        ],
        cwd=root,
        check=True,
    )
    assert ACTIONS == JSON_ACTION_NAMES
    assert {profile for profile, _structural, _schema, _executable in OUTPUT_CONTRACT_ROWS} >= {
        "json", "ast", "patch", "tool_action",
    }
    assert set(TREE_DOMAINS) == {"ast", "json", "patch", "tool_action"}
    assert {"json", "ast"} <= set(OUTPUT_MODE_NAMES)
