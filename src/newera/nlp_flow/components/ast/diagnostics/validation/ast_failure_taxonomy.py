"""Classify observed historical failures without inventing missing traces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar
from newera.nlp_flow.components.ast.diagnostics.tracing.ast_state_trace import snapshot
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeActionDecoder,
)

_ACTION_TYPES = {
    cls.__name__: cls
    for cls in (AttachEdgeAction, CloseScopeAction, DeclareLeafAction, OpenNodeAction, OpenScopeAction, TerminateAction)
}


def _same_state(left: object, right: object) -> bool:
    """Compare snapshots across in-memory tuples and JSON arrays."""
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _action_from_trace(record: Mapping[str, Any]):
    action_type = _ACTION_TYPES.get(record.get("type"))
    if action_type is None:
        raise ValueError("unknown trace action")
    return action_type(**{key: value for key, value in record.items() if key != "type"})


def inspect_generation_trace(
    trace: list[Mapping[str, Any]], grammar: ASTTransitionGrammar, *, expected_error: str | None
) -> dict[str, object]:
    """Replay a complete recorded trace, reporting observations rather than inferred causes.

    Consistency with a trace is not proof of its model, input, or checkpoint
    provenance; callers must independently bind and validate those identities.
    """
    if not trace or trace[-1].get("event") != "generation_end":
        raise ValueError("trace is missing its terminal event")
    decoder = TreeActionDecoder(grammar)
    observed_error: dict[str, object] | None = None
    for step, event in enumerate(trace[:-1]):
        if event.get("event") is not None or event.get("step") != step:
            raise ValueError("trace has a missing, duplicate, or out-of-order step")
        if observed_error is not None:
            raise ValueError("trace continues after a failed step")
        if not _same_state(event.get("before"), asdict(snapshot(decoder))):
            raise ValueError("trace pre-state does not match replay")
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise TypeError("trace has no choice records")
        empty_heads: list[str] = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise TypeError("invalid choice record")
            if "allowed_ids" not in choice:
                continue
            allowed = choice["allowed_ids"]
            selected = choice.get("selected_id")
            if not isinstance(allowed, list) or any(not isinstance(index, int) for index in allowed):
                raise ValueError("invalid candidate ids")
            if selected is None and not allowed:
                empty_heads.append(str(choice.get("head", "unknown")))
            elif selected not in allowed:
                raise ValueError("selected candidate is not allowed")
        action_record = event.get("action")
        error = event.get("error")
        if error is not None:
            if not isinstance(error, str) or not error:
                raise ValueError("invalid step error")
            if error == "no legal action candidate" and not empty_heads:
                raise ValueError("no-candidate error has no empty candidate mask")
            observed_error = {
                "kind": "empty_candidate_mask" if empty_heads else "selection_or_transition_error",
                "step": step,
                "empty_heads": empty_heads,
                "error": error,
            }
            if action_record is not None:
                if not isinstance(action_record, Mapping):
                    raise TypeError("invalid failed action record")
                try:
                    decoder.apply(_action_from_trace(action_record))
                except (ValueError, TypeError) as replay_error:
                    if str(replay_error) != error or type(replay_error).__name__ != event.get("error_type"):
                        raise ValueError("failed action does not reproduce the recorded error") from replay_error
                else:
                    raise ValueError("recorded failed action unexpectedly succeeded")
        else:
            if empty_heads or not isinstance(action_record, Mapping):
                raise ValueError("successful step has missing action or empty mask")
            decoder.apply(_action_from_trace(action_record))
        if not _same_state(event.get("after"), asdict(snapshot(decoder))):
            raise ValueError("trace post-state does not match replay")
    final = trace[-1]
    if not _same_state(final.get("state"), asdict(snapshot(decoder))):
        raise ValueError("terminal state does not match replay")
    if final.get("error") != expected_error:
        raise ValueError("terminal error does not match observed report")
    if observed_error is not None and observed_error["error"] != expected_error:
        raise ValueError("step error does not match terminal error")
    if expected_error is None and not decoder.terminated:
        raise ValueError("successful terminal event has no terminated tree")
    if observed_error is None and expected_error == "no legal action candidate":
        raise ValueError("no-candidate error has no recorded failing choice")
    if observed_error is None and expected_error is not None:
        observed_error = {"kind": "terminal_error", "step": None, "empty_heads": [], "error": expected_error}
    return {
        "trace_status": "replayed_structure_only",
        "observed_error": observed_error,
        "primary_cause": None,
        "cause_status": "unresolved_requires_semantic_diagnosis",
        "model_input_provenance": "not_verified",
        "action_count": len(decoder.actions),
    }


def classify_historical_case(case: Mapping[str, Any]) -> dict[str, object]:
    """Extract the observed outcome and distinguish it from its unknown cause."""
    failure = case.get("failure")
    parse_validity = case.get("parse_validity")
    metrics = case.get("metrics")
    if not isinstance(failure, str) or not failure:
        raise ValueError("historical failed case lacks an observed failure")
    if case.get("status") != "fail":
        raise ValueError("historical case is not a failure")
    if failure == "no legal action candidate":
        category = "no_candidate"
    elif failure.endswith("source contains a parse error"):
        category = "parser_rejection"
    elif failure == "quality threshold failed" and parse_validity is True and isinstance(metrics, Mapping):
        category = "parsed_quality_failure"
    else:
        category = "unclassified_observation"
    checks = case.get("checks")
    failed_checks = (
        [str(name) for name, passed in checks.items() if passed is False]
        if isinstance(metrics, Mapping) and isinstance(checks, Mapping)
        else []
    )
    return {
        "case_id": case["case_id"],
        "language": case["language"],
        "observed_category": category,
        "observed_failure": failure,
        "reported_action_count": case.get("actions"),
        "parse_validity": parse_validity,
        "failed_quality_checks": failed_checks,
        "quality_checks_measured": isinstance(metrics, Mapping),
        "trace_status": "missing",
        "first_failure_step": None,
        "primary_cause": None,
        "cause_status": "unresolved_missing_free_running_trace",
        "parser_delta": None,
        "ast_delta": None,
        "secondary_cascades": None,
    }
