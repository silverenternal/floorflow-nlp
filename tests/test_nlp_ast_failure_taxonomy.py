import copy
import json
from pathlib import Path

import pytest
import torch

from floorflow.nlp_flow.ast_failure_taxonomy import classify_historical_case, inspect_generation_trace
from floorflow.nlp_flow.ast_generation_trace import finish_trace, select_and_apply
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_tree_runtime import _masked_choice
from floorflow.nlp_flow.tree_action_decoder import OpenNodeAction, TerminateAction, TreeActionDecoder

ROOT = Path(__file__).resolve().parents[1]


def test_missing_metrics_do_not_turn_unmeasured_checks_into_failures() -> None:
    result = classify_historical_case(
        {
            "case_id": "case",
            "language": "python",
            "status": "fail",
            "failure": "no legal action candidate",
            "parse_validity": False,
            "checks": {"node_type_f1": False},
            "metrics": None,
            "actions": 20,
        }
    )
    assert result["observed_category"] == "no_candidate"
    assert result["quality_checks_measured"] is False
    assert result["failed_quality_checks"] == []
    assert result["first_failure_step"] is None
    assert result["primary_cause"] is None


def test_parsed_quality_failure_reports_only_measured_failed_checks() -> None:
    result = classify_historical_case(
        {
            "case_id": "case",
            "language": "python",
            "status": "fail",
            "failure": "quality threshold failed",
            "parse_validity": True,
            "checks": {"parse_validity": True, "node_type_f1": False},
            "metrics": {"node_type_f1": 0.5},
            "actions": 4,
        }
    )
    assert result["observed_category"] == "parsed_quality_failure"
    assert result["failed_quality_checks"] == ["node_type_f1"]


def test_inconsistent_quality_failure_is_not_asserted_as_measured() -> None:
    result = classify_historical_case(
        {
            "case_id": "case",
            "language": "rust",
            "status": "fail",
            "failure": "quality threshold failed",
            "parse_validity": False,
            "checks": {},
            "metrics": None,
        }
    )
    assert result["observed_category"] == "unclassified_observation"
    with pytest.raises(ValueError, match="not a failure"):
        classify_historical_case({"status": "pass", "failure": "quality threshold failed"})


def test_trace_replay_checks_masks_state_and_terminal_error() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    trace: list[dict[str, object]] = []
    select_and_apply(decoder, lambda: OpenNodeAction("module"), trace, budget=2, parent_candidates=[])

    def no_candidate():
        _masked_choice(torch.tensor([2.0]), set(), head="node_kind")
        return OpenNodeAction("identifier")

    with pytest.raises(ValueError, match="no legal action candidate"):
        select_and_apply(decoder, no_candidate, trace, budget=2, parent_candidates=[])
    finish_trace(trace, decoder, "no legal action candidate")
    loaded = json.loads(json.dumps(trace))
    result = inspect_generation_trace(loaded, grammar, expected_error="no legal action candidate")
    assert result["observed_error"]["step"] == 1
    assert result["observed_error"]["empty_heads"] == ["node_kind"]
    assert result["primary_cause"] is None
    assert result["model_input_provenance"] == "not_verified"
    corrupt = copy.deepcopy(loaded)
    corrupt[1]["choices"][0]["allowed_ids"] = [0]
    with pytest.raises(ValueError, match="not allowed"):
        inspect_generation_trace(corrupt, grammar, expected_error="no legal action candidate")
    with pytest.raises(ValueError, match="terminal error"):
        inspect_generation_trace(loaded, grammar, expected_error="different")
    with pytest.raises(ValueError, match="terminal event"):
        inspect_generation_trace(loaded[:-1], grammar, expected_error="no legal action candidate")
    fake_success = copy.deepcopy(loaded[:1] + loaded[-1:])
    fake_success[-1]["error"] = None
    with pytest.raises(ValueError, match="no terminated tree"):
        inspect_generation_trace(fake_success, grammar, expected_error=None)
    missing_choice = copy.deepcopy(loaded[:1] + loaded[-1:])
    with pytest.raises(ValueError, match="no recorded failing choice"):
        inspect_generation_trace(missing_choice, grammar, expected_error="no legal action candidate")


def test_trace_replays_rejected_action_without_assigning_root_cause() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    trace: list[dict[str, object]] = []
    select_and_apply(decoder, lambda: OpenNodeAction("module"), trace, budget=2, parent_candidates=[])
    with pytest.raises(ValueError, match="pending") as failure:
        select_and_apply(decoder, TerminateAction, trace, budget=2, parent_candidates=[])
    finish_trace(trace, decoder, str(failure.value))
    result = inspect_generation_trace(trace, grammar, expected_error=str(failure.value))
    assert result["observed_error"]["kind"] == "selection_or_transition_error"
    assert result["observed_error"]["step"] == 1
    assert result["primary_cause"] is None
