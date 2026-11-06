from pathlib import Path

import pytest
import torch

from floorflow.nlp_flow.ast_generation_trace import select_and_apply
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_reachability import inspect_teacher_mask
from floorflow.nlp_flow.ast_state_trace import snapshot, trace_action
from floorflow.nlp_flow.prefix_tree_runtime import generate_prefix_tree
from floorflow.nlp_flow.real_corpus_tree_runtime import (
    _masked_choice,
    _select_real_action,
    generate_real_task,
    terminalizable_kinds,
    viable_parent_kinds,
)
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeActionDecoder,
)

ROOT = Path(__file__).resolve().parents[1]


def test_trace_preserves_prefix_and_reports_rejected_action() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    first = trace_action(decoder, OpenNodeAction("module"))
    captured = snapshot(decoder)
    failure = trace_action(decoder, TerminateAction())

    assert first["accepted"] is True
    assert captured.action_count == 1
    assert captured.expected_scope is not None
    assert failure["accepted"] is False
    assert "pending" in str(failure["error"])
    assert snapshot(decoder) == captured


def test_teacher_root_reports_vocabulary_mask_separately_from_grammar_replay() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    result = inspect_teacher_mask(
        decoder,
        OpenNodeAction("module"),
        1,
        viable_parent_kinds(grammar),
        terminalizable_kinds(grammar),
        {"node_ids": {}},
    )
    assert result["allowed_action_types"] == ["open_node"]
    assert result["reachable"] is False
    assert "node_kind_masked" in result["reasons"]
    assert result["lexical_value_checked"] is False
    assert trace_action(decoder, OpenNodeAction("module"))["accepted"] is True


def test_empty_candidate_mask_does_not_insert_an_action() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    before = snapshot(decoder)
    with pytest.raises(ValueError, match="no legal action candidate"):
        _masked_choice(torch.tensor([0.0]), set())
    assert snapshot(decoder) == before


def test_runtime_trace_preserves_mask_failure_without_retry() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    trace: list[dict[str, object]] = []
    before = snapshot(decoder)

    def choose():
        _masked_choice(torch.tensor([1.0, -2.0]), set())
        return OpenNodeAction("module")

    with pytest.raises(ValueError, match="no legal action candidate"):
        select_and_apply(decoder, choose, trace, budget=1, parent_candidates=[])
    assert snapshot(decoder) == before
    assert len(trace) == 1
    assert trace[0]["error"] == "no legal action candidate"
    assert trace[0]["choices"] == [
        {"head": "unspecified", "allowed_ids": [], "logits": [1.0, -2.0], "selected_id": None}
    ]


def test_runtime_trace_records_selected_action_and_is_optional() -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    traced = TreeActionDecoder(grammar)
    ordinary = TreeActionDecoder(grammar)
    trace: list[dict[str, object]] = []

    def choose():
        _masked_choice(torch.tensor([1.0, 2.0]), {0})
        return OpenNodeAction("module")

    select_and_apply(traced, choose, trace, budget=2, parent_candidates=[])
    select_and_apply(ordinary, choose, None, budget=2, parent_candidates=[])
    assert snapshot(traced) == snapshot(ordinary)
    assert trace[0]["action"] == {"type": "OpenNodeAction", "node_kind": "module"}
    assert trace[0]["choices"][0]["selected_id"] == 0


@pytest.mark.parametrize("prefix", [False, True])
def test_generation_entrypoint_captures_empty_node_candidates(prefix: bool) -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    vocabulary = {
        key: {}
        for key in ("chars", "node_ids", "edge_ids", "scope_ids", "action_ids", "class_ids", "lexeme_ids", "gap_ids")
    }

    class FakeModel:
        def __call__(self, *args):
            shape = (1, 1, 6) if prefix else (1, 6)
            node_shape = (1, 1, 1) if prefix else (1, 1)
            return {
                "node_budget": torch.tensor([[0.0, 1.0]]),
                "action": torch.zeros(shape),
                "node_kind": torch.zeros(node_shape),
            }

    trace: list[dict[str, object]] = []
    generate = generate_prefix_tree if prefix else generate_real_task
    source, actions, error = generate(
        FakeModel(),
        "",
        "python",
        grammar,
        vocabulary,
        device=torch.device("cpu"),
        trace=trace,
    )
    assert source is None and actions == []
    assert error == "no legal action candidate"
    assert len(trace) == 2
    assert trace[0]["choices"][-1]["allowed_ids"] == []
    assert trace[0]["before"] == trace[0]["after"]
    assert trace[-1]["event"] == "generation_end"
    assert trace[-1]["error"] == error


@pytest.mark.parametrize("copy_selected", [True, False])
def test_lexical_trace_records_copy_and_unmasked_fallback(copy_selected: bool) -> None:
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    for action in [
        OpenNodeAction("module"),
        OpenScopeAction(0, 0, "module"),
        AttachEdgeAction(0, "child"),
        OpenNodeAction("expression_statement"),
        AttachEdgeAction(1, "child"),
        OpenNodeAction("assignment"),
        AttachEdgeAction(2, "left"),
        OpenNodeAction("identifier"),
    ]:
        decoder.apply(action)
    vocabulary = {
        "class_ids": {"identifier": 0},
        "class_names": ["identifier"],
        "lexemes_by_kind_class": {},
        "lexeme_names": ["fallback"],
        "lexeme_ids": {"fallback": 0},
        "gaps_by_kind": {},
        "gap_names": [""],
        "gap_ids": {"": 0},
    }
    logits = {
        "action": torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]),
        "leaf_node": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "leaf_class": torch.tensor([[1.0]]),
        "copy": torch.tensor([1.0 if copy_selected else -1.0]),
        "copy_start": torch.tensor([[1.0, 0.0, 0.0]]),
        "copy_length": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "lexical": torch.tensor([[1.0]]),
        "gap": torch.tensor([[1.0]]),
    }
    trace: list[dict[str, object]] = []

    def choose():
        return _select_real_action(
            decoder,
            grammar,
            logits,
            vocabulary,
            "foo",
            [],
            20,
            viable_parent_kinds(grammar),
            terminalizable_kinds(grammar),
        )

    action = select_and_apply(decoder, choose, trace, budget=20, parent_candidates=[])
    assert action.lexical == ("foo" if copy_selected else "fallback")
    choices = trace[0]["choices"]
    copy_event = next(choice for choice in choices if choice["head"] == "copy")
    assert copy_event["selected"] is copy_selected
    assert any(choice["head"] == "gap_fallback" for choice in choices)
    if copy_selected:
        assert copy_event["value"] == "foo"
        assert any(choice["head"] == "copy_length" for choice in choices)
    else:
        assert any(choice["head"] == "lexical_fallback" for choice in choices)


def test_lexical_teacher_reachability_catches_unseen_value_and_gap() -> None:
    from floorflow.nlp_flow.tree_action_decoder import DeclareLeafAction

    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    decoder = TreeActionDecoder(grammar)
    for action in [
        OpenNodeAction("module"),
        OpenScopeAction(0, 0, "module"),
        AttachEdgeAction(0, "child"),
        OpenNodeAction("expression_statement"),
        AttachEdgeAction(1, "child"),
        OpenNodeAction("assignment"),
        AttachEdgeAction(2, "left"),
        OpenNodeAction("identifier"),
    ]:
        decoder.apply(action)
    vocabulary = {
        "class_ids": {"identifier": 0},
        "lexeme_ids": {"known": 0},
        "lexemes_by_kind_class": {},
        "gap_ids": {"": 0},
        "gaps_by_kind": {},
    }
    verdict = inspect_teacher_mask(
        decoder,
        DeclareLeafAction(3, "identifier", "unseen", " "),
        20,
        viable_parent_kinds(grammar),
        terminalizable_kinds(grammar),
        vocabulary,
        prompt="no match",
    )
    assert verdict["lexical_value_checked"] is True
    assert {"lexical_value_unreachable", "gap_value_unreachable"} <= set(verdict["reasons"])
