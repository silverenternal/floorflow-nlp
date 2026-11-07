import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from newera.nlp_flow.components.ast.diagnostics.tracing.ast_generation_trace import finish_trace, select_and_apply
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.ast.diagnostics.tracing.ast_trace_provenance import (
    build_expected_identity,
    compare_historical_result,
    inspect_bound_trace,
    make_trace_envelope,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_selection import _masked_choice
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import OpenNodeAction, TreeActionDecoder

ROOT = Path(__file__).resolve().parents[5]


def test_bound_trace_checks_identity_content_and_structural_replay(tmp_path: Path) -> None:
    grammar_path = ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json"
    grammar = load_ast_grammar(grammar_path, "python")
    decoder = TreeActionDecoder(grammar)
    trace: list[dict[str, object]] = []
    select_and_apply(decoder, lambda: OpenNodeAction("module"), trace, budget=2, parent_candidates=[])

    def fail():
        _masked_choice(torch.tensor([float("-inf")]), set(), head="node_kind")
        return OpenNodeAction("identifier")

    with pytest.raises(ValueError, match="no legal action"):
        select_and_apply(decoder, fail, trace, budget=2, parent_candidates=[])
    finish_trace(trace, decoder, "no legal action candidate")
    corpus = tmp_path / "corpus"
    checkpoint = tmp_path / "checkpoint"
    runtime = tmp_path / "runtime.py"
    for path in (corpus, checkpoint, runtime):
        path.write_text("fixture")
    identity = build_expected_identity(
        case_id="fixture",
        language="python",
        policy="prefix_transformer",
        prompt="prompt",
        corpus=corpus,
        checkpoint=checkpoint,
        grammar=grammar_path,
        runtime_sources={"runtime.py": runtime},
        config={},
    )
    envelope = make_trace_envelope(identity, trace)
    serialized = json.dumps(envelope, allow_nan=False)
    loaded = json.loads(serialized)
    result = inspect_bound_trace(
        loaded, identity, grammar, grammar_path=grammar_path, expected_error="no legal action candidate"
    )
    assert result["observed_error"]["empty_heads"] == ["node_kind"]
    assert result["execution_authenticity"] == "not_authenticated"
    assert result["primary_cause"] is None
    with pytest.raises(ValueError, match="replay grammar"):
        inspect_bound_trace(
            loaded,
            identity,
            replace(grammar, legal_root_kinds=("wrong",)),
            grammar_path=grammar_path,
            expected_error="no legal action candidate",
        )
    historical = {
        "case_id": "fixture",
        "language": "python",
        "actions": 1,
        "generation_error": "no legal action candidate",
    }
    comparison = compare_historical_result(result, identity, historical)
    assert comparison["comparison"] == "metadata_consistent_only"
    assert comparison["trajectory_equivalence"] == "not_established"
    with pytest.raises(ValueError, match="structurally replayed"):
        compare_historical_result({**result, "trace_status": "missing"}, identity, historical)
    with pytest.raises(ValueError, match="action count mismatch"):
        compare_historical_result(result, identity, {**historical, "actions": 2})
    with pytest.raises(ValueError, match="generation error mismatch"):
        compare_historical_result(result, identity, {**historical, "generation_error": "different"})

    changed = copy.deepcopy(loaded)
    changed["trace"][1]["choices"][0]["head"] = "action"
    with pytest.raises(ValueError, match="content hash"):
        inspect_bound_trace(
            changed, identity, grammar, grammar_path=grammar_path, expected_error="no legal action candidate"
        )
    other_identity = {**identity, "case_id": "other"}
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_bound_trace(
            loaded, other_identity, grammar, grammar_path=grammar_path, expected_error="no legal action candidate"
        )
    with pytest.raises(ValueError, match="validation split"):
        make_trace_envelope({**identity, "split": "test"}, trace)
    runtime.write_text("changed runtime")
    changed_identity = build_expected_identity(
        case_id="fixture",
        language="python",
        policy="prefix_transformer",
        prompt="prompt",
        corpus=corpus,
        checkpoint=checkpoint,
        grammar=grammar_path,
        runtime_sources={"runtime.py": runtime},
        config={},
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_bound_trace(
            loaded, changed_identity, grammar, grammar_path=grammar_path, expected_error="no legal action candidate"
        )
