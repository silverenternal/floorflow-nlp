from __future__ import annotations

import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_discrete_flow import legal_action_candidates
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.tree_action_decoder import (
    DeclareLeafAction,
    TreeActionDecoder,
    trace_teacher_actions,
)
from floorflow.nlp_flow.tree_rollout import (
    TreeRollout,
    action_is_legal,
    align_teacher_next_labels,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def _teacher() -> tuple[object, list[object]]:
    grammar = load_ast_grammar(GRAMMAR, "python")
    payload = json.loads(
        serialize_ast_graph(parse_ast_graph("value = 1\n", "python"))
    )
    return grammar, trace_teacher_actions(grammar, payload)


def test_rollout_uses_the_same_target_free_masks_as_discrete_inference() -> None:
    grammar, actions = _teacher()
    rollout = TreeRollout(grammar)

    for action in actions:
        candidates = legal_action_candidates(rollout.decoder, grammar)
        assert rollout.legal_candidates == {
            key: tuple(values) for key, values in candidates.items()
        }
        assert action_is_legal(rollout.decoder, action)
        rollout.apply(action)

    assert rollout.decoder.terminated


def test_teacher_next_labels_are_masked_after_a_rollout_divergence() -> None:
    grammar, actions = _teacher()
    leaf_index = next(
        index for index, action in enumerate(actions)
        if isinstance(action, DeclareLeafAction)
    )
    divergent = list(actions)
    original = divergent[leaf_index]
    divergent[leaf_index] = DeclareLeafAction(
        node_index=original.node_index,
        leaf_class=original.leaf_class,
        lexical=f"{original.lexical}!",
        gap_before=original.gap_before,
    )

    alignment = align_teacher_next_labels(grammar, actions, divergent)

    assert alignment.compatible[: leaf_index + 1] == (True,) * (leaf_index + 1)
    assert alignment.compatible[leaf_index + 1 :] == (False,) * (
        len(actions) - leaf_index - 1
    )
    assert alignment.reasons[leaf_index + 1] == "prefix_state_diverged"
    assert alignment.divergence_indices == tuple(range(leaf_index + 1, len(actions)))


def test_rollout_rejects_actions_outside_the_shared_transition_mask() -> None:
    grammar, _ = _teacher()
    rollout = TreeRollout(grammar)
    illegal = DeclareLeafAction(node_index=0, leaf_class="identifier", lexical="x")

    assert not action_is_legal(TreeActionDecoder(grammar), illegal)
    with pytest.raises(ValueError, match="shared transition mask"):
        rollout.apply(illegal)
