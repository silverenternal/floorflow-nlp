import pytest

from floorflow.nlp_flow.tree_sequence_batching import TreeActionSequence, plan_tree_sequence_chunks


def _sequence() -> TreeActionSequence:
    return TreeActionSequence(
        action_ids=(10, 11, 20, 21, 30),
        roles=(1, 2, 3, 4, 5),
        language_ids=(2, 2, 2, 2, 2),
        attention_mask=(True, True, True, True, True),
        action_mask=(True, True, True, True, True),
        loss_mask=(True, True, True, True, True),
        parent_refs=(-1, 0, 0, 2, 2),
        subtree_spans=((0, 2), (2, 4), (4, 5)),
        eos_id=99,
    )


def test_chunks_preserve_whole_subtrees_and_copy_masked_ancestor_context() -> None:
    result = plan_tree_sequence_chunks(_sequence(), max_actions=4)
    assert result.status == "pass"
    assert result.rejection_count == 0
    assert [chunk.subtree_span_ids for chunk in result.chunks] == [(0,), (1,), (2,)]
    root, child, leaf = result.chunks
    assert root.action_ids == (10, 11, 99)
    assert child.action_ids == (10, 20, 21, 99)
    assert child.roles == (1, 3, 4, 0)
    assert child.language_ids == (2, 2, 2, 2)
    assert child.attention_mask == (True, True, True, True)
    assert child.parent_refs == (-1, 0, 1, -1)
    assert child.action_mask == (False, True, True, True)
    assert child.loss_mask == (False, True, True, True)
    assert child.source_positions == (0, 2, 3, -1)
    assert leaf.action_ids == (10, 20, 30, 99)


def test_oversized_subtree_is_counted_and_never_truncated() -> None:
    sequence = TreeActionSequence(
        action_ids=(1, 2, 3, 4),
        roles=(0, 0, 0, 0),
        language_ids=(1, 1, 1, 1),
        attention_mask=(True, True, True, True),
        action_mask=(True, True, True, True),
        loss_mask=(True, True, True, True),
        parent_refs=(-1, 0, 0, 0),
        subtree_spans=((0, 4),),
        eos_id=9,
    )
    result = plan_tree_sequence_chunks(sequence, max_actions=3)
    assert result.status == "rejected_all"
    assert result.chunks == ()
    assert result.rejected_subtree_ids == (0,)
    assert result.rejection_count == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"roles": (1,)},
        {"subtree_spans": ((0, 1), (2, 5))},
        {"parent_refs": (-1, 2, 0, 0, 0)},
        {"action_ids": (10, 11, 20, 21, 99)},
    ],
)
def test_invalid_alignment_coverage_parent_or_embedded_eos_is_rejected(changes: dict) -> None:
    base = _sequence().__dict__
    with pytest.raises(ValueError):
        plan_tree_sequence_chunks(TreeActionSequence(**(base | changes)), max_actions=4)


def test_all_action_positions_are_accounted_once_as_supervision() -> None:
    result = plan_tree_sequence_chunks(_sequence(), max_actions=4)
    supervised = [
        position
        for chunk in result.chunks
        for position, keep in zip(chunk.source_positions, chunk.loss_mask)
        if keep and position >= 0
    ]
    assert sorted(supervised) == list(range(len(_sequence().action_ids)))


def test_source_action_and_loss_masks_are_retained_for_selected_actions() -> None:
    changes = {"action_mask": (True, False, True, True, True), "loss_mask": (True, True, False, True, True)}
    sequence = TreeActionSequence(
        **(_sequence().__dict__ | changes)
    )
    result = plan_tree_sequence_chunks(sequence, max_actions=4)
    child = result.chunks[1]
    assert child.action_mask == (False, True, True, True)
    assert child.loss_mask == (False, False, True, True)
