"""Lossless action-aligned chunking for long tree action sequences."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeActionSequence:
    """One action stream with complete-subtree spans and global parent refs.

    Spans partition the action stream, excluding EOS. Parent references point
    to earlier action positions and allow the chunker to copy required ancestor
    actions into a context-only prefix.
    """

    action_ids: tuple[int, ...]
    roles: tuple[int, ...]
    language_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    action_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]
    parent_refs: tuple[int, ...]
    subtree_spans: tuple[tuple[int, int], ...]
    eos_id: int


@dataclass(frozen=True)
class TreeSequenceChunk:
    action_ids: tuple[int, ...]
    roles: tuple[int, ...]
    language_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    parent_refs: tuple[int, ...]
    action_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]
    source_positions: tuple[int, ...]
    subtree_span_ids: tuple[int, ...]


@dataclass(frozen=True)
class TreeSequenceBatchPlan:
    status: str
    chunks: tuple[TreeSequenceChunk, ...]
    rejected_subtree_ids: tuple[int, ...]
    rejection_count: int


def _validate(sequence: TreeActionSequence) -> None:
    size = len(sequence.action_ids)
    aligned = (
        sequence.roles,
        sequence.language_ids,
        sequence.attention_mask,
        sequence.action_mask,
        sequence.loss_mask,
        sequence.parent_refs,
    )
    if size == 0 or any(len(values) != size for values in aligned):
        raise ValueError("action, role, language and parent arrays must be nonempty and aligned")
    if sequence.eos_id in sequence.action_ids:
        raise ValueError("EOS may only be appended by the chunker")
    expected = 0
    for start, end in sequence.subtree_spans:
        if start != expected or end <= start or end > size:
            raise ValueError("subtree spans must partition the action stream in order")
        expected = end
    if expected != size or not sequence.subtree_spans:
        raise ValueError("subtree spans must cover every action")
    for index, parent in enumerate(sequence.parent_refs):
        if parent < -1 or parent >= index:
            raise ValueError("parent refs must be -1 or an earlier action position")


def _build_chunk(
    sequence: TreeActionSequence,
    subtree_ids: tuple[int, ...],
    max_actions: int,
) -> TreeSequenceChunk | None:
    selected: set[int] = set()
    for subtree_id in subtree_ids:
        start, end = sequence.subtree_spans[subtree_id]
        selected.update(range(start, end))
    context: set[int] = set()
    pending = [sequence.parent_refs[index] for index in selected]
    while pending:
        parent = pending.pop()
        if parent < 0 or parent in selected or parent in context:
            continue
        context.add(parent)
        pending.append(sequence.parent_refs[parent])
    positions = sorted(context) + sorted(selected)
    if len(positions) + 1 > max_actions:
        return None
    local = {source: index for index, source in enumerate(positions)}
    parents = tuple(
        -1 if sequence.parent_refs[source] < 0 else local[sequence.parent_refs[source]]
        for source in positions
    )
    supervised = [source in selected for source in positions]
    return TreeSequenceChunk(
        action_ids=tuple(sequence.action_ids[source] for source in positions) + (sequence.eos_id,),
        roles=tuple(sequence.roles[source] for source in positions) + (0,),
        language_ids=tuple(sequence.language_ids[source] for source in positions) + (sequence.language_ids[0],),
        attention_mask=tuple(sequence.attention_mask[source] for source in positions) + (True,),
        parent_refs=parents + (-1,),
        action_mask=tuple(
            sequence.action_mask[source] and is_selected for source, is_selected in zip(positions, supervised)
        )
        + (True,),
        loss_mask=tuple(
            sequence.loss_mask[source] and is_selected for source, is_selected in zip(positions, supervised)
        )
        + (True,),
        source_positions=tuple(positions) + (-1,),
        subtree_span_ids=subtree_ids,
    )


def plan_tree_sequence_chunks(sequence: TreeActionSequence, *, max_actions: int) -> TreeSequenceBatchPlan:
    """Greedily pack whole subtrees, copying ancestors as masked context.

    Each chunk receives EOS. Context copies retain their original action/role/
    language values, have valid remapped parent references, and are masked from
    action loss. A subtree that cannot fit with its required context is reported
    by ID and never truncated.
    """
    _validate(sequence)
    if max_actions < 2:
        raise ValueError("max_actions must fit at least one action and EOS")
    chunks: list[TreeSequenceChunk] = []
    rejected: list[int] = []
    pending: list[int] = []
    for subtree_id in range(len(sequence.subtree_spans)):
        trial = _build_chunk(sequence, tuple(pending + [subtree_id]), max_actions)
        if trial is not None:
            pending.append(subtree_id)
            continue
        if pending:
            chunks.append(_build_chunk(sequence, tuple(pending), max_actions))  # type: ignore[arg-type]
            pending = []
        single = _build_chunk(sequence, (subtree_id,), max_actions)
        if single is None:
            rejected.append(subtree_id)
        else:
            pending.append(subtree_id)
    if pending:
        final = _build_chunk(sequence, tuple(pending), max_actions)
        if final is None:
            raise RuntimeError("validated pending subtree group no longer fits")
        chunks.append(final)
    status = "pass" if not rejected else "partial_with_counted_rejections" if chunks else "rejected_all"
    return TreeSequenceBatchPlan(status, tuple(chunks), tuple(rejected), len(rejected))
