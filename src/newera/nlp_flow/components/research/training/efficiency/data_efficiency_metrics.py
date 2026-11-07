"""Selection, gap-isolation, and forgetting metrics for NLP training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import (
        CorpusExample,
        SelectionMetrics,
    )


def selection_metrics(
    selected: Sequence[CorpusExample],
    *,
    reference: Sequence[CorpusExample],
) -> SelectionMetrics:
    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import (
        FAMILY_NAMES,
        GAP_NAMES,
        SelectionMetrics,
    )

    if not selected:
        raise ValueError("selected examples cannot be empty")
    if not reference:
        raise ValueError("reference examples cannot be empty")
    family_counts = Counter(item.family for item in selected)
    gap_counts = Counter(gap for item in selected for gap in item.gaps)
    reference_counts = Counter(gap for item in reference for gap in item.gaps)
    hashes = Counter(item.content_hash for item in selected)
    return SelectionMetrics(
        examples=len(selected),
        tokens=sum(item.token_count for item in selected),
        unique_examples=len({item.id for item in selected}),
        family_distribution={
            family: family_counts[family] / len(selected)
            for family in FAMILY_NAMES
        },
        gap_coverage={
            gap: (
                gap_counts[gap] / len(selected)
                if reference_counts[gap]
                else 0.0
            )
            for gap in GAP_NAMES
        },
        mean_difficulty=sum(item.difficulty for item in selected) / len(selected),
        duplicate_exposures=sum(count - 1 for count in hashes.values()),
    )


def gap_training_examples(
    examples: Sequence[CorpusExample],
    gap: str,
    *,
    replay: Sequence[CorpusExample] = (),
    replay_fraction: float = 0.1,
) -> tuple[CorpusExample, ...]:
    """Return one gap's isolated supervision set plus bounded replay."""

    from newera.nlp_flow.components.research.training.efficiency.data_efficiency import GAP_NAMES

    if gap not in GAP_NAMES:
        raise ValueError(f"unknown gap: {gap}")
    if not 0.0 <= replay_fraction <= 1.0:
        raise ValueError("replay_fraction must be in [0, 1]")
    primary = [example for example in examples if gap in example.gaps]
    if not primary:
        return ()
    replay_count = round(len(primary) * replay_fraction)
    replay_items = [
        example
        for index, example in enumerate(replay)
        if index < replay_count and example.id not in {item.id for item in primary}
    ]
    return tuple(primary + replay_items)


def token_efficiency_gain(
    loss_before: float,
    loss_after: float,
    seen_tokens: int,
) -> float:
    if seen_tokens <= 0:
        raise ValueError("seen_tokens must be positive")
    return (float(loss_before) - float(loss_after)) / seen_tokens


def forgetting(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    missing = set(before) ^ set(after)
    if missing:
        raise ValueError(f"loss keys do not match: {sorted(missing)}")
    return {
        key: max(0.0, float(after[key]) - float(before[key]))
        for key in before
    }


__all__ = [
    "forgetting",
    "gap_training_examples",
    "selection_metrics",
    "token_efficiency_gain",
]
