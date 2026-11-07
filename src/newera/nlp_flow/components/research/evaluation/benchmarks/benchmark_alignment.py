"""Rank, retrieval, and calibration metrics for proxy alignment."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end + 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs need equal length and at least two rows")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_norm = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
    )
    right_norm = math.sqrt(
        sum((value - right_mean) ** 2 for value in right)
    )
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("correlation is undefined for a constant vector")
    return numerator / (left_norm * right_norm)


def spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def kendall_tau(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs need equal length and at least two rows")
    concordant = 0
    discordant = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            product = left_delta * right_delta
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    return (concordant - discordant) / math.comb(len(left), 2)


def pairwise_sign_agreement(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("agreement inputs need equal length and at least two rows")
    matched = 0
    total = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0.0 or right_delta == 0.0:
                continue
            total += 1
            matched += (left_delta > 0.0) == (right_delta > 0.0)
    return matched / max(1, total)


def expected_calibration_error(
    confidence: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = 10,
) -> float:
    if len(confidence) != len(correct) or not confidence:
        raise ValueError("calibration inputs need equal non-empty length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if any(not 0.0 <= value <= 1.0 for value in confidence):
        raise ValueError("confidence must be in [0, 1]")
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, value in enumerate(confidence)
            if lower <= value < upper or (index + 1 == bins and value == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(confidence[position] for position in members) / len(members)
        accuracy = sum(correct[position] for position in members) / len(members)
        error += len(members) / total * abs(mean_confidence - accuracy)
    return error


def retrieval_metrics(
    ranked: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    *,
    k: int,
) -> dict[str, float]:
    if len(ranked) != len(relevant) or not ranked:
        raise ValueError("retrieval rows need equal non-empty length")
    if k <= 0:
        raise ValueError("k must be positive")
    recall_values = []
    reciprocal_ranks = []
    ndcg_values = []
    for ranked_row, relevant_row in zip(ranked, relevant):
        if not relevant_row:
            raise ValueError("each retrieval row needs at least one relevant item")
        top_k = list(ranked_row[:k])
        recall_values.append(len(set(top_k) & relevant_row) / len(relevant_row))
        first_relevant = next(
            (
                index
                for index, item in enumerate(ranked_row)
                if item in relevant_row
            ),
            None,
        )
        reciprocal_ranks.append(
            0.0 if first_relevant is None else 1.0 / (first_relevant + 1)
        )
        dcg = sum(
            1.0 / math.log2(index + 2)
            for index, item in enumerate(top_k)
            if item in relevant_row
        )
        ideal = sum(
            1.0 / math.log2(index + 2)
            for index in range(min(k, len(relevant_row)))
        )
        ndcg_values.append(dcg / max(1.0e-12, ideal))
    return {
        f"recall_at_{k}": sum(recall_values) / len(recall_values),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        f"ndcg_at_{k}": sum(ndcg_values) / len(ndcg_values),
    }


__all__ = [
    "expected_calibration_error",
    "kendall_tau",
    "pairwise_sign_agreement",
    "retrieval_metrics",
    "spearman_rank_correlation",
]
