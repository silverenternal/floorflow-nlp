from __future__ import annotations

import pytest

from newera.nlp_flow.components.research.evaluation.benchmarks.benchmark_alignment import (
    expected_calibration_error,
    kendall_tau,
    pairwise_sign_agreement,
    retrieval_metrics,
    spearman_rank_correlation,
)


def test_rank_correlations_and_sign_agreement() -> None:
    left = [1.0, 2.0, 3.0, 4.0, 5.0]
    right = [2.0, 4.0, 6.0, 8.0, 10.0]

    assert spearman_rank_correlation(left, right) == pytest.approx(1.0)
    assert kendall_tau(left, right) == pytest.approx(1.0)
    assert pairwise_sign_agreement(left, right) == pytest.approx(1.0)
    assert spearman_rank_correlation(left, list(reversed(right))) == pytest.approx(-1.0)


def test_ties_and_constant_series_are_handled_explicitly() -> None:
    assert spearman_rank_correlation(
        [1.0, 1.0, 2.0],
        [3.0, 3.0, 5.0],
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="constant"):
        spearman_rank_correlation([1.0, 1.0], [2.0, 3.0])


def test_expected_calibration_error_and_retrieval_metrics() -> None:
    assert expected_calibration_error(
        [0.8, 0.8, 0.8, 0.8],
        [True, True, False, False],
        bins=5,
    ) == pytest.approx(0.3)
    metrics = retrieval_metrics(
        ranked=[["a", "b", "c"], ["x", "y", "z"]],
        relevant=[{"a", "c"}, {"z"}],
        k=2,
    )

    assert metrics["recall_at_2"] == pytest.approx(0.25)
    assert metrics["mrr"] == pytest.approx(2.0 / 3.0)
    assert 0.0 < metrics["ndcg_at_2"] < 1.0
