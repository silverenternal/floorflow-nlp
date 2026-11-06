from __future__ import annotations

import pytest
import torch

from floorflow.nlp_flow.code_objectives import (
    FeedbackBudget,
    VerifierSignal,
    encode_verifier_feedback,
    make_diff_example,
    make_fim_example,
)


def test_fim_is_deterministic_and_hides_middle_from_inference() -> None:
    ids = list(range(2, 32))
    first = make_fim_example(
        ids,
        seed=17,
        prefix_sentinel=100,
        middle_sentinel=101,
        suffix_sentinel=102,
    )
    second = make_fim_example(
        ids,
        seed=17,
        prefix_sentinel=100,
        middle_sentinel=101,
        suffix_sentinel=102,
    )

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.labels, second.labels)
    assert first.input_ids.shape == first.labels.shape
    assert set(first.inference_view()) == {"input_ids", "prefix_ids", "suffix_ids"}
    middle = set(first.middle_ids.tolist())
    assert middle
    assert middle.isdisjoint(set(first.inference_view()["input_ids"].tolist()))
    active_labels = first.labels[first.labels >= 0]
    assert torch.equal(active_labels, first.middle_ids)


def test_diff_targets_only_changed_lines() -> None:
    original = "def add(a, b):\n    return a - b\n"
    patched = "def add(a, b):\n    return a + b\n"
    example = make_diff_example(original, patched)

    assert example.changed_line_count == 2
    assert "-    return a - b" in example.patch_text
    assert "+    return a + b" in example.patch_text
    assert set(example.inference_view()) == {"original_text"}


def test_verifier_feedback_and_retry_budget_are_bounded() -> None:
    signal = VerifierSignal(
        exit_code=1,
        failed_test_count=2,
        timed_out=False,
        stdout_tail="AssertionError: expected 3",
    )
    feedback = encode_verifier_feedback(signal, buckets=16)
    repeated = encode_verifier_feedback(signal, buckets=16)
    budget = FeedbackBudget(max_attempts=1)
    budget.record_attempt()

    assert torch.equal(feedback, repeated)
    assert feedback.tolist()[:3] == [1, 2, 0]
    assert budget.primary_retry_count == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.record_attempt()
