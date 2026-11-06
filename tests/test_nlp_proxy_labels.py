from __future__ import annotations

import pytest

from floorflow.nlp_flow.proxy_labels import (
    ProxyLabelAdapter,
    collate_proxy_labels,
)


def test_proxy_adapter_standardizes_all_head_families() -> None:
    adapter = ProxyLabelAdapter(sequence_length=6)
    batch = adapter.adapt(
        {
            "case_id": "case-1",
            "split": "train",
            "scalar_targets": {
                "execution_python": 1.0,
                "line_completion": 0.0,
            },
            "regression_targets": {"runtime_ratio": 1.2},
            "token_targets": {
                "file_localization": [0, 1, 0],
                "branch_execution_token": [1, -1, 1, 0],
            },
            "test_output_token": [1, 0, -1],
            "multiple_choice_target": 2,
            "choice_embeddings": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        }
    )

    assert batch.scalar_targets["execution_python"] == 1.0
    assert batch.scalar_targets["knowledge_correct"] == -1.0
    assert batch.regression_targets["runtime_ratio"] == 1.2
    assert batch.token_targets["file_localization"].shape == (6,)
    assert batch.token_masks["branch_execution_token"].sum() == 3
    assert batch.test_output_token is not None
    assert batch.test_output_mask is not None
    assert batch.test_output_mask.sum() == 2
    assert batch.multiple_choice_target == 2
    assert batch.choice_embeddings is not None
    assert batch.choice_embeddings.shape == (3, 2)


def test_proxy_collator_preserves_unknown_targets_and_masks() -> None:
    adapter = ProxyLabelAdapter(sequence_length=4)
    rows = [
        adapter.adapt(
            {
                "case_id": f"case-{index}",
                "split": "validation",
                "scalar_targets": {"execution_python": float(index)},
                "regression_targets": {"runtime_ratio": float(index + 1)},
                "token_targets": {
                    "file_localization": [index, -1],
                    "branch_execution_token": [-1],
                },
            }
        )
        for index in range(2)
    ]

    batch = collate_proxy_labels(rows)

    assert batch["proxy_execution_python"].tolist() == [0.0, 1.0]
    assert batch["proxy_knowledge_correct"].tolist() == [-1.0, -1.0]
    assert batch["proxy_regression_mask"].shape == (2, 3)
    assert batch["proxy_token_file_localization"].shape == (2, 4)
    assert batch["proxy_token_mask_file_localization"].sum() == 2
    assert "proxy_multiple_choice" not in batch


def test_proxy_adapter_rejects_test_labels() -> None:
    adapter = ProxyLabelAdapter(sequence_length=4)

    with pytest.raises(ValueError, match="cannot enter training"):
        adapter.adapt({"split": "test"})


def test_proxy_labels_do_not_turn_padding_into_loss_targets() -> None:
    adapter = ProxyLabelAdapter(sequence_length=4)
    batch = adapter.adapt({"split": "train"})

    assert not any(mask.any() for mask in batch.token_masks.values())
    assert not batch.regression_mask.any()


def test_proxy_adapter_rejects_oversized_token_targets() -> None:
    adapter = ProxyLabelAdapter(sequence_length=2)

    with pytest.raises(ValueError, match="exceed"):
        adapter.adapt(
            {
                "split": "validation",
                "token_targets": {"branch_execution_token": [0, 0, 0]},
            }
        )
