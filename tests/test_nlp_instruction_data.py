from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from floorflow.nlp_flow.instruction_data import RowAlignedInstructionDataset
from floorflow.nlp_flow.model.benchmark_proxy import SCALAR_PROXY_TARGETS

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"


def test_row_aligned_dataset_masks_prompt_and_padding(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "Write an add function.",
                "response": "def add(a, b):\n    return a + b",
                "surface_valid": True,
                "execution_valid": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = RowAlignedInstructionDataset(
        source,
        TOKENIZER,
        context_length=64,
        seed=3,
    )
    sample = next(iter(dataset))

    assert sample["input_ids"].shape == (64,)
    assert sample["labels"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)
    assert sample["turn_ids"].shape == (64,)
    assert sample["family_ids"].shape == (64,)
    assert sample["language_ids"].shape == (64,)
    assert sample["continue_targets"].shape == (64,)
    assert sample["reply_mode_targets"].shape == (64,)
    assert sample["language_targets"].shape == (64,)
    assert sample["turn_boundary_targets"].shape == (64,)
    assert sample["validity_targets"].shape == (64, 3)
    assert sample["validity_mask"].shape == (64, 3)
    assert {
        key.removeprefix("proxy_")
        for key in sample
        if key.startswith("proxy_")
    } == set(SCALAR_PROXY_TARGETS)
    supervised = sample["labels"].ne(-100)
    assert supervised.any()
    assert not sample["labels"][0].ne(-100)
    assert not sample["attention_mask"][-1]
    decoded = Tokenizer.from_file(str(TOKENIZER)).decode(
        sample["input_ids"][supervised].tolist()
    )
    assert "return a + b" in decoded
    assert "User:" not in decoded


def test_row_aligned_dataset_is_deterministic_and_bounded(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [
        {
            "prompt": f"Question {index}",
            "response": f"Answer {index}",
        }
        for index in range(20)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = RowAlignedInstructionDataset(
        source,
        TOKENIZER,
        context_length=32,
        max_sequences=5,
        seed=9,
    )
    second = RowAlignedInstructionDataset(
        source,
        TOKENIZER,
        context_length=32,
        max_sequences=5,
        seed=9,
    )

    left = list(first)
    right = list(second)

    assert len(left) == len(right) == 5
    assert all(
        (a["input_ids"] == b["input_ids"]).all()
        for a, b in zip(left, right)
    )


def test_role_formatted_prompt_preserves_prior_turns(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": (
                    "User: Write add.\n"
                    "Assistant:\ndef add(a, b): return a + b\n"
                    "User: Rename it to plus."
                ),
                "response": "def plus(a, b): return a + b",
                "role_formatted": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = next(
        iter(
            RowAlignedInstructionDataset(
                source,
                TOKENIZER,
                context_length=96,
            )
        )
    )
    text = Tokenizer.from_file(str(TOKENIZER)).decode(
        sample["input_ids"].tolist()
    )

    assert "User: Write add." in text
    assert "User: Rename it to plus." in text
    assert text.count("Assistant:") == 2
    assert set(sample["turn_ids"].tolist()) >= {0, 1}


def test_instruction_replay_is_bounded_and_deterministic(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "Who are you?",
                "response": "A compact coding assistant.",
                "replay_count": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    left = list(
        RowAlignedInstructionDataset(
            source,
            TOKENIZER,
            context_length=32,
        )
    )
    right = list(
        RowAlignedInstructionDataset(
            source,
            TOKENIZER,
            context_length=32,
        )
    )

    assert len(left) == len(right) == 3
    assert all(
        (a["input_ids"] == b["input_ids"]).all()
        for a, b in zip(left, right)
    )


def test_long_prompt_keeps_full_response_and_eos(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    response = "def plus(a, b):\n    return a + b"
    source.write_text(
        json.dumps(
            {
                "prompt": "old context " * 200 + "final request",
                "response": response,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    sample = next(
        iter(
            RowAlignedInstructionDataset(
                source,
                TOKENIZER,
                context_length=128,
            )
        )
    )
    supervised = sample["labels"].ne(-100)
    decoded = tokenizer.decode(sample["labels"][supervised].tolist())

    assert response in decoded
    assert sample["labels"][supervised][-1].item() == tokenizer.token_to_id("<eos>")


def test_oversized_response_is_skipped_instead_of_corrupted(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "write code",
                "response": "x" * 100_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = list(
        RowAlignedInstructionDataset(
            source,
            TOKENIZER,
            context_length=64,
        )
    )

    assert samples == []
