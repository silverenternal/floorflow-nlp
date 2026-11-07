from __future__ import annotations

import json
from types import SimpleNamespace

from tokenizers import ByteLevelBPETokenizer

from newera.nlp_flow.components.research.training.data import training_data
from newera.nlp_flow.components.research.training.data.training_data import (
    PackedTokenDataset,
    packed_token_batches,
    render_training_text,
)


def _fixture(tmp_path):
    corpus = tmp_path / "train.jsonl"
    rows = [
        {"text": "alpha beta gamma", "family": "code"},
        {"text": "delta epsilon zeta", "family": "general_language"},
    ]
    corpus.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        [row["text"] for row in rows],
        vocab_size=260,
        min_frequency=1,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return corpus, tokenizer_path, tokenizer


def test_packing_is_deterministic_and_marks_tail_labels(tmp_path) -> None:
    corpus, _tokenizer_path, tokenizer = _fixture(tmp_path)
    first = list(
        packed_token_batches(
            corpus,
            tokenizer,
            context_length=8,
            max_sequences=4,
            drop_last=False,
        )
    )
    second = list(
        packed_token_batches(
            corpus,
            tokenizer,
            context_length=8,
            max_sequences=4,
            drop_last=False,
        )
    )

    assert first
    assert [row["input_ids"].tolist() for row in first] == [
        row["input_ids"].tolist() for row in second
    ]
    for row in first:
        assert row["input_ids"].shape == (8,)
        assert row["labels"].shape == (8,)
        assert row["family_ids"].shape == (8,)
        assert row["language_ids"].shape == (8,)
        assert row["segment_ids"].shape == (8,)
        assert (row["labels"][row["labels"] >= 0] == row["input_ids"][row["labels"] >= 0]).all()
        assert (row["segment_ids"][row["labels"] == -100] == -1).all()


def test_streaming_dataset_uses_only_requested_path(tmp_path) -> None:
    corpus, tokenizer_path, _tokenizer = _fixture(tmp_path)
    dataset = PackedTokenDataset(
        corpus,
        tokenizer_path,
        context_length=8,
        max_sequences=3,
        drop_last=True,
    )

    rows = list(dataset)
    assert 1 <= len(rows) <= 3
    assert all(row["input_ids"].shape == (8,) for row in rows)


def test_dataset_sequence_limit_is_shared_across_workers(tmp_path, monkeypatch) -> None:
    corpus, tokenizer_path, _tokenizer = _fixture(tmp_path)
    dataset = PackedTokenDataset(
        corpus,
        tokenizer_path,
        context_length=2,
        max_sequences=3,
        drop_last=True,
    )
    worker_rows = []
    for worker_id in range(2):
        monkeypatch.setattr(
            training_data,
            "get_worker_info",
            lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=2),
        )
        worker_rows.append(list(dataset))

    assert [len(rows) for rows in worker_rows] == [2, 1]
    assert sum(map(len, worker_rows)) == 3


def test_packing_accepts_multiple_ordered_shards(tmp_path) -> None:
    corpus, _tokenizer_path, tokenizer = _fixture(tmp_path)
    second = tmp_path / "delta.jsonl"
    second.write_text(
        json.dumps({"text": "eta theta iota", "family": "code"}) + "\n",
        encoding="utf-8",
    )
    rows = list(
        packed_token_batches(
            [corpus, second],
            tokenizer,
            context_length=8,
            max_sequences=8,
            drop_last=False,
        )
    )
    assert rows


def test_packer_emits_stable_namespaced_node_ids_when_requested(tmp_path) -> None:
    corpus, _tokenizer_path, tokenizer = _fixture(tmp_path)
    kwargs = {
        "context_length": 8,
        "drop_last": False,
        "node_identity_namespace": 23,
    }
    first = list(packed_token_batches(corpus, tokenizer, **kwargs))
    second = list(packed_token_batches(corpus, tokenizer, **kwargs))
    first_ids = [row["neuron_node_ids"].tolist() for row in first]

    assert first_ids == [row["neuron_node_ids"].tolist() for row in second]
    active_ids = [node_id for row in first_ids for node_id in row if node_id >= 0]
    assert len(active_ids) == len(set(active_ids))
    assert all(node_id == -1 for row in first for node_id, label in zip(
        row["neuron_node_ids"].tolist(), row["labels"].tolist(), strict=True
    ) if label == -100)


def test_chat_template_has_user_and_assistant_turns() -> None:
    text = render_training_text(
        {
            "family": "general_language",
            "language": "text",
            "text": "Once upon a time there was a tiny robot. It learned to help.",
        },
        text_mode="chat",
    )

    assert text is not None
    assert text.startswith("User: Tell me a short story about:")
    assert "\nAssistant: " in text
    assert text.endswith("\n")


def test_chat_template_family_filter_hides_unselected_families() -> None:
    assert (
        render_training_text(
            {"family": "code", "text": "def f(): return 1"},
            text_mode="chat",
            families=frozenset({"general_language"}),
        )
        is None
    )


def test_preformatted_chat_instruction_is_not_wrapped_twice() -> None:
    text = "User: What is a variable?\nAssistant: A named value store.\n"

    assert (
        render_training_text(
            {"family": "chat_instruction", "text": text},
            text_mode="chat",
        )
        == text.strip()
    )
