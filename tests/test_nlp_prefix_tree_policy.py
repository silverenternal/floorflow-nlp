from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.prefix_policy_data import (
    audit_prefix_batch,
    encode_prefix_rows,
)
from floorflow.nlp_flow.prefix_tree_policy import (
    PrefixTreePolicy,
    train_prefix_tree_policy,
)
from floorflow.nlp_flow.prefix_tree_runtime import generate_prefix_tree
from floorflow.nlp_flow.real_corpus_action_data import (
    build_real_corpus_action_dataset,
)
from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def _rows(source: str, split: str) -> list[dict[str, object]]:
    graph = parse_ast_graph(source, "python")
    row = {
        "case_id": f"python-{split}",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": split,
        "ast": serialize_ast_graph(graph),
    }
    grammar = load_ast_grammar(GRAMMAR, "python")
    action_rows, report = build_real_corpus_action_dataset(
        [row],
        {"python": grammar},
        {
            "max_prompt_chars": 512,
            "max_symbol_names": 8,
            "max_identifiers": 24,
            "max_operators": 12,
            "max_literals": 12,
            "max_constructs": 16,
            "max_comment_fragments": 4,
            "max_comment_chars": 240,
            "max_docstring_chars": 480,
            "minimum_source_line_chars_for_leak_check": 24,
        },
    )
    assert report["status"] == "pass"
    return [json.loads(json.dumps(value.as_dict())) for value in action_rows]


def test_prefix_batches_pad_complete_action_sequences() -> None:
    train = _rows("def add(a, b):\n    return a + b\n", "train")
    validation = _rows(
        "def subtract(a, b):\n    return a - b\n",
        "validation",
    )
    grammar = load_ast_grammar(GRAMMAR, "python")
    vocabulary = build_vocabulary(train, {"python": grammar})

    batch = encode_prefix_rows(train, {"python": grammar}, vocabulary)
    report = audit_prefix_batch(batch)

    assert report["status"] == "pass"
    assert batch.action.shape[0] == 1
    assert batch.step_mask.any()
    assert batch.history["action"].shape[-1] == 8
    assert validation


def test_prefix_transformer_forward_and_training_contract() -> None:
    train = _rows("def add(a, b):\n    return a + b\n", "train")
    validation = _rows(
        "def subtract(a, b):\n    return a - b\n",
        "validation",
    )
    grammar = load_ast_grammar(GRAMMAR, "python")
    vocabulary = build_vocabulary(train, {"python": grammar})
    train_batch = encode_prefix_rows(train, {"python": grammar}, vocabulary)
    validation_batch = encode_prefix_rows(
        validation,
        {"python": grammar},
        vocabulary,
    )
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=32,
        layers=1,
        heads=4,
        max_steps=train_batch.action.shape[1],
    )

    output = model(train_batch)

    assert output["action"].shape[:2] == train_batch.action.shape
    assert output["copy_start"].shape == (
        1,
        train_batch.action.shape[1],
        512,
    )
    trained, metrics = train_prefix_tree_policy(
        train_batch,
        validation_batch,
        vocabulary,
        device=torch.device("cpu"),
        seed=11,
        epochs=1,
        batch_rows=1,
        hidden_size=32,
        layers=1,
        heads=4,
    )
    assert isinstance(trained, PrefixTreePolicy)
    assert metrics["validation_rows"] == 1


def test_prefix_generation_has_no_target_input() -> None:
    parameters = inspect.signature(generate_prefix_tree).parameters

    assert "prompt" in parameters
    assert not any("target" in name.lower() for name in parameters)
