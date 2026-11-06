from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_action_data import (
    build_real_corpus_action_dataset,
)
from floorflow.nlp_flow.real_corpus_policy_data import (
    encode_action_rows,
)
from floorflow.nlp_flow.real_corpus_tree_policy import (
    RealCorpusTreePolicy,
    train_real_corpus_tree_policy,
)
from floorflow.nlp_flow.real_corpus_tree_runtime import generate_real_task
from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def _action_rows(source: str, split: str) -> list[dict[str, object]]:
    graph = parse_ast_graph(source, "python")
    corpus_row = {
        "case_id": f"python-{split}",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": split,
        "ast": serialize_ast_graph(graph),
    }
    grammar = load_ast_grammar(GRAMMAR, "python")
    rows, report = build_real_corpus_action_dataset(
        [corpus_row],
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
    return [json.loads(json.dumps(row.as_dict())) for row in rows]


def test_real_corpus_policy_encodes_and_trains_on_actions() -> None:
    train = _action_rows("def add(a, b):\n    return a + b\n", "train")
    validation = _action_rows(
        "def subtract(a, b):\n    return a - b\n",
        "validation",
    )
    grammar = load_ast_grammar(GRAMMAR, "python")
    vocabulary = build_vocabulary(train, {"python": grammar})
    batch = encode_action_rows(train, {"python": grammar}, vocabulary)
    model = RealCorpusTreePolicy(vocabulary)

    logits = model(
        batch.prompt_token_ids[batch.event_prompt_rows[:4]],
        batch.prompt_token_mask[batch.event_prompt_rows[:4]],
        batch.prompts[batch.event_prompt_rows[:4]],
        batch.prompt_masks[batch.event_prompt_rows[:4]],
        batch.state_features[:4],
        batch.context_kind[:4],
        batch.context_edge[:4],
        batch.candidate_kind[:4],
        batch.candidate_scope[:4],
        batch.candidate_position[:4],
        batch.candidate_mask[:4],
        {
            name: value[:4] for name, value in batch.history.items()
        },
        {
            name: value[:4] for name, value in batch.previous.items()
        },
    )

    assert logits["action"].shape == (4, 6)
    assert torch.isfinite(logits["action"]).all()
    trained, metrics = train_real_corpus_tree_policy(
        train,
        validation,
        {"python": grammar},
        device=torch.device("cpu"),
        seed=7,
        epochs=1,
        batch_size=128,
    )
    assert isinstance(trained, RealCorpusTreePolicy)
    assert metrics["validation_events"] > 0


def test_generation_api_has_no_target_ast_parameter() -> None:
    parameters = inspect.signature(generate_real_task).parameters

    assert "prompt" in parameters
    assert "language" in parameters
    assert not any("target" in name.lower() for name in parameters)
