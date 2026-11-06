from __future__ import annotations

import json
from pathlib import Path

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_action_data import (
    build_prompt,
    build_real_corpus_action_dataset,
    iter_declaration_fragments,
    prompt_answer_leaks,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def _row(source: str, split: str = "train") -> dict[str, object]:
    graph = parse_ast_graph(source, "python")
    return {
        "case_id": f"python-{split}",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": split,
        "ast": serialize_ast_graph(graph),
    }


def test_prompt_uses_metadata_without_answer_leakage() -> None:
    source = (
        "# Return the sum of two values.\n"
        "def add(left, right):\n"
        "    return left + right\n"
    )
    payload = json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))

    prompt, source_kind, _metadata = build_prompt(
        "python",
        source,
        payload,
        max_prompt_chars=512,
        max_symbol_names=8,
        max_identifiers=24,
        max_operators=12,
        max_literals=12,
        max_constructs=16,
        max_comment_fragments=4,
        max_comment_chars=240,
        max_docstring_chars=480,
    )
    leakage = prompt_answer_leaks(
        prompt=prompt,
        source=source,
        ast_payload=payload,
        minimum_line_chars=24,
    )

    assert source_kind == "comments_symbols"
    assert "add" in prompt
    assert "sum of two values" in prompt
    assert leakage["pass"]


def test_real_corpus_actions_replay_exactly() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    rows, report = build_real_corpus_action_dataset(
        [_row("def add(a, b):\n    return a + b\n")],
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
    assert report["exact_source_replay_rate"] == 1.0
    assert report["canonical_ast_roundtrip_rate"] == 1.0
    assert rows[0].actions


def test_declaration_fragments_preserve_split_and_source_hash() -> None:
    source = (
        "def first(value):\n"
        "    return value + 1\n\n"
        "def second(value):\n"
        "    return value - 1\n"
    )
    fragments = iter_declaration_fragments(
        [_row(source, "validation")],
        {
            "max_fragments_per_row": 4,
            "max_fragment_chars": 2000,
            "max_fragment_nodes": 64,
            "declaration_types": {
                "python": ["function_definition"],
            },
        },
    )

    assert len(fragments) == 2
    assert all(row["split"] == "validation" for row in fragments)
    assert all(row["source_sha256"] for row in fragments)
    assert all("fragment_kind" in row for row in fragments)
