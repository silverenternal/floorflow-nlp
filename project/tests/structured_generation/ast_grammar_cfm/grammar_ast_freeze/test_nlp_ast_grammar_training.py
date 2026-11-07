from __future__ import annotations

import json
from pathlib import Path

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTGrammarState, load_ast_grammar
from newera.nlp_flow.components.ast.grammar.training.ast_grammar_ablation import evaluate_action_ablation
from newera.nlp_flow.components.ast.grammar.training.ast_grammar_audit import replay_teacher_ast
from newera.nlp_flow.components.ast.grammar.training.ast_grammar_training import (
    build_grammar_examples,
    encode_grammar_examples,
    train_grammar_legality_head,
)

ROOT = Path(__file__).resolve().parents[5]
GRAMMAR = ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json"


def test_grammar_training_receives_legal_and_illegal_actions() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    examples = build_grammar_examples(
        {"python": grammar},
        [
            {
                "language": "python",
                "kind": "premature_eos",
                "state": {},
            }
        ],
    )

    assert {example.label for example in examples} == {0, 1}
    _model, metrics = train_grammar_legality_head(
        examples,
        device=torch.device("cpu"),
        seed=7,
        epochs=1,
        batch_size=128,
    )

    assert metrics["train_examples"] > 0
    assert metrics["validation_examples"] > 0
    assert metrics["invalid_false_accepts"] >= 0


def test_teacher_replay_has_no_invalid_action_accepts() -> None:
    source = "def add(a, b):\n    return a + b\n"
    payload = json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))
    state = ASTGrammarState.from_file(GRAMMAR, "python")

    result = replay_teacher_ast(state, payload)

    assert result["invalid_probes"] > 0
    assert result["invalid_accepts"] == 0
    assert result["nodes"] > 1


def test_grammar_constrained_action_ablation_rejects_invalid_choices() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    examples = build_grammar_examples({"python": grammar}, [])
    encoded = encode_grammar_examples(examples)
    model, _metrics = train_grammar_legality_head(
        examples,
        device=torch.device("cpu"),
        seed=11,
        epochs=1,
        batch_size=256,
    )
    source = "value = 1\n"
    graph = parse_ast_graph(source, "python")
    rows = [
        {
            "split": "validation",
            "language": "python",
            "source": source,
            "ast": serialize_ast_graph(graph),
        }
    ]

    report = evaluate_action_ablation(
        model,
        encoded.category_indices,
        rows,
        device=torch.device("cpu"),
    )

    constrained = report["arms"]["grammar_constrained"]
    assert constrained["invalid_actions"] == 0
    assert constrained["parser_validity"] == 1.0
