from __future__ import annotations

from pathlib import Path

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.leaf_pointer import (
    LeafExample,
    encode_leaf_examples,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def test_leaf_categories_are_unique_and_grammar_masked() -> None:
    train = [
        LeafExample(
            language="python",
            node_kind="identifier",
            leaf_class="identifier",
            context=("return",),
            pointer_label=0,
            lexical="value",
        ),
        LeafExample(
            language="python",
            node_kind="integer",
            leaf_class="numeric_literal",
            context=("value",),
            pointer_label=None,
            lexical="1",
        ),
    ]
    validation = [
        LeafExample(
            language="python",
            node_kind="identifier",
            leaf_class="identifier",
            context=("call",),
            pointer_label=None,
            lexical="name",
        )
    ]
    grammars = {"python": load_ast_grammar(GRAMMAR, "python")}

    train_batch, validation_batch, metadata = encode_leaf_examples(
        train,
        validation,
        grammars,
    )

    assert len(metadata["class_names"]) == 2
    assert train_batch.class_mask.shape == (2, 2)
    assert validation_batch.context_ids.shape[1] == 128
