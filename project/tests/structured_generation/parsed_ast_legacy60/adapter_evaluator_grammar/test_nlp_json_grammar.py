from __future__ import annotations

import pytest
import torch

from newera.nlp_flow.model import (
    CompactNLPDecoder,
    JSONGrammarState,
    JSONSchemaKeyHead,
    JSONTypedHeads,
    NLPModelConfig,
    SchemaGrammarState,
    SchemaNode,
    json_action_loss,
    schema_key_loss,
    validate_json_action_sequence,
)
from newera.nlp_flow.model.components.tasks.output.json_grammar import ACTIONS, JSONActionHead


def test_json_grammar_builds_nested_valid_structure() -> None:
    state = JSONGrammarState.initial()
    for action in (
        "start_object",
        "key",
        "colon",
        "start_array",
        "number",
        "comma",
        "string",
        "end_array",
        "comma",
        "key",
        "colon",
        "true",
        "end_object",
    ):
        state = state.transition(action)

    assert state.complete


def test_json_grammar_rejects_invalid_structure() -> None:
    state = JSONGrammarState.initial()

    try:
        state.transition("comma")
    except ValueError as error:
        assert "invalid JSON transition" in str(error)
    else:
        raise AssertionError("invalid root comma was accepted")


def test_json_action_head_sets_invalid_actions_to_negative_infinity() -> None:
    head = JSONActionHead(16)
    hidden = torch.randn(2, 16)
    states = [JSONGrammarState.initial(), JSONGrammarState.initial().transition("start_array")]

    logits = head(hidden, states)

    assert logits.shape == (2, len(ACTIONS))
    assert torch.isneginf(logits[0, ACTIONS.index("comma")])
    assert torch.isfinite(logits[0, ACTIONS.index("string")])
    assert torch.isfinite(logits[1, ACTIONS.index("end_array")])
    assert torch.isneginf(logits[1, ACTIONS.index("colon")])


def test_model_native_json_head_is_available() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=128,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            enable_json_action_head=True,
        )
    )

    logits = model.json_action_logits(
        torch.randn(2, 32),
        [JSONGrammarState.initial(), JSONGrammarState.initial()],
    )

    assert logits.shape == (2, len(ACTIONS))
    assert torch.isneginf(logits[:, ACTIONS.index("comma")]).all()
    typed = model.json_typed_logits(torch.randn(2, 32))
    assert set(typed) == set(model.json_typed_heads.NAMES)


def test_schema_grammar_enforces_required_fields_and_duplicate_keys() -> None:
    schema = SchemaNode(
        "object",
        required=("kind", "payload"),
        properties={
            "kind": SchemaNode("string"),
            "payload": SchemaNode(
                "object",
                required=("tool",),
                properties={
                    "tool": SchemaNode("string"),
                    "arguments": SchemaNode("array", items=SchemaNode("string")),
                },
            ),
        },
    )
    state = SchemaGrammarState.initial(schema)
    state = state.transition("start_object")
    state = state.transition("key", key="kind")
    state = state.transition("colon")
    state = state.transition("string")

    try:
        state.transition("end_object")
    except ValueError as error:
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("missing payload was accepted")

    state = state.transition("comma")
    state = state.transition("key", key="payload")
    state = state.transition("colon")
    state = state.transition("start_object")
    state = state.transition("key", key="tool")
    state = state.transition("colon")
    state = state.transition("string")
    state = state.transition("end_object")
    state = state.transition("end_object")

    assert state.complete


def test_schema_grammar_masks_illegal_close_and_type_actions() -> None:
    schema = SchemaNode(
        "object",
        required=("kind", "payload"),
        properties={
            "kind": SchemaNode("string"),
            "payload": SchemaNode("array", items=SchemaNode("number")),
        },
    )
    state = SchemaGrammarState.initial(schema).transition("start_object")

    assert "end_object" not in state.allowed_actions()
    state = state.transition("key", key="kind").transition("colon")
    assert state.allowed_actions() == ("string",)
    with pytest.raises(ValueError, match="schema expected string"):
        state.transition("number")
    state = state.transition("string").transition("comma")

    assert state.allowed_keys() == ("payload",)
    state = state.transition("key", key="payload").transition("colon")
    assert state.allowed_actions() == ("start_array",)
    with pytest.raises(ValueError, match="schema expected array"):
        state.transition("start_object")


def test_schema_action_head_masks_invalid_required_field_close() -> None:
    schema = SchemaNode(
        "object",
        required=("answer",),
        properties={"answer": SchemaNode("string")},
    )
    state = SchemaGrammarState.initial(schema).transition("start_object")
    logits = JSONActionHead(16)(
        torch.randn(1, 16),
        [state],
    )

    assert torch.isneginf(logits[0, ACTIONS.index("end_object")])
    assert torch.isfinite(logits[0, ACTIONS.index("key")])


def test_schema_key_head_scores_only_allowed_unseen_properties() -> None:
    schema = SchemaNode(
        "object",
        required=("answer",),
        properties={
            "answer": SchemaNode("string"),
            "explanation": SchemaNode("string"),
        },
    )
    initial = SchemaGrammarState.initial(schema).transition("start_object")
    after_answer = (
        initial.transition("key", key="answer")
        .transition("colon")
        .transition("string")
        .transition("comma")
    )
    head = JSONSchemaKeyHead(16)
    hidden = torch.randn(2, 16, requires_grad=True)

    logits = head(hidden, [initial, after_answer])
    loss = schema_key_loss(logits[:1], [initial], ["explanation"])
    loss.backward()

    assert logits[0].shape == (2,)
    assert logits[1].shape == (1,)
    assert hidden.grad is not None
    assert head.query.weight.grad is not None


def test_eight_json_typed_heads_are_independently_trainable() -> None:
    heads = JSONTypedHeads(16)
    hidden = torch.randn(2, 16, requires_grad=True)
    outputs = heads(hidden)
    loss = sum(value.square().mean() for value in outputs.values())
    loss.backward()

    assert set(outputs) == set(heads.NAMES)
    assert all(value.shape == (2, 2) for value in outputs.values())
    assert hidden.grad is not None
    assert all(heads.heads[name].weight.grad is not None for name in heads.NAMES)


def test_json_action_training_validates_target_sequence_and_gradients() -> None:
    action_names = ["start_object", "key", "colon", "string", "end_object"]
    actions = [ACTIONS.index(action) for action in action_names]
    targets = torch.tensor([actions])
    logits = torch.randn(1, len(actions), len(ACTIONS), requires_grad=True)

    loss = json_action_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert validate_json_action_sequence(action_names).complete


def test_json_action_training_rejects_invalid_target() -> None:
    actions = [ACTIONS.index("comma")]
    logits = torch.randn(1, 1, len(ACTIONS))

    with pytest.raises(ValueError, match="invalid JSON transition"):
        json_action_loss(logits, torch.tensor([actions]))
