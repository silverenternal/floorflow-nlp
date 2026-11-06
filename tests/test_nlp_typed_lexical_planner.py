from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.typed_lexical_planner import (
    VALUE_HEADS,
    TypedLexicalPlanner,
    shift_lexical_history,
    valid_span_length_mask,
)


def _inputs():
    batch, steps, evidence, hidden = 2, 4, 5, 16
    model = TypedLexicalPlanner(
        node_kinds=8,
        leaf_classes=5,
        lexical_tokens=20,
        value_vocab_sizes={name: 7 + index for index, name in enumerate(VALUE_HEADS)},
        hidden_size=hidden,
        heads=4,
        max_steps=8,
        max_span_length=6,
        layers=1,
    )
    targets = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    history = shift_lexical_history(targets, bos_id=1)
    structure = torch.randn(batch, steps, hidden)
    node_ids = torch.randint(1, 8, (batch, steps))
    prompt_states = torch.randn(batch, evidence, hidden)
    prompt_mask = torch.tensor([[True, True, True, False, True], [True, True, True, True, True]])
    scope_mask = torch.ones(batch, steps, evidence, dtype=torch.bool)
    scope_mask[0, 1] = False
    class_mask = torch.ones(batch, steps, 5, dtype=torch.bool)
    class_mask[:, :, 4] = False
    value_masks = {
        name: torch.ones(batch, steps, 7 + index, dtype=torch.bool)
        for index, name in enumerate(VALUE_HEADS)
    }
    value_masks["literal"][:, :, 0] = False
    return model, targets, structure, node_ids, history, prompt_states, prompt_mask, scope_mask, class_mask, value_masks


def test_copy_generate_and_native_type_masks() -> None:
    model, _targets, structure, node_ids, history, prompt, prompt_mask, scope, classes, values = _inputs()
    output = model(structure, node_ids, history, prompt, prompt_mask, scope, classes, values)
    assert output["mode_logits"].shape == (2, 4, 2)
    assert output["leaf_class_logits"].shape == (2, 4, 5)
    assert output["copy_start_logits"].shape == (2, 4, 5)
    assert output["copy_length_logits"].shape == (2, 4, 6)
    assert torch.isneginf(output["mode_logits"][0, 1, 0])
    assert torch.isfinite(output["mode_logits"][0, 1, 1])
    assert torch.isneginf(output["copy_start_logits"][0, 1]).all()
    assert torch.isneginf(output["leaf_class_logits"][:, :, 4]).all()
    assert torch.isneginf(output["literal_logits"][:, :, 0]).all()
    assert all(f"{name}_logits" in output for name in VALUE_HEADS)


def test_future_lexical_targets_do_not_enter_earlier_decisions() -> None:
    model, targets, structure, node_ids, history, prompt, prompt_mask, scope, classes, values = _inputs()
    model.eval()
    first = model(structure, node_ids, history, prompt, prompt_mask, scope, classes, values)
    future_changed = targets.clone()
    future_changed[:, 2] += 1
    changed_history = shift_lexical_history(future_changed, bos_id=1)
    second = model(structure, node_ids, changed_history, prompt, prompt_mask, scope, classes, values)
    for key in ("mode_logits", "leaf_class_logits", "copy_start_logits", "identifier_logits"):
        torch.testing.assert_close(first[key][:, :3], second[key][:, :3])


def test_typed_heads_and_copy_generate_losses_have_finite_gradients() -> None:
    model, _targets, structure, node_ids, history, prompt, prompt_mask, scope, classes, values = _inputs()
    output = model(structure, node_ids, history, prompt, prompt_mask, scope, classes, values)
    loss = F.cross_entropy(output["mode_logits"].reshape(-1, 2), torch.ones(8, dtype=torch.long))
    loss = loss + F.cross_entropy(output["leaf_class_logits"].reshape(-1, 5), torch.zeros(8, dtype=torch.long))
    for name in VALUE_HEADS:
        target = 1 if name == "literal" else 0
        loss = loss + F.cross_entropy(
            output[f"{name}_logits"].reshape(-1, 7 + VALUE_HEADS.index(name)),
            torch.full((8,), target, dtype=torch.long),
        )
    copy_sites = torch.tensor([[True, False, False, False], [True, True, True, True]])
    loss = loss + F.cross_entropy(output["copy_start_logits"][copy_sites], torch.zeros(int(copy_sites.sum()), dtype=torch.long))
    loss = loss + F.cross_entropy(output["copy_length_logits"][copy_sites], torch.zeros(int(copy_sites.sum()), dtype=torch.long))
    loss.backward()
    for parameter in (
        model.mode_head.weight,
        model.class_head.weight,
        model.value_heads["identifier"].weight,
        model.value_heads["literal"].weight,
        model.value_heads["operator"].weight,
        model.value_heads["trivia"].weight,
        model.copy_query.weight,
        model.copy_length_head.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_variable_unicode_span_lengths_are_bounded_by_source() -> None:
    start = torch.tensor([[0, 3]])
    source_length = torch.tensor([[5, 4]])
    mask = valid_span_length_mask(start, source_length, max_length=5)
    assert mask.shape == (1, 2, 5)
    assert mask[0, 0].tolist() == [True, True, True, True, True]
    assert mask[0, 1].tolist() == [True, False, False, False, False]

