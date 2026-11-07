from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_training import (
    LOSS_NAMES,
    RolloutCurriculum,
    hierarchical_tree_losses,
)
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_action_data import build_real_corpus_action_dataset
from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary

ROOT = Path(__file__).resolve().parents[5]


def _training_batch():
    grammar = load_ast_grammar(ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json", "python")
    source = "def add(left, right):\n    return left + right\n"
    graph = parse_ast_graph(source, "python")
    row = {
        "case_id": "fixture-train",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": "train",
        "ast": serialize_ast_graph(graph),
    }
    rows, report = build_real_corpus_action_dataset(
        [row], {"python": grammar}, {
            "max_prompt_chars": 512, "max_symbol_names": 8, "max_identifiers": 24,
            "max_operators": 12, "max_literals": 12, "max_constructs": 16,
            "max_comment_fragments": 4, "max_comment_chars": 240,
            "max_docstring_chars": 480,
            "minimum_source_line_chars_for_leak_check": 24,
        }
    )
    assert report["status"] == "pass"
    records = [rows[0].as_dict()]
    vocabulary = build_vocabulary(records, {"python": grammar})
    return grammar, encode_prefix_rows(records, {"python": grammar}, vocabulary), vocabulary


def test_joint_masked_losses_are_finite_and_reach_each_head() -> None:
    _grammar, batch, vocabulary = _training_batch()
    model = PrefixTreePolicy(vocabulary, hidden_size=32, layers=1, heads=4, max_steps=batch.action.shape[1])
    output = model(batch)
    total, losses = hierarchical_tree_losses(output, batch)
    assert set(losses) == set(LOSS_NAMES)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in losses.values())

    total.backward()
    for name in (
        "action", "node_kind", "edge_field", "leaf_node", "leaf_class", "copy",
        "copy_length", "lexical", "gap", "node_budget",
    ):
        assert model.heads[name].weight.grad is not None
        assert model.heads[name].weight.grad.abs().sum() > 0
    assert model.copy_query.weight.grad.abs().sum() > 0
    assert model.blueprint.node_head.weight.grad.abs().sum() > 0
    assert model.blueprint.edge_head.weight.grad.abs().sum() > 0
    assert model.blueprint.parent_query.weight.grad.abs().sum() > 0
    assert model.blueprint.presence_head.weight.grad.abs().sum() > 0
    assert model.heads["copy_length"].out_features == 65


def test_future_targets_and_validation_identity_do_not_change_model_inputs() -> None:
    _grammar, batch, vocabulary = _training_batch()
    model = PrefixTreePolicy(vocabulary, hidden_size=32, layers=1, heads=4, max_steps=batch.action.shape[1]).eval()
    changed = replace(
        batch,
        action=torch.flip(batch.action, dims=(1,)),
        node_kind=torch.flip(batch.node_kind, dims=(1,)),
        blueprint_node=torch.flip(batch.blueprint_node, dims=(1,)),
    )
    with torch.no_grad():
        original_output = model(batch)
        changed_output = model(changed)
    for name in ("action", "node_kind", "lexical", "blueprint_memory"):
        torch.testing.assert_close(original_output[name], changed_output[name])

    changed_previous = {name: value.clone() for name, value in batch.previous.items()}
    changed_history = {name: value.clone() for name, value in batch.history.items()}
    future_index = batch.action.shape[1] // 2
    changed_previous["lexical"][:, future_index:] = changed_previous["lexical"][:, future_index:].roll(1, dims=1)
    changed_history["lexical"][:, future_index:] = changed_history["lexical"][:, future_index:].roll(1, dims=1)
    future_changed = replace(batch, previous=changed_previous, history=changed_history)
    with torch.no_grad():
        future_output = model(future_changed)
    torch.testing.assert_close(
        original_output["action"][:, :future_index],
        future_output["action"][:, :future_index],
    )


def test_alignment_masks_action_objectives_and_curriculum_rejects_non_train_rows() -> None:
    _grammar, batch, vocabulary = _training_batch()
    model = PrefixTreePolicy(vocabulary, hidden_size=32, layers=1, heads=4, max_steps=batch.action.shape[1])
    mask = batch.step_mask.clone()
    mask[:, 0] = False
    total, losses = hierarchical_tree_losses(model(batch), batch, aligned_mask=mask)
    assert torch.isfinite(total)
    assert torch.isfinite(losses["action"])
    assert RolloutCurriculum(warmup_steps=2, decay_steps=4).teacher_probability(3) == pytest.approx(0.8125)
    with pytest.raises(ValueError, match="train rows"):
        RolloutCurriculum.require_train_row({"split": "validation"})


def test_viability_negatives_train_the_rollout_viability_head() -> None:
    _grammar, batch, vocabulary = _training_batch()
    model = PrefixTreePolicy(vocabulary, hidden_size=32, layers=1, heads=4, max_steps=batch.action.shape[1])
    target = torch.zeros_like(batch.step_mask, dtype=torch.float32)

    total, losses = hierarchical_tree_losses(model(batch), batch, viability_target=target)

    assert torch.isfinite(total)
    assert losses["viability"] > 0
    total.backward()
    assert model.heads["viability"].weight.grad is not None
    assert model.heads["viability"].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="match rollout action steps"):
        hierarchical_tree_losses(model(batch), batch, viability_target=target[:, :-1])


def test_training_optimizer_consumes_train_rollout_batch(monkeypatch) -> None:
    import newera.nlp_flow.components.generation.policies.prefix.prefix_tree_training as training

    _grammar, batch, vocabulary = _training_batch()
    train_row = {"split": "train", "case_id": "fixture-train", "actions": []}

    def rollout_batch(*_args, **_kwargs):
        target = torch.zeros_like(batch.step_mask, dtype=torch.float32)
        return batch, target, {
            "rollout_rows": 1,
            "successful_rollouts": 0,
            "failed_rollouts": 1,
        }

    monkeypatch.setattr(training, "_build_rollout_batch", rollout_batch)
    _model, metrics = training.train_prefix_tree_policy(
        batch,
        batch,
        vocabulary,
        device=torch.device("cpu"),
        seed=13,
        epochs=1,
        batch_rows=1,
        hidden_size=32,
        layers=1,
        heads=4,
        learning_rate=0.001,
        train_rows=[train_row],
        grammars={"fixture": object()},
        rollout_curriculum=RolloutCurriculum(
            warmup_steps=0,
            decay_steps=1,
            initial_teacher_probability=0.0,
            final_teacher_probability=0.0,
        ),
    )

    assert metrics["training_regime"] == "teacher_forced_plus_train_only_free_running_rollout"
    assert metrics["optimizer"]["learning_rate"] == 0.001
    assert metrics["rollout_training"]["failed_rollouts"] == 1
    assert metrics["rollout_training"]["sampled_actions_used_as_training_inputs"] is True
