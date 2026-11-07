from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.research.architecture.experts.efficient_output_heads import (
    FactorizedClassHead,
    apply_legal_candidate_mask,
    rebase_residual_budget_logits,
)
from newera.nlp_flow.components.generation.trees.hierarchical.hierarchical_tree_training import hierarchical_tree_losses
from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import audit_prefix_batch, encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_action_data import build_real_corpus_action_dataset
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import REAL_ACTIONS
from newera.nlp_flow.components.generation.policies.vocabulary.real_corpus_vocabulary import build_vocabulary
from newera.nlp_flow.components.generation.trees.runtime.tree_completion_runtime import load_tree_completion_budget

ROOT = Path(__file__).resolve().parents[4]
GRAMMAR = ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json"


def _training_batch(*, with_budget_minimum: bool = False):
    source = "def add(left, right):\n    return left + right\n"
    grammar = load_ast_grammar(GRAMMAR, "python")
    graph = parse_ast_graph(source, "python")
    row = {
        "case_id": "factorized-head-train",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": "train",
        "ast": serialize_ast_graph(graph),
    }
    rows, report = build_real_corpus_action_dataset(
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
    records = [rows[0].as_dict()]
    vocabulary = build_vocabulary(records, {"python": grammar})
    budget_minimums = None
    if with_budget_minimum:
        schema = ROOT / ".data/nlp/parser_schema_v1/python-node-types.json"
        completion = load_tree_completion_budget(
            grammar,
            schema,
            expected_schema_sha256=hashlib.sha256(schema.read_bytes()).hexdigest(),
        )
        budget_minimums = {"python": completion.minimum_root_nodes()}
    return (
        encode_prefix_rows(
            records,
            {"python": grammar},
            vocabulary,
            budget_minimums=budget_minimums,
        ),
        vocabulary,
    )


def test_factorized_head_reduces_parameters_and_keeps_class_logits() -> None:
    hidden_size, class_count, rank = 128, 324, 32
    head = FactorizedClassHead(hidden_size, class_count, rank)
    baseline_parameters = nn.Linear(hidden_size, class_count).weight.numel() + class_count

    logits = head(torch.randn(2, 5, hidden_size))

    assert logits.shape == (2, 5, class_count)
    assert head.parameter_count < baseline_parameters
    with pytest.raises(ValueError, match="rank must be positive"):
        FactorizedClassHead(hidden_size, class_count, hidden_size)


def test_legal_candidate_mask_fails_closed_and_preserves_target_gradient() -> None:
    logits = torch.tensor([[[1.0, 4.0, 3.0]]], requires_grad=True)
    legal = torch.tensor([[[True, False, True]]])
    active = torch.tensor([[True]])

    masked = apply_legal_candidate_mask(logits, legal, active, torch.tensor([[2]]))
    loss = nn.functional.cross_entropy(masked.reshape(1, 3), torch.tensor([2]))
    loss.backward()

    assert torch.isneginf(masked[0, 0, 1])
    assert logits.grad is not None and logits.grad[0, 0, 2] < 0
    with pytest.raises(ValueError, match="outside the grammar candidate set"):
        apply_legal_candidate_mask(logits, legal, active, torch.tensor([[1]]))
    with pytest.raises(ValueError, match="no legal grammar candidate"):
        apply_legal_candidate_mask(logits, torch.zeros_like(legal), active)


def test_residual_budget_scores_cannot_choose_below_certified_floor() -> None:
    residual = torch.tensor([[7.0, 9.0, 8.0, 6.0]])
    minimum = torch.tensor([2])

    absolute = rebase_residual_budget_logits(residual, minimum, torch.tensor([3]))

    assert absolute.tolist()[0][:2] == [float("-inf"), float("-inf")]
    assert absolute.argmax(dim=-1).item() == 3
    with pytest.raises(ValueError, match="below its certified grammar minimum"):
        rebase_residual_budget_logits(residual, minimum, torch.tensor([1]))


def test_factorized_node_head_uses_teacher_and_runtime_grammar_masks() -> None:
    batch, vocabulary = _training_batch()
    audit = audit_prefix_batch(batch)
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=64,
        layers=1,
        heads=4,
        max_steps=batch.action.shape[1],
        blueprint_conditioning=False,
        node_kind_rank=16,
    )
    logits = model(batch)["node_kind"]
    open_steps = batch.action.eq(REAL_ACTIONS.index("open_node")) & batch.step_mask
    target_is_finite = logits[open_steps].gather(
        1,
        batch.node_kind[open_steps, None],
    ).isfinite().all()
    illegal = open_steps[..., None] & ~batch.node_kind_legal_mask

    assert audit["status"] == "pass"
    assert target_is_finite
    assert torch.isneginf(logits[illegal]).all()
    assert logits.shape[-1] == len(vocabulary["node_ids"]) + 1
    assert model.heads["node_kind"].parameter_count < (
        64 * (len(vocabulary["node_ids"]) + 1) + len(vocabulary["node_ids"]) + 1
    )
    total, _ = hierarchical_tree_losses(model(batch), batch)
    total.backward()
    assert model.heads["node_kind"].query_projection.weight.grad.abs().sum() > 0
    assert model.heads["node_kind"].class_embeddings.grad.abs().sum() > 0

    invalid_mask = batch.node_kind_legal_mask.clone()
    rows, steps = open_steps.nonzero(as_tuple=True)
    invalid_mask[rows, steps, batch.node_kind[rows, steps]] = False
    with pytest.raises(ValueError):
        model(replace(batch, node_kind_legal_mask=invalid_mask))


@pytest.mark.skip(reason="Requires the external legacy parser schema under .data/nlp/parser_schema_v1/.")
def test_budget_residual_mode_requires_contract_minimum_and_masks_impossible_sizes() -> None:
    batch, vocabulary = _training_batch(with_budget_minimum=True)
    minimum = int(batch.node_budget_minimum[0])
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=64,
        layers=1,
        heads=4,
        max_steps=batch.action.shape[1],
        blueprint_conditioning=False,
        node_kind_rank=16,
        budget_residual=True,
    )

    output = model(batch)

    assert minimum == 1
    assert torch.isneginf(output["node_budget"][0, :minimum]).all()
    assert torch.isfinite(output["node_budget"][0, batch.node_budget[0]])
    with pytest.raises(ValueError, match="requires certified per-example minimums"):
        model(replace(batch, node_budget_minimum=None))
