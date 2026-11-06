from __future__ import annotations

from dataclasses import replace

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.ast_structure_blueprint import ASTStructureBlueprint, parameter_count
from floorflow.nlp_flow.hierarchical_tree_policy import (
    blueprint_structure_loss,
    encode_blueprint_targets,
)
from floorflow.nlp_flow.prefix_policy_data import encode_prefix_rows
from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy, _loss
from floorflow.nlp_flow.real_corpus_action_data import build_real_corpus_action_dataset
from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary


def _row(source: str) -> dict[str, object]:
    graph = parse_ast_graph(source, "python")
    return {
        "case_id": "python-blueprint",
        "language": "python",
        "source": source,
        "source_sha256": graph.source_sha256,
        "split": "train",
        "ast": serialize_ast_graph(graph),
    }


def _dataset():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    grammar = load_ast_grammar(root / "configs/nlp_ast_transition_grammar_v1.json", "python")
    ast_row = _row("def add(left, right):\n    return left + right\n")
    actions, report = build_real_corpus_action_dataset(
        [ast_row],
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
    action_rows = [actions[0].as_dict()]
    vocabulary = build_vocabulary(action_rows, {"python": grammar})
    return grammar, action_rows, vocabulary


def test_blueprint_targets_follow_preorder_and_teacher_parent_edges() -> None:
    _grammar, rows, vocabulary = _dataset()
    targets = encode_blueprint_targets(rows, vocabulary, max_nodes=32)
    count = int(targets["blueprint_presence"][0].sum())
    actions = rows[0]["actions"]
    open_nodes = [action for action in actions if action["type"] == "open_node"]
    assert count == len(open_nodes)
    assert int(targets["blueprint_parent"][0, 0]) == 0
    assert int(targets["blueprint_node"][0, 0]) == vocabulary["node_ids"][open_nodes[0]["node_kind"]]
    assert torch.all(targets["blueprint_parent"][0, 1:count] < torch.arange(2, count + 1))


def test_blueprint_accepts_first_valid_edge_id_zero() -> None:
    _grammar, rows, vocabulary = _dataset()
    first_edge = next(
        action["edge_field"] for action in rows[0]["actions"] if action["type"] == "attach_edge"
    )
    fields = [first_edge] + [field for field in vocabulary["edge_ids"] if field != first_edge]
    vocabulary["edge_ids"] = {field: index for index, field in enumerate(fields)}

    targets = encode_blueprint_targets(rows, vocabulary, max_nodes=32)

    child_edges = targets["blueprint_edge"][targets["blueprint_parent"].ne(0)]
    assert torch.any(child_edges.eq(0))
    assert targets["blueprint_presence"].sum() > 1


def test_blueprint_planner_shapes_parent_masks_and_causality() -> None:
    planner = ASTStructureBlueprint(
        node_vocab_size=7,
        edge_vocab_size=5,
        hidden_size=16,
        heads=4,
        max_steps=6,
        layers=1,
    ).eval()
    prompt = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    initial = planner(prompt, mask)
    assert initial["memory"].shape == (2, 6, 16)
    assert initial["node_logits"].shape == (2, 6, 8)
    assert initial["edge_logits"].shape == (2, 6, 6)
    assert initial["parent_logits"].shape == (2, 6, 7)
    assert torch.isneginf(initial["parent_logits"][:, 0, 1:]).all()
    assert parameter_count(planner) > 0

    with torch.no_grad():
        planner.query_position[:, 5].add_(17.0)
    changed = planner(prompt, mask)
    for key in ("memory", "node_logits", "edge_logits", "parent_logits", "presence_logits"):
        torch.testing.assert_close(initial[key][:, :5], changed[key][:, :5])


def test_predicted_blueprint_conditions_decoder_without_target_leak_and_trains_all_heads() -> None:
    grammar, rows, vocabulary = _dataset()
    batch = encode_prefix_rows(rows, {"python": grammar}, vocabulary)
    model = PrefixTreePolicy(
        vocabulary,
        hidden_size=32,
        layers=1,
        heads=4,
        max_steps=batch.action.shape[1],
    )
    model.eval()
    output = model(batch)
    changed_targets = replace(batch, blueprint_node=torch.flip(batch.blueprint_node, dims=(1,)))
    without_target_input = model(changed_targets)
    torch.testing.assert_close(output["blueprint_memory"], without_target_input["blueprint_memory"])
    assert output["action"].shape[:2] == batch.action.shape
    assert output["blueprint_memory"].shape == (1, 256, 32)

    model.train()
    training_output = model(batch)
    loss = _loss(training_output, batch)
    loss.backward()
    for module in (
        model.blueprint.planner,
        model.blueprint.node_head,
        model.blueprint.edge_head,
        model.blueprint.parent_query,
        model.blueprint.presence_head,
        model.blueprint.memory_projection,
    ):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert parameter_count(model) > parameter_count(model.blueprint)


def test_structure_loss_rejects_empty_blueprint_supervision() -> None:
    output = {"action": torch.zeros(1, 2, 3)}
    with torch.no_grad():
        empty = torch.zeros(1, 2, dtype=torch.bool)
    try:
        blueprint_structure_loss(output, node=empty.long(), edge=empty.long(), parent=empty.long(), presence=empty)
    except ValueError as error:
        assert "cannot be empty" in str(error)
    else:
        raise AssertionError("empty blueprint target was accepted")
