from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

from newera.nlp_flow.components.ast.flow.core.ast_cfm import (
    AST_LEVELS,
    ASTCodecVocabulary,
    ASTStageAFlow,
    ast_cfm_objective_losses,
    collate_ast_batches,
    decode_generated_source,
    decode_generated_template,
    decode_generated_typed_tree,
    encode_ast_graph,
    prompt_context,
)
from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast, render_source
from newera.nlp_flow.components.ast.diagnostics.tracing.ast_generation import decode_structured_ast
from newera.nlp_flow.components.generation.trees.typed.typed_tree import render_typed_tree


def _batch():
    graphs = [
        parse_ast("def add(a, b):\n    return a + b\n", "python"),
        parse_ast("fn add(a: i32, b: i32) -> i32 { a + b }\n", "rust"),
    ]
    vocabulary = ASTCodecVocabulary.from_graphs(graphs)
    return collate_ast_batches(
        [
            encode_ast_graph(graph, vocabulary, max_nodes=48, template_id=index)
            for index, graph in enumerate(graphs)
        ]
    ), vocabulary


def test_ast_cfm_objectives_and_all_heads_receive_gradients() -> None:
    torch.manual_seed(11)
    batch, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    )
    target_state = model.embed_target(batch, "cpu")
    noise = torch.randn_like(target_state)
    time = torch.tensor([0.25, 0.75])
    state = (1 - time[:, None, None]) * noise + time[:, None, None] * target_state
    context = torch.randn(2, 32)

    output = model.forward(
        state,
        time,
        context,
        batch,
        source_token_inputs=batch.source_token_ids[:, :-1],
        source_token_node_ids=batch.source_token_node_ids[:, :-1],
        source_token_phase_ids=batch.source_token_phase_ids[:, :-1],
    )
    losses = ast_cfm_objective_losses(
        output,
        target_state,
        noise,
        batch,
    )
    losses.total.backward()

    assert set(output.level_velocities) == set(AST_LEVELS)
    assert output.source_token_logits is not None
    assert output.source_token_free_logits is not None
    assert output.source_token_logits.shape[:2] == (
        2,
        batch.source_token_ids.shape[1] - 1,
    )
    for name in AST_LEVELS:
        assert output.level_velocities[name].shape == output.velocity.shape
    for value in losses.as_dict().values():
        assert torch.isfinite(value)
    for head in (
        model.velocity_head,
        model.categorical_conditioner[-1],
        model.parent_context_conditioner[-1],
        model.node_type_head,
        model.edge_type_head,
        model.lexical_head,
        model.gap_head,
        model.level_head,
        model.parent_query,
        model.parent_key,
        model.root_head,
        model.node_count_head[-1],
        model.source_length_head,
        model.template_head,
        model.parseability_head,
        model.token_endpoint.output,
    ):
        assert head.weight.grad is not None


def test_ast_vocabulary_retains_language_specific_structure() -> None:
    _, vocabulary = _batch()

    assert set(vocabulary.node_types_by_language) == {"python", "rust"}
    assert (
        vocabulary.node_types_by_language["python"]
        != vocabulary.node_types_by_language["rust"]
    )
    assert "module" in vocabulary.node_types_by_language["python"]
    assert "source_file" not in vocabulary.node_types_by_language["python"]


def test_sampling_path_is_target_free_and_uses_hierarchical_masks() -> None:
    torch.manual_seed(13)
    batch, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    ).eval()
    first = batch.node_mask[0]
    context = prompt_context("implement addition", 32)[None]
    noise = torch.randn(1, 48, 32)

    output = model.sample(
        context,
        first[None],
        steps=4,
        noise=noise,
    )

    signature = inspect.signature(model.sample)
    assert "target" not in signature.parameters
    assert not any("target" in name for name in signature.parameters)
    assert output.endpoint.shape == noise.shape
    assert output.parent_logits.shape == (1, 48, 49)


def test_target_free_sampling_predicts_its_own_node_budget() -> None:
    torch.manual_seed(19)
    _, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    ).eval()
    context = torch.randn(2, 32)

    output = model.sample_target_free(
        context,
        steps=2,
        noise=torch.randn(2, 48, 32),
    )
    node_mask = output.predicted_node_mask()
    signature = inspect.signature(model.sample_target_free)

    assert output.node_count_logits.shape == (2, 49)
    assert output.source_length_logits.shape == (2, 98)
    assert output.source_token_ids is not None
    assert output.source_token_ids.shape == (2, 97)
    assert node_mask.shape == (2, 48)
    assert node_mask.any(dim=1).all()
    assert "node_mask" not in signature.parameters
    assert not any("target" in name for name in signature.parameters)


def test_endpoint_uses_remaining_time_instead_of_full_velocity() -> None:
    torch.manual_seed(12)
    batch, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    )
    state = torch.randn(2, 48, 32)
    time = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 32)

    output = model(state, time, context, batch)

    assert torch.allclose(
        output.flow_endpoint,
        state + (1.0 - time)[:, None, None] * output.velocity,
    )
    assert not torch.equal(output.endpoint, output.flow_endpoint)
    with pytest.raises(ValueError, match="shortcuts are disabled"):
        decode_generated_template(output, vocabulary)
    assert isinstance(
        decode_generated_template(
            output,
            vocabulary,
            allow_template_shortcut=True,
        ),
        str,
    )


def test_gap_and_lexical_heads_reconstruct_the_full_token_endpoint_source() -> None:
    torch.manual_seed(14)
    source = "def add(a, b):\n    return a + b\n"
    graph = parse_ast(source, "python")
    vocabulary = ASTCodecVocabulary.from_graphs((graph,))
    batch = collate_ast_batches(
        [
            encode_ast_graph(
                graph,
                vocabulary,
                max_nodes=48,
                template_id=0,
            )
        ]
    )
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    ).eval()
    target = model.embed_target(batch, "cpu")
    with torch.no_grad():
        output = model(
            target,
            torch.ones(1),
            torch.randn(1, 32),
            batch,
        )
        output = replace(
            output,
            gap_logits=torch.nn.functional.one_hot(
                batch.gaps,
                vocabulary.num_gaps,
            ).float(),
            lexical_logits=torch.nn.functional.one_hot(
                batch.lexemes,
                vocabulary.num_lexemes,
            ).float(),
        )
        decoded = decode_generated_source(
            output,
            batch.node_mask,
            vocabulary,
        )
        typed_tree, typed_source = decode_generated_typed_tree(
            output,
            batch.node_mask,
            vocabulary,
            "python",
        )

    assert decoded == render_source(graph)
    assert typed_tree.domain == "ast"
    assert typed_source == decoded
    assert render_typed_tree(typed_tree) == decoded


def test_structure_aware_decoder_uses_node_fields_and_parent_tree() -> None:
    torch.manual_seed(23)
    source = "def add(a, b):\n    return a + b\n"
    graph = parse_ast(source, "python")
    vocabulary = ASTCodecVocabulary.from_graphs((graph,))
    batch = collate_ast_batches(
        [encode_ast_graph(graph, vocabulary, max_nodes=48, template_id=0)]
    )
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    ).eval()
    target = model.embed_target(batch, "cpu")
    with torch.no_grad():
        output = model(target, torch.ones(1), torch.randn(1, 32), batch)
        output = replace(
            output,
            node_type_logits=torch.nn.functional.one_hot(
                batch.node_types,
                len(vocabulary.node_types),
            ).float(),
            edge_type_logits=torch.nn.functional.one_hot(
                batch.edge_fields,
                len(vocabulary.edge_fields),
            ).float(),
            lexical_logits=torch.nn.functional.one_hot(
                batch.lexemes,
                vocabulary.num_lexemes,
            ).float(),
            gap_logits=torch.nn.functional.one_hot(
                batch.gaps,
                vocabulary.num_gaps,
            ).float(),
            level_logits=torch.nn.functional.one_hot(
                batch.levels.clamp_max(2),
                3,
            ).float(),
            parent_logits=torch.nn.functional.one_hot(
                batch.parent_indices + 1,
                49,
            ).float(),
            root_logits=torch.where(
                batch.parent_indices.eq(-1),
                10.0,
                -10.0,
            ),
        )
        decoded = decode_structured_ast(
            output,
            vocabulary,
            "python",
            node_mask=batch.node_mask,
        )

    assert decoded.source == source
    assert decoded.node_count == int(batch.node_mask.sum())


def test_optional_planner_level_masks_fail_closed_when_invalid() -> None:
    batch, vocabulary = _batch()
    model = ASTStageAFlow(
        vocabulary,
        hidden_size=32,
        num_heads=4,
        max_nodes=48,
    )
    node_mask = batch.node_mask[:1]
    level_masks = {
        name: batch.level_masks[name][:1]
        for name in AST_LEVELS
    }
    level_masks["leaf"] = level_masks["leaf"] | level_masks["subtree"]

    try:
        model.sample(
            torch.randn(1, 32),
            node_mask,
            level_masks=level_masks,
            noise=torch.randn(1, 48, 32),
        )
    except ValueError as error:
        assert "partition" in str(error)
    else:
        raise AssertionError("overlapping level masks were accepted")
