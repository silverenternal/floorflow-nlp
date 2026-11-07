from __future__ import annotations

from dataclasses import replace

import pytest

from newera.nlp_flow.components.ast.representation.codec.ast_codec import ASTGraph, ASTNode, parse_ast_graph
from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf
from newera.nlp_flow.components.generation.trees.typed.typed_lexical_contract import TypedLexicalValue
from newera.nlp_flow.components.generation.trees.typed.typed_source_renderer import (
    InvalidLexicalDecisionError,
    MissingLexicalDecisionError,
    SourceReplayMismatchError,
    render_typed_source,
)

FIXTURES = (
    (
        "python",
        '# π comment\ndef combine(left, right):\n    """Escaped: π\\n"""\n    return left + right * 2\n',
    ),
    (
        "rust",
        '// Unicode π\npub fn choose<T>(left: Option<T>, right: T) -> T { right }\n',
    ),
    (
        "cpp",
        'template <typename T>\nstruct Box { T value; };\nconst char* text() { return "π\\n"; }\n',
    ),
)


def _typed_values(graph: ASTGraph) -> list[TypedLexicalValue]:
    values = []
    node_id = 0

    def visit(node: ASTNode) -> None:
        nonlocal node_id
        current_id = node_id
        node_id += 1
        if node.children:
            for _field, child in node.children:
                visit(child)
            return
        assert node.lexical_value is not None
        values.append(
            TypedLexicalValue(
                node_id=current_id,
                node_kind=node.node_type,
                leaf_class=classify_leaf(node.node_type, node.lexical_value),
                scope_id=node.scope_id,
                value=node.lexical_value,
                origin="generated",
                minimum_length=0,
                maximum_length=max(1, len(node.lexical_value)),
            )
        )

    visit(graph.root)
    return values


@pytest.mark.parametrize(("language", "source"), FIXTURES)
def test_typed_ast_renders_and_replays_multilanguage_source(language: str, source: str) -> None:
    graph = parse_ast_graph(source, language)
    rendered = render_typed_source(graph, _typed_values(graph))
    assert rendered.language == language
    assert rendered.source == source
    assert len(rendered.source_sha256) == 64
    assert rendered.canonical_signature[0] == graph.root.node_type


def test_renderer_rejects_missing_extra_and_wrong_node_decisions() -> None:
    graph = parse_ast_graph("value = left + right\n", "python")
    values = _typed_values(graph)
    with pytest.raises(MissingLexicalDecisionError, match="missing"):
        render_typed_source(graph, values[:-1])
    with pytest.raises(InvalidLexicalDecisionError, match="extra"):
        render_typed_source(graph, values + [replace(values[-1], node_id=999)])
    with pytest.raises(InvalidLexicalDecisionError, match="does not match"):
        render_typed_source(graph, [replace(value, node_kind="string") if value.node_kind == "identifier" else value for value in values])


def test_renderer_does_not_repair_changed_structure_or_operators() -> None:
    graph = parse_ast_graph("def add(left, right):\n    return left + right\n", "python")
    values = _typed_values(graph)
    plus = next(index for index, value in enumerate(values) if value.value == "+")
    values[plus] = replace(values[plus], value="*")
    with pytest.raises(SourceReplayMismatchError, match="changed the committed AST"):
        render_typed_source(graph, values)

