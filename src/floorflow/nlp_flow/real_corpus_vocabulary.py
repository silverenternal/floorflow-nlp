"""Vocabulary builder for real-corpus tree-action policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.real_corpus_policy_data import (
    REAL_ACTIONS,
    action_from_record,
)
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TreeActionDecoder,
)


def _category(values):
    ordered = sorted(set(values))
    return ordered, {value: index for index, value in enumerate(ordered)}


def build_vocabulary(
    train_rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
) -> dict[str, object]:
    actions = [
        action_from_record(record)
        for row in train_rows
        for record in row["actions"]
    ]
    grammar_values = list(grammars.values())
    node_names, node_ids = _category(
        [
            value.node_kind
            for value in actions
            if isinstance(value, OpenNodeAction)
        ]
        + [
            kind
            for grammar in grammar_values
            for kind in grammar.legal_root_kinds
        ]
        + [
            value
            for grammar in grammar_values
            for transition in grammar.transitions
            for value in (transition.parent_kind, transition.child_kind)
        ]
    )
    edge_names, edge_ids = _category(
        [
            value.edge_field
            for value in actions
            if isinstance(value, AttachEdgeAction)
        ]
        + [
            transition.edge_field
            for grammar in grammar_values
            for transition in grammar.transitions
        ]
    )
    class_names, class_ids = _category(
        [
            value.leaf_class
            for value in actions
            if isinstance(value, DeclareLeafAction)
        ]
        + [
            leaf_class
            for grammar in grammar_values
            for values in grammar.leaf_classes.values()
            for leaf_class in values
        ]
    )
    lexeme_names, lexeme_ids = _category(
        value.lexical
        for value in actions
        if isinstance(value, DeclareLeafAction)
        and value.lexical is not None
    )
    gap_names, gap_ids = _category(
        value.gap_before
        for value in actions
        if isinstance(value, DeclareLeafAction)
    )
    scope_names, scope_ids = _category(
        [
            value.scope_kind
            for value in actions
            if isinstance(value, OpenScopeAction)
        ]
        + [
            transition.parent_scope_kind
            for grammar in grammar_values
            for transition in grammar.transitions
        ]
        + [
            transition.child_scope_kind
            for grammar in grammar_values
            for transition in grammar.transitions
        ]
    )
    characters = sorted(
        {
            character
            for row in train_rows
            for character in str(row["prompt"])
        }
    )
    lexemes_by_kind_class: dict[str, set[int]] = {}
    gaps_by_kind: dict[str, set[int]] = {}
    for row in train_rows:
        decoder = TreeActionDecoder(grammars[str(row["language"])])
        for record in row["actions"]:
            action = action_from_record(record)
            if isinstance(action, DeclareLeafAction) and action.lexical is not None:
                node_kind = decoder.nodes[action.node_index].kind
                key = f"{node_kind}|{action.leaf_class}"
                lexemes_by_kind_class.setdefault(key, set()).add(
                    lexeme_ids[action.lexical]
                )
                gaps_by_kind.setdefault(node_kind, set()).add(
                    gap_ids[action.gap_before]
                )
            decoder.apply(action)
    return {
        "action_ids": {value: index for index, value in enumerate(REAL_ACTIONS)},
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "class_ids": class_ids,
        "lexeme_ids": lexeme_ids,
        "gap_ids": gap_ids,
        "node_names": node_names,
        "edge_names": edge_names,
        "class_names": class_names,
        "lexeme_names": lexeme_names,
        "gap_names": gap_names,
        "scope_names": scope_names,
        "scope_ids": scope_ids,
        "chars": {value: index + 1 for index, value in enumerate(characters)},
        "lexemes_by_kind_class": {
            key: sorted(values)
            for key, values in lexemes_by_kind_class.items()
        },
        "gaps_by_kind": {
            key: sorted(values)
            for key, values in gaps_by_kind.items()
        },
    }


__all__ = ["build_vocabulary"]
