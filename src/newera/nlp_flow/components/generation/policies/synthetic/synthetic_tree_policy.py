"""Bounded target-free NL-to-AST action policy for a synthetic code domain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
    trace_teacher_actions,
)

ACTIONS = (
    "attach_edge",
    "open_node",
    "open_scope",
    "declare_leaf",
    "close_scope",
    "terminate",
)
MAX_NODES = 32
MAX_PROMPT = 128
MAX_LEXEME = 32


@dataclass(frozen=True)
class SyntheticTreeTask:
    task_id: str
    prompt: str
    function_name: str
    constant: int
    source: str
    actions: tuple[TreeAction, ...]


def build_synthetic_tasks(
    *,
    start: int,
    count: int,
) -> list[SyntheticTreeTask]:
    tasks = []
    for index in range(start, start + count):
        name = f"fn_{index:04d}"
        constant = (index * 7 + 3) % 19
        prompt = (
            f"Write a Python function named {name} taking value "
            f"and returning value plus {constant}."
        )
        source = f"def {name}(value):\n    return value + {constant}\n"
        payload = json.loads(serialize_ast_graph(parse_ast_graph(source, "python")))
        actions = tuple(trace_teacher_actions(_python_grammar(), payload))
        tasks.append(
            SyntheticTreeTask(
                task_id=f"tree-{index:05d}",
                prompt=prompt,
                function_name=name,
                constant=constant,
                source=source,
                actions=actions,
            )
        )
    return tasks


@lru_cache(maxsize=1)
def _python_grammar() -> ASTTransitionGrammar:
    from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar

    return load_ast_grammar(
        "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_v1.json",
        "python",
    )


def _action_fields(action: TreeAction) -> dict[str, int]:
    parent = (
        action.parent_index
        if isinstance(action, AttachEdgeAction)
        else -1
    )
    leaf_node = (
        action.node_index
        if isinstance(action, DeclareLeafAction)
        else -1
    )
    return {
        "parent": parent,
        "leaf_node": leaf_node,
    }


def _state_features(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
) -> list[float]:
    open_parents = [
        index
        for index, node in enumerate(decoder.nodes)
        if node.scope_id in decoder.state.open_scopes
    ]
    can_attach = any(
        any(
            transition.parent_kind == decoder.nodes[index].kind
            for transition in grammar.transitions
        )
        for index in open_parents
    )
    can_leaf = any(
        not node.children
        and node.leaf_class is None
        and bool(grammar.leaf_classes.get(node.kind))
        for node in decoder.nodes
    )
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    return [
        len(decoder.nodes) / MAX_NODES,
        len(decoder.state.open_scopes) / 8.0,
        float(decoder.expected_scope is not None),
        float(decoder.pending_edge is not None),
        float(not decoder.nodes),
        float(can_attach),
        float(can_leaf),
        float(bool(decoder.state.open_scopes)),
        float(decoder.state.eos_ready and not undeclared),
    ]


def _categories(values):
    ordered = sorted(set(values))
    return ordered, {value: index for index, value in enumerate(ordered)}


def _build_vocabulary(tasks: list[SyntheticTreeTask]) -> dict[str, object]:
    actions = [action for task in tasks for action in task.actions]
    characters = sorted({character for task in tasks for character in task.prompt})
    node_names, node_ids = _categories(
        action.node_kind
        for action in actions
        if isinstance(action, OpenNodeAction)
    )
    edge_names, edge_ids = _categories(
        action.edge_field
        for action in actions
        if isinstance(action, AttachEdgeAction)
    )
    class_names, class_ids = _categories(
        action.leaf_class
        for action in actions
        if isinstance(action, DeclareLeafAction)
    )
    lexeme_names, lexeme_ids = _categories(
        action.lexical
        for action in actions
        if isinstance(action, DeclareLeafAction)
        and action.lexical is not None
    )
    gap_names, gap_ids = _categories(
        action.gap_before
        for action in actions
        if isinstance(action, DeclareLeafAction)
    )
    return {
        "chars": {value: index + 1 for index, value in enumerate(characters)},
        "action_ids": {value: index for index, value in enumerate(ACTIONS)},
        "node_names": node_names,
        "node_ids": node_ids,
        "edge_names": edge_names,
        "edge_ids": edge_ids,
        "class_names": class_names,
        "class_ids": class_ids,
        "lexeme_names": lexeme_names,
        "lexeme_ids": lexeme_ids,
        "gap_names": gap_names,
        "gap_ids": gap_ids,
    }


def _action_previous_fields(action: TreeAction | None) -> dict[str, str]:
    if action is None:
        return {
            "action": "<start>",
            "node": "<none>",
            "edge": "<none>",
            "leaf_class": "<none>",
            "lexical": "<none>",
            "gap": "<none>",
        }
    if isinstance(action, AttachEdgeAction):
        return {
            "action": "attach_edge",
            "node": "<none>",
            "edge": action.edge_field,
            "leaf_class": "<none>",
            "lexical": "<none>",
            "gap": "<none>",
        }
    if isinstance(action, OpenNodeAction):
        return {
            "action": "open_node",
            "node": action.node_kind,
            "edge": "<none>",
            "leaf_class": "<none>",
            "lexical": "<none>",
            "gap": "<none>",
        }
    if isinstance(action, DeclareLeafAction):
        return {
            "action": "declare_leaf",
            "node": "<none>",
            "edge": "<none>",
            "leaf_class": action.leaf_class,
            "lexical": action.lexical or "<none>",
            "gap": action.gap_before,
        }
    if isinstance(action, OpenScopeAction):
        name = "open_scope"
    elif isinstance(action, CloseScopeAction):
        name = "close_scope"
    else:
        name = "terminate"
    return {
        "action": name,
        "node": "<none>",
        "edge": "<none>",
        "leaf_class": "<none>",
        "lexical": "<none>",
        "gap": "<none>",
    }


def _copy_span(prompt: str, lexical: str) -> tuple[int, int] | None:
    if not lexical:
        return None
    position = prompt.find(lexical)
    if position < 0 or len(lexical) > MAX_LEXEME:
        return None
    return position, len(lexical)


@dataclass(frozen=True)
class PolicyBatch:
    prompt: torch.Tensor
    prompt_mask: torch.Tensor
    state_features: torch.Tensor
    previous: dict[str, torch.Tensor]
    action: torch.Tensor
    node_kind: torch.Tensor
    edge_field: torch.Tensor
    parent: torch.Tensor
    leaf_node: torch.Tensor
    leaf_class: torch.Tensor
    copy_label: torch.Tensor
    copy_start: torch.Tensor
    copy_length: torch.Tensor
    lexical: torch.Tensor
    gap: torch.Tensor
    step_mask: torch.Tensor


def encode_tasks(
    tasks: list[SyntheticTreeTask],
    vocabulary: dict[str, object],
) -> PolicyBatch:
    grammar = _python_grammar()
    max_steps = max(len(task.actions) for task in tasks) + 2
    prompts = []
    prompt_masks = []
    state_rows = []
    previous_rows = {
        name: []
        for name in ("action", "node", "edge", "leaf_class", "lexical", "gap")
    }
    labels = {
        name: []
        for name in (
            "action",
            "node_kind",
            "edge_field",
            "parent",
            "leaf_node",
            "leaf_class",
            "copy_label",
            "copy_start",
            "copy_length",
            "lexical",
            "gap",
        )
    }
    step_masks = []
    unknown = 0
    for task in tasks:
        char_ids = [
            int(vocabulary["chars"].get(character, unknown))
            for character in task.prompt[:MAX_PROMPT]
        ]
        prompts.append(char_ids + [unknown] * (MAX_PROMPT - len(char_ids)))
        prompt_masks.append(
            [True] * len(char_ids)
            + [False] * (MAX_PROMPT - len(char_ids))
        )
        decoder = TreeActionDecoder(grammar)
        previous: TreeAction | None = None
        for step in range(max_steps):
            state_rows.append(_state_features(decoder, grammar))
            prev_fields = _action_previous_fields(previous)
            for name, value in prev_fields.items():
                mapping = vocabulary[
                    {
                        "action": "action_ids",
                        "node": "node_ids",
                        "edge": "edge_ids",
                        "leaf_class": "class_ids",
                        "lexical": "lexeme_ids",
                        "gap": "gap_ids",
                    }[name]
                ]
                previous_rows[name].append(int(mapping.get(value, 0)))
            action = None if step >= len(task.actions) else task.actions[step]
            if action is None:
                labels["action"].append(0)
                for name in (
                    "node_kind",
                    "edge_field",
                    "parent",
                    "leaf_node",
                    "leaf_class",
                    "copy_label",
                    "copy_start",
                    "copy_length",
                    "lexical",
                    "gap",
                ):
                    labels[name].append(0)
                step_masks.append(0)
                continue
            step_masks.append(1)
            action_name = {
                AttachEdgeAction: "attach_edge",
                OpenNodeAction: "open_node",
                OpenScopeAction: "open_scope",
                DeclareLeafAction: "declare_leaf",
                CloseScopeAction: "close_scope",
                TerminateAction: "terminate",
            }[type(action)]
            labels["action"].append(ACTIONS.index(action_name))
            labels["node_kind"].append(
                int(vocabulary["node_ids"].get(
                    action.node_kind if isinstance(action, OpenNodeAction) else "<none>",
                    0,
                ))
            )
            labels["edge_field"].append(
                int(vocabulary["edge_ids"].get(
                    action.edge_field if isinstance(action, AttachEdgeAction) else "<none>",
                    0,
                ))
            )
            labels["parent"].append(
                max(0, action.parent_index)
                if isinstance(action, AttachEdgeAction)
                else 0
            )
            labels["leaf_node"].append(
                action.node_index if isinstance(action, DeclareLeafAction) else 0
            )
            labels["leaf_class"].append(
                int(vocabulary["class_ids"].get(
                    action.leaf_class if isinstance(action, DeclareLeafAction) else "<none>",
                    0,
                ))
            )
            lexical = action.lexical if isinstance(action, DeclareLeafAction) else None
            span = _copy_span(task.prompt, lexical or "")
            labels["copy_label"].append(int(isinstance(action, DeclareLeafAction) and span is not None))
            labels["copy_start"].append(span[0] if span else 0)
            labels["copy_length"].append(span[1] if span else 0)
            labels["lexical"].append(
                int(vocabulary["lexeme_ids"].get(lexical or "<none>", 0))
            )
            labels["gap"].append(
                int(vocabulary["gap_ids"].get(
                    action.gap_before if isinstance(action, DeclareLeafAction) else "",
                    0,
                ))
            )
            decoder.apply(action)
            previous = action
    return PolicyBatch(
        prompt=torch.tensor(prompts, dtype=torch.long),
        prompt_mask=torch.tensor(prompt_masks, dtype=torch.bool),
        state_features=torch.tensor(
            state_rows,
            dtype=torch.float32,
        ).view(len(tasks), max_steps, 9),
        previous={
            name: torch.tensor(values, dtype=torch.long).view(
                len(tasks),
                max_steps,
            )
            for name, values in previous_rows.items()
        },
        action=torch.tensor(labels["action"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        node_kind=torch.tensor(labels["node_kind"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        edge_field=torch.tensor(
            labels["edge_field"],
            dtype=torch.long,
        ).view(len(tasks), max_steps),
        parent=torch.tensor(labels["parent"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        leaf_node=torch.tensor(labels["leaf_node"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        leaf_class=torch.tensor(
            labels["leaf_class"],
            dtype=torch.long,
        ).view(len(tasks), max_steps),
        copy_label=torch.tensor(
            labels["copy_label"],
            dtype=torch.float32,
        ).view(len(tasks), max_steps),
        copy_start=torch.tensor(
            labels["copy_start"],
            dtype=torch.long,
        ).view(len(tasks), max_steps),
        copy_length=torch.tensor(
            labels["copy_length"],
            dtype=torch.long,
        ).view(len(tasks), max_steps),
        lexical=torch.tensor(labels["lexical"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        gap=torch.tensor(labels["gap"], dtype=torch.long).view(
            len(tasks),
            max_steps,
        ),
        step_mask=torch.tensor(step_masks, dtype=torch.bool).view(
            len(tasks),
            max_steps,
        ),
    )


__all__ = [
    "ACTIONS",
    "SyntheticTreeTask",
    "build_synthetic_tasks",
    "encode_tasks",
]
