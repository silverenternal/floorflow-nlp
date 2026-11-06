"""Event-conditioned tensors for the real-corpus tree-action policy."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
)

REAL_ACTIONS = (
    "attach_edge",
    "open_node",
    "open_scope",
    "declare_leaf",
    "close_scope",
    "terminate",
)
MAX_PROMPT = 512
MAX_NODES = 256
MAX_LEXEME = 64
STATE_FEATURES = 18
MAX_PARENT_CANDIDATES = 64
MAX_PROMPT_TOKENS = 192
PROMPT_TOKEN_BUCKETS = 32768
ACTION_HISTORY = 8


@dataclass(frozen=True)
class RealPolicyBatch:
    prompts: torch.Tensor
    prompt_masks: torch.Tensor
    prompt_token_ids: torch.Tensor
    prompt_token_mask: torch.Tensor
    event_prompt_rows: torch.Tensor
    state_features: torch.Tensor
    context_kind: torch.Tensor
    context_edge: torch.Tensor
    candidate_kind: torch.Tensor
    candidate_scope: torch.Tensor
    candidate_position: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_target: torch.Tensor
    node_budget: torch.Tensor
    history: dict[str, torch.Tensor]
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


def action_from_record(record: Mapping[str, object]) -> TreeAction:
    action_type = str(record["type"])
    if action_type == "attach_edge":
        return AttachEdgeAction(
            int(record["parent_index"]),
            str(record["edge_field"]),
        )
    if action_type == "open_node":
        return OpenNodeAction(str(record["node_kind"]))
    if action_type == "open_scope":
        return OpenScopeAction(
            int(record["scope_id"]),
            int(record["node_index"]),
            str(record["scope_kind"]),
        )
    if action_type == "declare_leaf":
        lexical = record.get("lexical")
        return DeclareLeafAction(
            int(record["node_index"]),
            str(record["leaf_class"]),
            str(lexical) if isinstance(lexical, str) else None,
            str(record.get("gap_before", "")),
        )
    if action_type == "close_scope":
        return CloseScopeAction(int(record["scope_id"]))
    if action_type == "terminate":
        return TerminateAction()
    raise ValueError(f"unknown real tree action: {action_type}")


def load_action_rows(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or "actions" not in value:
                raise TypeError(f"invalid action row {line_number}")
            rows.append(value)
    return rows


def state_features(decoder: TreeActionDecoder) -> list[float]:
    open_parents = [
        index
        for index, node in enumerate(decoder.nodes)
        if node.scope_id in decoder.state.open_scopes
    ]
    transitions = decoder.grammar.transitions
    can_attach = any(
        any(value.parent_kind == decoder.nodes[index].kind for value in transitions)
        for index in open_parents
    )
    leaves = [
        node
        for node in decoder.nodes
        if not node.children
        and node.leaf_class is None
        and bool(decoder.grammar.leaf_classes.get(node.kind))
    ]
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    action_counts = [
        sum(
            isinstance(action, action_type)
            for action in decoder.actions
        ) / max(1, len(decoder.actions))
        for action_type in (
            AttachEdgeAction,
            OpenNodeAction,
            OpenScopeAction,
            DeclareLeafAction,
            CloseScopeAction,
            TerminateAction,
        )
    ]
    return [
        len(decoder.nodes) / MAX_NODES,
        len(decoder.state.open_scopes) / 32.0,
        float(decoder.expected_scope is not None),
        float(decoder.pending_edge is not None),
        float(not decoder.nodes),
        float(can_attach),
        float(bool(leaves)),
        float(bool(decoder.state.open_scopes)),
        float(decoder.state.eos_ready and not undeclared),
        len(leaves) / MAX_NODES,
        len(decoder.actions) / (MAX_NODES * 8),
        float(decoder.terminated),
    ] + action_counts


def previous_fields(action: TreeAction | None) -> dict[str, str]:
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
    if not lexical or len(lexical) > MAX_LEXEME:
        return None
    position = prompt.find(lexical)
    if position < 0 or position >= MAX_PROMPT:
        return None
    return position, len(lexical)


def prompt_token_ids(prompt: str) -> list[int]:
    tokens = re.findall(
        r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]",
        prompt[:MAX_PROMPT],
    )[:MAX_PROMPT_TOKENS]
    return [
        1 + int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        % PROMPT_TOKEN_BUCKETS
        for token in tokens
    ]


def encode_action_rows(
    rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
    vocabulary: Mapping[str, object],
) -> RealPolicyBatch:
    parent_kinds_by_language = {
        language: {
            transition.parent_kind
            for transition in grammar.transitions
        }
        for language, grammar in grammars.items()
    }
    prompts = []
    prompt_masks = []
    prompt_token_rows = []
    prompt_token_masks = []
    event_rows = []
    state_rows = []
    context_kind_rows = []
    context_edge_rows = []
    candidate_kind_rows = []
    candidate_scope_rows = []
    candidate_position_rows = []
    candidate_mask_rows = []
    candidate_target_rows = []
    node_budget_rows = []
    history_rows = {
        name: []
        for name in ("action", "node", "edge", "leaf_class", "lexical", "gap")
    }
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
    unknown = 0
    for row_index, row in enumerate(rows):
        prompt = str(row["prompt"])
        target_node_budget = sum(
            record["type"] == "open_node" for record in row["actions"]
        )
        char_ids = [
            int(vocabulary["chars"].get(character, unknown))
            for character in prompt[:MAX_PROMPT]
        ]
        prompts.append(char_ids + [unknown] * (MAX_PROMPT - len(char_ids)))
        prompt_masks.append(
            [True] * len(char_ids)
            + [False] * (MAX_PROMPT - len(char_ids))
        )
        token_ids = prompt_token_ids(prompt)
        prompt_token_rows.append(
            token_ids + [0] * (MAX_PROMPT_TOKENS - len(token_ids))
        )
        prompt_token_masks.append(
            [True] * len(token_ids)
            + [False] * (MAX_PROMPT_TOKENS - len(token_ids))
        )
        language = str(row["language"])
        parent_kinds = parent_kinds_by_language[language]
        decoder = TreeActionDecoder(grammars[language])
        previous: TreeAction | None = None
        history: dict[str, list[int]] = {
            name: [] for name in history_rows
        }
        for record in row["actions"]:
            action = action_from_record(record)
            event_rows.append(row_index)
            state_rows.append(state_features(decoder))
            context_kind = "<none>"
            context_edge = "<none>"
            if isinstance(action, OpenNodeAction):
                if decoder.pending_edge is None:
                    context_kind = "<start>"
                else:
                    pending = decoder.pending_edge
                    context_kind = decoder.nodes[pending.parent_index].kind
                    context_edge = pending.edge_field
            context_kind_rows.append(
                int(vocabulary["node_ids"].get(context_kind, unknown))
            )
            context_edge_rows.append(
                int(vocabulary["edge_ids"].get(context_edge, unknown))
            )
            candidate_kinds = [0] * MAX_PARENT_CANDIDATES
            candidate_scopes = [0] * MAX_PARENT_CANDIDATES
            candidate_positions = [0] * MAX_PARENT_CANDIDATES
            candidate_masks = [False] * MAX_PARENT_CANDIDATES
            candidate_target = 0
            if isinstance(action, AttachEdgeAction):
                active_scope = (
                    decoder.state.open_scopes[-1]
                    if decoder.state.open_scopes
                    else -1
                )
                open_indices = [
                    index
                    for index, node in enumerate(decoder.nodes)
                    if node.scope_id == active_scope
                    and node.kind in parent_kinds
                ]
                target = action.parent_index
                ordered = open_indices[:MAX_PARENT_CANDIDATES]
                if target not in ordered:
                    ordered[-1] = target
                candidate_target = ordered.index(target)
                scope_kinds = decoder.state.scope_kind_by_id
                for candidate_index, node_index in enumerate(ordered):
                    node = decoder.nodes[node_index]
                    candidate_kinds[candidate_index] = int(
                        vocabulary["node_ids"].get(node.kind, unknown)
                    )
                    candidate_scopes[candidate_index] = int(
                        vocabulary["scope_ids"].get(
                            scope_kinds[node.scope_id],
                            unknown,
                        )
                    )
                    candidate_positions[candidate_index] = min(
                        node_index,
                        MAX_NODES - 1,
                    )
                    candidate_masks[candidate_index] = True
            candidate_kind_rows.append(candidate_kinds)
            candidate_scope_rows.append(candidate_scopes)
            candidate_position_rows.append(candidate_positions)
            candidate_mask_rows.append(candidate_masks)
            candidate_target_rows.append(candidate_target)
            node_budget_rows.append(target_node_budget)
            for name, value in previous_fields(previous).items():
                mapping_name = {
                    "action": "action_ids",
                    "node": "node_ids",
                    "edge": "edge_ids",
                    "leaf_class": "class_ids",
                    "lexical": "lexeme_ids",
                    "gap": "gap_ids",
                }[name]
                previous_rows[name].append(
                    int(vocabulary[mapping_name].get(value, unknown))
                )
            for name, values in history.items():
                history_rows[name].append(
                    [0] * (ACTION_HISTORY - len(values)) + values[-ACTION_HISTORY:]
                )
            if isinstance(action, AttachEdgeAction):
                action_id = REAL_ACTIONS.index("attach_edge")
            elif isinstance(action, OpenNodeAction):
                action_id = REAL_ACTIONS.index("open_node")
            elif isinstance(action, OpenScopeAction):
                action_id = REAL_ACTIONS.index("open_scope")
            elif isinstance(action, DeclareLeafAction):
                action_id = REAL_ACTIONS.index("declare_leaf")
            elif isinstance(action, CloseScopeAction):
                action_id = REAL_ACTIONS.index("close_scope")
            else:
                action_id = REAL_ACTIONS.index("terminate")
            labels["action"].append(action_id)
            labels["node_kind"].append(
                int(vocabulary["node_ids"].get(
                    action.node_kind if isinstance(action, OpenNodeAction) else "<none>",
                    unknown,
                ))
            )
            labels["edge_field"].append(
                int(vocabulary["edge_ids"].get(
                    action.edge_field if isinstance(action, AttachEdgeAction) else "<none>",
                    unknown,
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
                    unknown,
                ))
            )
            lexical = action.lexical if isinstance(action, DeclareLeafAction) else None
            span = _copy_span(prompt, lexical or "")
            labels["copy_label"].append(int(span is not None))
            labels["copy_start"].append(span[0] if span else 0)
            labels["copy_length"].append(span[1] if span else 0)
            labels["lexical"].append(
                int(vocabulary["lexeme_ids"].get(lexical or "<none>", unknown))
            )
            labels["gap"].append(
                int(vocabulary["gap_ids"].get(
                    action.gap_before if isinstance(action, DeclareLeafAction) else "",
                    unknown,
                ))
            )
            decoder.apply(action)
            previous = action
            fields = previous_fields(action)
            for name, value in fields.items():
                mapping_name = {
                    "action": "action_ids",
                    "node": "node_ids",
                    "edge": "edge_ids",
                    "leaf_class": "class_ids",
                    "lexical": "lexeme_ids",
                    "gap": "gap_ids",
                }[name]
                history[name].append(
                    int(vocabulary[mapping_name].get(value, unknown))
                )
    return RealPolicyBatch(
        prompts=torch.tensor(prompts, dtype=torch.long),
        prompt_masks=torch.tensor(prompt_masks, dtype=torch.bool),
        prompt_token_ids=torch.tensor(prompt_token_rows, dtype=torch.long),
        prompt_token_mask=torch.tensor(prompt_token_masks, dtype=torch.bool),
        event_prompt_rows=torch.tensor(event_rows, dtype=torch.long),
        state_features=torch.tensor(state_rows, dtype=torch.float32),
        context_kind=torch.tensor(context_kind_rows, dtype=torch.long),
        context_edge=torch.tensor(context_edge_rows, dtype=torch.long),
        candidate_kind=torch.tensor(candidate_kind_rows, dtype=torch.long),
        candidate_scope=torch.tensor(candidate_scope_rows, dtype=torch.long),
        candidate_position=torch.tensor(candidate_position_rows, dtype=torch.long),
        candidate_mask=torch.tensor(candidate_mask_rows, dtype=torch.bool),
        candidate_target=torch.tensor(candidate_target_rows, dtype=torch.long),
        node_budget=torch.tensor(node_budget_rows, dtype=torch.long),
        history={
            name: torch.tensor(values, dtype=torch.long)
            for name, values in history_rows.items()
        },
        previous={
            name: torch.tensor(values, dtype=torch.long)
            for name, values in previous_rows.items()
        },
        **{
            name: torch.tensor(values, dtype=torch.long)
            for name, values in labels.items()
        },
    )


def regenerate_state_features(
    records: Sequence[Mapping[str, object]],
    grammar: ASTTransitionGrammar,
) -> list[list[float]]:
    decoder = TreeActionDecoder(grammar)
    output = []
    for record in records:
        output.append(state_features(decoder))
        decoder.apply(action_from_record(record))
    return output


def event_distribution(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(record["type"]) for record in row["actions"])
    return dict(counts)


__all__ = [
    "ACTION_HISTORY",
    "MAX_LEXEME",
    "MAX_NODES",
    "MAX_PARENT_CANDIDATES",
    "MAX_PROMPT",
    "MAX_PROMPT_TOKENS",
    "PROMPT_TOKEN_BUCKETS",
    "REAL_ACTIONS",
    "STATE_FEATURES",
    "RealPolicyBatch",
    "action_from_record",
    "encode_action_rows",
    "event_distribution",
    "load_action_rows",
    "previous_fields",
    "prompt_token_ids",
    "regenerate_state_features",
    "state_features",
]
