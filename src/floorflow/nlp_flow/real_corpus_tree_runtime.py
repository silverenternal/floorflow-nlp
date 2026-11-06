"""Target-free runtime for the real-corpus tree policy."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import partial

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_generation_trace import (
    finish_trace,
    select_and_apply,
)
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.real_corpus_policy_data import (
    ACTION_HISTORY,
    MAX_NODES,
    MAX_PARENT_CANDIDATES,
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    previous_fields,
    prompt_token_ids,
    state_features,
)
from floorflow.nlp_flow.real_corpus_tree_policy import RealCorpusTreePolicy
from floorflow.nlp_flow.tree_action_decoder import (
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
    render_tree_structure,
)
from floorflow.nlp_flow.tree_action_selection import _select_real_action
from floorflow.nlp_flow.tree_completion_runtime import TreeCompletionBudget


def terminalizable_kinds(grammar: ASTTransitionGrammar) -> set[str]:
    terminal = {
        kind
        for kind, classes in grammar.leaf_classes.items()
        if classes
    }
    changed = True
    while changed:
        changed = False
        for transition in grammar.transitions:
            if (
                transition.parent_kind not in terminal
                and transition.child_kind in terminal
            ):
                terminal.add(transition.parent_kind)
                changed = True
    return terminal


def viable_parent_kinds(grammar: ASTTransitionGrammar) -> set[str]:
    terminal = terminalizable_kinds(grammar)
    return {
        transition.parent_kind
        for transition in grammar.transitions
        if transition.child_kind in terminal
    }


def _previous_tensor(
    action: TreeAction | None,
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for name, value in previous_fields(action).items():
        mapping_name = {
            "action": "action_ids",
            "node": "node_ids",
            "edge": "edge_ids",
            "leaf_class": "class_ids",
            "lexical": "lexeme_ids",
            "gap": "gap_ids",
        }[name]
        output[name] = torch.tensor(
            [int(vocabulary[mapping_name].get(value, 0))],
            device=device,
        )
    return output


def _history_tensor(
    actions: Sequence[TreeAction],
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output = {}
    for name in ("action", "node", "edge", "leaf_class", "lexical", "gap"):
        mapping_name = {
            "action": "action_ids",
            "node": "node_ids",
            "edge": "edge_ids",
            "leaf_class": "class_ids",
            "lexical": "lexeme_ids",
            "gap": "gap_ids",
        }[name]
        values = [
            int(
                vocabulary[mapping_name].get(
                    previous_fields(action)[name],
                    0,
                )
            )
            for action in actions[-ACTION_HISTORY:]
        ]
        output[name] = torch.tensor(
            [[0] * (ACTION_HISTORY - len(values)) + values],
            device=device,
        )
    return output


def _inference_context(
    decoder: TreeActionDecoder,
    parent_kinds: set[str],
    vocabulary: Mapping[str, object],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
]:
    context_kind = "<none>"
    context_edge = "<none>"
    if decoder.pending_edge is not None:
        pending = decoder.pending_edge
        context_kind = decoder.nodes[pending.parent_index].kind
        context_edge = pending.edge_field
    elif not decoder.nodes:
        context_kind = "<start>"
    active_scope = (
        decoder.state.open_scopes[-1]
        if decoder.state.open_scopes
        else None
    )
    candidate_indices = [
        index
        for index, node in enumerate(decoder.nodes)
        if node.scope_id == active_scope
        and node.kind in parent_kinds
    ][-MAX_PARENT_CANDIDATES:]
    candidate_kind = [0] * MAX_PARENT_CANDIDATES
    candidate_scope = [0] * MAX_PARENT_CANDIDATES
    candidate_position = [0] * MAX_PARENT_CANDIDATES
    candidate_mask = [False] * MAX_PARENT_CANDIDATES
    scope_kinds = decoder.state.scope_kind_by_id
    for position, index in enumerate(candidate_indices):
        node = decoder.nodes[index]
        candidate_kind[position] = int(
            vocabulary["node_ids"].get(node.kind, 0)
        )
        candidate_scope[position] = int(
            vocabulary["scope_ids"].get(scope_kinds[node.scope_id], 0)
        )
        candidate_position[position] = min(index, MAX_NODES - 1)
        candidate_mask[position] = True
    return (
        torch.tensor(
            [int(vocabulary["node_ids"].get(context_kind, 0))],
            device=device,
        ),
        torch.tensor(
            [int(vocabulary["edge_ids"].get(context_edge, 0))],
            device=device,
        ),
        torch.tensor([candidate_kind], device=device),
        torch.tensor([candidate_scope], device=device),
        torch.tensor([candidate_position], device=device),
        torch.tensor([candidate_mask], dtype=torch.bool, device=device),
        candidate_indices,
    )


@torch.no_grad()
def generate_real_task(
    model: RealCorpusTreePolicy,
    prompt: str,
    language: str,
    grammar: ASTTransitionGrammar,
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    max_steps: int = 4096,
    trace: list[dict[str, object]] | None = None,
    completion_budget: TreeCompletionBudget | None = None,
) -> tuple[str | None, list[TreeAction], str | None]:
    token_ids = prompt_token_ids(prompt)
    char_ids = [
        int(vocabulary["chars"].get(character, 0))
        for character in prompt[:MAX_PROMPT]
    ]
    prompt_ids = torch.tensor(
        [token_ids + [0] * (MAX_PROMPT_TOKENS - len(token_ids))],
        device=device,
    )
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    prompt_chars = torch.tensor(
        [char_ids + [0] * (MAX_PROMPT - len(char_ids))],
        device=device,
    )
    prompt_char_mask = torch.tensor(
        [[True] * len(char_ids) + [False] * (MAX_PROMPT - len(char_ids))],
        dtype=torch.bool,
        device=device,
    )
    decoder = TreeActionDecoder(grammar)
    previous: TreeAction | None = None
    actions = []
    node_budget: int | None = None
    parent_kinds = viable_parent_kinds(grammar)
    terminal_kinds = terminalizable_kinds(grammar)
    try:
        for _step in range(max_steps):
            (
                context_kind,
                context_edge,
                candidate_kind,
                candidate_scope,
                candidate_position,
                candidate_mask,
                parent_candidates,
            ) = _inference_context(
                decoder,
                parent_kinds,
                vocabulary,
                device,
            )
            logits = model(
                prompt_ids,
                prompt_mask,
                prompt_chars,
                prompt_char_mask,
                torch.tensor(
                    [state_features(decoder)],
                    dtype=torch.float32,
                    device=device,
                ),
                context_kind,
                context_edge,
                candidate_kind,
                candidate_scope,
                candidate_position,
                candidate_mask,
                _history_tensor(actions, vocabulary, device),
                _previous_tensor(previous, vocabulary, device),
            )
            if node_budget is None:
                node_budget = max(
                    1,
                    int(logits["node_budget"][0].argmax()),
                )
            action = select_and_apply(decoder, partial(_select_real_action,
                decoder,
                grammar,
                logits,
                vocabulary,
                prompt,
                parent_candidates,
                node_budget,
                parent_kinds,
                terminal_kinds,
                completion_budget,
            ), trace, budget=node_budget, parent_candidates=parent_candidates)
            actions.append(action)
            previous = action
            if isinstance(action, TerminateAction):
                source = render_tree_structure(decoder.structure())
                parse_ast_graph(source, language)
                finish_trace(trace, decoder, None)
                return source, actions, None
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        finish_trace(trace, decoder, str(error))
        return None, actions, str(error)
    finish_trace(trace, decoder, "action generation exceeded the step cap")
    return None, actions, "action generation exceeded the step cap"


def save_policy_metadata(
    path,
    *,
    vocabulary: Mapping[str, object],
    metrics: Mapping[str, object],
) -> None:
    path.write_text(
        json.dumps({"vocabulary": vocabulary, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["generate_real_task", "save_policy_metadata"]
