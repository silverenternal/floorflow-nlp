"""Autoregressive runtime for the causal-prefix transformer policy."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_generation_trace import finish_trace, select_and_apply
from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.experiment_deadline import ExperimentDeadline
from floorflow.nlp_flow.prefix_policy_data import PrefixPolicyBatch
from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy
from floorflow.nlp_flow.real_corpus_policy_data import (
    MAX_PROMPT,
    MAX_PROMPT_TOKENS,
    prompt_token_ids,
    state_features,
)
from floorflow.nlp_flow.real_corpus_tree_runtime import (
    _history_tensor,
    _inference_context,
    _previous_tensor,
    terminalizable_kinds,
    viable_parent_kinds,
)
from floorflow.nlp_flow.tree_action_decoder import (
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
    render_tree_structure,
)
from floorflow.nlp_flow.tree_action_selection import _select_real_action
from floorflow.nlp_flow.tree_completion_runtime import TreeCompletionBudget


def _prefix_batch(
    *,
    prompt_token_ids_value: torch.Tensor,
    prompt_token_mask: torch.Tensor,
    prompt_chars: torch.Tensor,
    prompt_char_mask: torch.Tensor,
    node_budget: int,
    minimum_budget: int,
    state_rows: list[list[float]],
    context_kind: list[int],
    context_edge: list[int],
    candidate_kind: list[list[int]],
    candidate_scope: list[list[int]],
    candidate_position: list[list[int]],
    candidate_mask: list[list[bool]],
    histories: dict[str, list[list[int]]],
    previous: dict[str, list[int]],
    device: torch.device,
) -> PrefixPolicyBatch:
    steps = len(state_rows)
    zero = torch.zeros((1, steps), dtype=torch.long, device=device)
    return PrefixPolicyBatch(
        prompts=prompt_chars,
        prompt_masks=prompt_char_mask,
        prompt_token_ids=prompt_token_ids_value,
        prompt_token_mask=prompt_token_mask,
        node_budget=torch.tensor([node_budget], dtype=torch.long, device=device),
        node_budget_minimum=torch.tensor([minimum_budget], dtype=torch.long, device=device),
        step_mask=torch.ones((1, steps), dtype=torch.bool, device=device),
        state_features=torch.tensor([state_rows], dtype=torch.float32, device=device),
        context_kind=torch.tensor([context_kind], dtype=torch.long, device=device),
        context_edge=torch.tensor([context_edge], dtype=torch.long, device=device),
        candidate_kind=torch.tensor([candidate_kind], dtype=torch.long, device=device),
        candidate_scope=torch.tensor([candidate_scope], dtype=torch.long, device=device),
        candidate_position=torch.tensor(
            [candidate_position],
            dtype=torch.long,
            device=device,
        ),
        candidate_mask=torch.tensor(
            [candidate_mask],
            dtype=torch.bool,
            device=device,
        ),
        candidate_target=zero,
        history={
            name: torch.tensor([values], dtype=torch.long, device=device)
            for name, values in histories.items()
        },
        previous={
            name: torch.tensor([values], dtype=torch.long, device=device)
            for name, values in previous.items()
        },
        action=zero,
        node_kind=zero,
        edge_field=zero,
        parent=zero,
        leaf_node=zero,
        leaf_class=zero,
        copy_label=zero,
        copy_start=zero,
        copy_length=zero,
        lexical=zero,
        gap=zero,
    )


@torch.no_grad()
def generate_prefix_tree(
    model: PrefixTreePolicy,
    prompt: str,
    language: str,
    grammar: ASTTransitionGrammar,
    vocabulary: Mapping[str, object],
    *,
    device: torch.device,
    max_steps: int = 768,
    trace: list[dict[str, object]] | None = None,
    completion_budget: TreeCompletionBudget | None = None,
    use_completion_candidate_mask: bool = True,
    deadline: ExperimentDeadline | None = None,
    local_transition_mask: bool = True,
    lexical_candidate_mask: bool = True,
) -> tuple[str | None, list[TreeAction], str | None]:
    token_values = prompt_token_ids(prompt)
    token_ids = torch.tensor(
        [token_values + [0] * (MAX_PROMPT_TOKENS - len(token_values))],
        device=device,
    )
    token_mask = torch.tensor(
        [[True] * len(token_values) + [False] * (MAX_PROMPT_TOKENS - len(token_values))],
        dtype=torch.bool,
        device=device,
    )
    char_values = [
        int(vocabulary["chars"].get(character, 0))
        for character in prompt[:MAX_PROMPT]
    ]
    chars = torch.tensor(
        [char_values + [0] * (MAX_PROMPT - len(char_values))],
        device=device,
    )
    char_mask = torch.tensor(
        [[True] * len(char_values) + [False] * (MAX_PROMPT - len(char_values))],
        dtype=torch.bool,
        device=device,
    )
    decoder = TreeActionDecoder(grammar)
    parent_kinds = (
        viable_parent_kinds(grammar)
        if local_transition_mask
        else set(vocabulary["node_names"])
    )
    terminal_kinds = terminalizable_kinds(grammar)
    if model.budget_residual:
        if completion_budget is None:
            raise ValueError("residual budget inference requires a completion contract")
        minimum_budget = completion_budget.minimum_root_nodes()
    else:
        minimum_budget = 1
    actions: list[TreeAction] = []
    state_rows: list[list[float]] = []
    context_kind: list[int] = []
    context_edge: list[int] = []
    candidate_kind: list[list[int]] = []
    candidate_scope: list[list[int]] = []
    candidate_position: list[list[int]] = []
    candidate_mask: list[list[bool]] = []
    histories: dict[str, list[list[int]]] = {
        name: [] for name in ("action", "node", "edge", "leaf_class", "lexical", "gap")
    }
    previous_rows: dict[str, list[int]] = {
        name: [] for name in histories
    }
    node_budget = 1
    try:
        for _step in range(max_steps):
            if deadline is not None:
                deadline.check("free_running_action_generation")
            (
                context_kind_tensor,
                context_edge_tensor,
                candidate_kind_tensor,
                candidate_scope_tensor,
                candidate_position_tensor,
                candidate_mask_tensor,
                parent_candidates,
            ) = _inference_context(decoder, parent_kinds, vocabulary, device)
            previous = _previous_tensor(
                None if not actions else actions[-1],
                vocabulary,
                device,
            )
            history = _history_tensor(actions, vocabulary, device)
            state_rows.append(state_features(decoder))
            context_kind.append(int(context_kind_tensor[0]))
            context_edge.append(int(context_edge_tensor[0]))
            candidate_kind.append(candidate_kind_tensor[0].tolist())
            candidate_scope.append(candidate_scope_tensor[0].tolist())
            candidate_position.append(candidate_position_tensor[0].tolist())
            candidate_mask.append(candidate_mask_tensor[0].tolist())
            for name, values in history.items():
                histories[name].append(values[0].tolist())
            for name, values in previous.items():
                previous_rows[name].append(int(values[0]))
            batch = _prefix_batch(
                prompt_token_ids_value=token_ids,
                prompt_token_mask=token_mask,
                prompt_chars=chars,
                prompt_char_mask=char_mask,
                node_budget=node_budget,
                minimum_budget=minimum_budget,
                state_rows=state_rows,
                context_kind=context_kind,
                context_edge=context_edge,
                candidate_kind=candidate_kind,
                candidate_scope=candidate_scope,
                candidate_position=candidate_position,
                candidate_mask=candidate_mask,
                histories=histories,
                previous=previous_rows,
                device=device,
            )
            output = model(batch)
            if len(actions) == 0:
                node_budget = max(1, int(output["node_budget"][0].argmax()))
            logits = {}
            for name, value in output.items():
                if name in {"node_budget", "pooled_prompt"}:
                    continue
                last = value[0, -1]
                logits[name] = last if name == "copy" else last[None]
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
                completion_budget if use_completion_candidate_mask else None,
                local_transition_mask,
                lexical_candidate_mask,
            ), trace, budget=node_budget, parent_candidates=parent_candidates)
            actions.append(action)
            if isinstance(action, TerminateAction):
                source = render_tree_structure(decoder.structure())
                parse_ast_graph(source, language)
                finish_trace(trace, decoder, None)
                return source, actions, None
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        finish_trace(trace, decoder, str(error))
        return None, actions, str(error)
    finish_trace(trace, decoder, "prefix action generation exceeded the step cap")
    return None, actions, "prefix action generation exceeded the step cap"


__all__ = ["generate_prefix_tree"]
