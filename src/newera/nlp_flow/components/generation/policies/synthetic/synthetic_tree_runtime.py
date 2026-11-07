"""Target-free runtime and training for the bounded synthetic tree policy."""

from __future__ import annotations

import torch
from torch import nn

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar
from newera.nlp_flow.engineering.components.free_schedule_spec.schedule_free import ScheduleFreeAdamW
from newera.nlp_flow.components.generation.policies.synthetic.synthetic_tree_policy import (
    ACTIONS,
    MAX_LEXEME,
    MAX_NODES,
    MAX_PROMPT,
    SyntheticTreeTask,
    _action_previous_fields,
    _python_grammar,
    _state_features,
    encode_tasks,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TerminateAction,
    TreeAction,
    TreeActionDecoder,
    render_tree_structure,
)


class TreeActionPolicy(nn.Module):
    def __init__(
        self,
        vocabulary: dict[str, object],
        *,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(len(vocabulary["chars"]) + 1, 24)
        self.prompt_encoder = nn.GRU(24, hidden_size, batch_first=True)
        self.previous_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(vocabulary[key]) + 1, hidden_size)
                for key in (
                    "action_ids",
                    "node_ids",
                    "edge_ids",
                    "class_ids",
                    "lexeme_ids",
                    "gap_ids",
                )
            ]
        )
        self.state_projection = nn.Linear(9, hidden_size)
        self.step = nn.GRUCell(hidden_size * 2, hidden_size)
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(hidden_size, len(ACTIONS)),
                "node_kind": nn.Linear(hidden_size, len(vocabulary["node_ids"]) + 1),
                "edge_field": nn.Linear(hidden_size, len(vocabulary["edge_ids"]) + 1),
                "parent": nn.Linear(hidden_size, MAX_NODES),
                "leaf_node": nn.Linear(hidden_size, MAX_NODES),
                "leaf_class": nn.Linear(hidden_size, len(vocabulary["class_ids"]) + 1),
                "copy": nn.Linear(hidden_size, 1),
                "copy_start": nn.Linear(hidden_size, MAX_PROMPT),
                "copy_length": nn.Linear(hidden_size, MAX_LEXEME),
                "lexical": nn.Linear(hidden_size, len(vocabulary["lexeme_ids"]) + 1),
                "gap": nn.Linear(hidden_size, len(vocabulary["gap_ids"]) + 1),
            }
        )

    def encode_prompt(
        self,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        _output, hidden = self.prompt_encoder(self.char_embedding(prompt))
        _lengths = prompt_mask.to(prompt.dtype).sum(dim=1).clamp_min(1)
        return hidden[-1]

    def step_logits(
        self,
        hidden: torch.Tensor,
        state_features: torch.Tensor,
        previous: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        previous_value = sum(
            embedding(previous[key])
            for embedding, key in zip(
                self.previous_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        value = torch.cat(
            (previous_value, self.state_projection(state_features)),
            dim=-1,
        )
        next_hidden = self.step(value, hidden)
        return next_hidden, {
            name: head(next_hidden)
            for name, head in self.heads.items()
        }


def train_tree_action_policy(
    tasks: list[SyntheticTreeTask],
    vocabulary: dict[str, object],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[TreeActionPolicy, dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batch = encode_tasks(tasks, vocabulary)
    model = TreeActionPolicy(vocabulary).to(device)
    optimizer = ScheduleFreeAdamW(model.parameters())
    tensors = {
        name: value.to(device)
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
    }
    previous = {
        name: value.to(device) for name, value in batch.previous.items()
    }
    prompt = tensors["prompt"]
    prompt_mask = tensors["prompt_mask"]
    history = []
    for _epoch in range(epochs):
        model.train()
        optimizer.train()
        optimizer.zero_grad(set_to_none=True)
        hidden = model.encode_prompt(prompt, prompt_mask)
        total = prompt.new_zeros((), dtype=torch.float32)
        steps = batch.state_features.shape[1]
        for step in range(steps):
            state = tensors["state_features"][:, step]
            prev = {name: values[:, step] for name, values in previous.items()}
            hidden, logits = model.step_logits(hidden, state, prev)
            mask = tensors["step_mask"][:, step]
            if not mask.any():
                continue
            action = tensors["action"][:, step]
            total = total + torch.nn.functional.cross_entropy(
                logits["action"][mask],
                action[mask],
            )
            open_mask = mask & action.eq(ACTIONS.index("open_node"))
            if open_mask.any():
                total = total + torch.nn.functional.cross_entropy(
                    logits["node_kind"][open_mask],
                    tensors["node_kind"][:, step][open_mask],
                )
            attach_mask = mask & action.eq(ACTIONS.index("attach_edge"))
            if attach_mask.any():
                total = (
                    total
                    + torch.nn.functional.cross_entropy(
                        logits["edge_field"][attach_mask],
                        tensors["edge_field"][:, step][attach_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["parent"][attach_mask],
                        tensors["parent"][:, step][attach_mask],
                    )
                )
            leaf_mask = mask & action.eq(ACTIONS.index("declare_leaf"))
            if leaf_mask.any():
                total = (
                    total
                    + torch.nn.functional.cross_entropy(
                        logits["leaf_node"][leaf_mask],
                        tensors["leaf_node"][:, step][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["leaf_class"][leaf_mask],
                        tensors["leaf_class"][:, step][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["lexical"][leaf_mask],
                        tensors["lexical"][:, step][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["gap"][leaf_mask],
                        tensors["gap"][:, step][leaf_mask],
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        logits["copy"][leaf_mask, 0],
                        tensors["copy_label"][:, step][leaf_mask],
                    )
                )
                copy_mask = leaf_mask & tensors["copy_label"][:, step].bool()
                if copy_mask.any():
                    total = (
                        total
                        + torch.nn.functional.cross_entropy(
                            logits["copy_start"][copy_mask],
                            tensors["copy_start"][:, step][copy_mask],
                        )
                        + torch.nn.functional.cross_entropy(
                            logits["copy_length"][copy_mask],
                            tensors["copy_length"][:, step][copy_mask],
                        )
                    )
        loss = total / steps
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    optimizer.eval()
    return model, {"loss_history": history, "steps": batch.state_features.shape[1]}


def generate_task(
    model: TreeActionPolicy,
    task: SyntheticTreeTask,
    vocabulary: dict[str, object],
    *,
    device: torch.device,
    max_steps: int = 128,
) -> tuple[str | None, list[TreeAction], str | None]:
    grammar = _python_grammar()
    decoder = TreeActionDecoder(grammar)
    prompt_ids = torch.tensor(
        [[
            int(vocabulary["chars"].get(character, 0))
            for character in task.prompt[:MAX_PROMPT]
        ]],
        device=device,
    )
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    hidden = model.encode_prompt(prompt_ids, prompt_mask)
    actions = []
    previous: TreeAction | None = None
    try:
        for _step in range(max_steps):
            previous_ids = {}
            for name, value in _action_previous_fields(previous).items():
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
                previous_ids[name] = torch.tensor(
                    [int(mapping.get(value, 0))],
                    device=device,
                )
            state = torch.tensor(
                [_state_features(decoder, grammar)],
                dtype=torch.float32,
                device=device,
            )
            hidden, logits = model.step_logits(hidden, state, previous_ids)
            action = _select_action(
                decoder,
                grammar,
                logits,
                vocabulary,
                task.prompt,
            )
            decoder.apply(action)
            actions.append(action)
            previous = action
            if isinstance(action, TerminateAction):
                return render_tree_structure(decoder.structure()), actions, None
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        return None, actions, str(error)
    return None, actions, "generation exceeded the step cap"


def _masked_choice(logits: torch.Tensor, allowed: set[int]) -> int:
    if not allowed:
        raise ValueError("no legal choices")
    mask = torch.full_like(logits, -torch.inf)
    for index in allowed:
        mask[index] = logits[index]
    return int(mask.argmax())


def _select_action(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
    logits: dict[str, torch.Tensor],
    vocabulary: dict[str, object],
    prompt: str,
) -> TreeAction:
    if decoder.expected_scope is not None:
        scope_id, node_index, scope_kind = decoder.expected_scope
        return OpenScopeAction(scope_id, node_index, scope_kind)
    if not decoder.nodes:
        candidates = {
            vocabulary["node_ids"][value]
            for value in grammar.legal_root_kinds
            if value in vocabulary["node_ids"]
        }
        node_id = _masked_choice(logits["node_kind"][0], candidates)
        return OpenNodeAction(vocabulary["node_names"][node_id])
    if decoder.pending_edge is not None:
        pending = decoder.pending_edge
        candidates = {
            vocabulary["node_ids"][value]
            for value in decoder.state.allowed_child_kinds(
                parent_index=pending.parent_index,
                edge_field=pending.edge_field,
            )
            if value in vocabulary["node_ids"]
        }
        node_id = _masked_choice(logits["node_kind"][0], candidates)
        return OpenNodeAction(vocabulary["node_names"][node_id])
    candidates = _allowed_actions(decoder, grammar)
    action_name = ACTIONS[_masked_choice(logits["action"][0], candidates)]
    if action_name == "attach_edge":
        parent_allowed = {
            index
            for index, node in enumerate(decoder.nodes)
            if node.scope_id in decoder.state.open_scopes
            and any(
                transition.parent_kind == node.kind
                for transition in grammar.transitions
            )
        }
        parent = _masked_choice(logits["parent"][0], parent_allowed)
        edge_allowed = {
            vocabulary["edge_ids"][transition.edge_field]
            for transition in grammar.transitions
            if transition.parent_kind == decoder.nodes[parent].kind
            and transition.edge_field in vocabulary["edge_ids"]
        }
        edge_id = _masked_choice(logits["edge_field"][0], edge_allowed)
        return AttachEdgeAction(parent, vocabulary["edge_names"][edge_id])
    if action_name == "declare_leaf":
        leaf_allowed = {
            index
            for index, node in enumerate(decoder.nodes)
            if not node.children
            and node.leaf_class is None
            and bool(grammar.leaf_classes.get(node.kind))
        }
        node_index = _masked_choice(logits["leaf_node"][0], leaf_allowed)
        node = decoder.nodes[node_index]
        class_allowed = {
            vocabulary["class_ids"][value]
            for value in grammar.leaf_classes[node.kind]
            if value in vocabulary["class_ids"]
        }
        class_id = _masked_choice(logits["leaf_class"][0], class_allowed)
        if bool(torch.sigmoid(logits["copy"][0]).ge(0.5)):
            start = int(logits["copy_start"][0].argmax())
            length = int(logits["copy_length"][0].argmax())
            lexical = prompt[start : start + length]
        else:
            lexical = vocabulary["lexeme_names"][int(logits["lexical"][0].argmax())]
        gap = vocabulary["gap_names"][int(logits["gap"][0].argmax())]
        return DeclareLeafAction(
            node_index,
            vocabulary["class_names"][class_id],
            lexical,
            gap,
        )
    if action_name == "close_scope":
        return CloseScopeAction(decoder.state.open_scopes[-1])
    return TerminateAction()


def _allowed_actions(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
) -> set[int]:
    allowed = set()
    if any(
        node.scope_id in decoder.state.open_scopes
        and any(
            transition.parent_kind == node.kind
            for transition in grammar.transitions
        )
        for node in decoder.nodes
    ):
        allowed.add(ACTIONS.index("attach_edge"))
    if any(
        not node.children
        and node.leaf_class is None
        and bool(grammar.leaf_classes.get(node.kind))
        for node in decoder.nodes
    ):
        allowed.add(ACTIONS.index("declare_leaf"))
    if decoder.state.open_scopes:
        allowed.add(ACTIONS.index("close_scope"))
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    if decoder.state.eos_ready and not undeclared:
        allowed.add(ACTIONS.index("terminate"))
    return allowed


__all__ = ["TreeActionPolicy", "generate_task", "train_tree_action_policy"]
