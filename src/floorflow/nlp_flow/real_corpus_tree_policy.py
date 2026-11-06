"""Real-corpus event-conditioned Seq2Tree action policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW
from floorflow.nlp_flow.real_corpus_policy_data import (
    MAX_LEXEME,
    MAX_NODES,
    MAX_PROMPT,
    PROMPT_TOKEN_BUCKETS,
    REAL_ACTIONS,
    STATE_FEATURES,
    RealPolicyBatch,
    encode_action_rows,
)
from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary


class RealCorpusTreePolicy(nn.Module):
    def __init__(
        self,
        vocabulary: Mapping[str, object],
        *,
        hidden_size: int = 96,
    ) -> None:
        super().__init__()
        self.vocabulary = dict(vocabulary)
        self.prompt_encoder = nn.EmbeddingBag(
            PROMPT_TOKEN_BUCKETS + 1,
            hidden_size,
            mode="mean",
            padding_idx=0,
        )
        self.prompt_char_embedding = nn.Embedding(
            len(vocabulary["chars"]) + 1,
            24,
        )
        self.prompt_position_embedding = nn.Embedding(MAX_PROMPT, hidden_size)
        self.prompt_char_projection = nn.Linear(24, hidden_size)
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
        self.history_embeddings = nn.ModuleList(
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
        self.state_projection = nn.Linear(STATE_FEATURES, hidden_size)
        self.context_kind_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.context_edge_embedding = nn.Embedding(
            len(vocabulary["edge_ids"]) + 1,
            hidden_size,
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(hidden_size, len(REAL_ACTIONS)),
                "node_kind": nn.Linear(hidden_size, len(vocabulary["node_ids"]) + 1),
                "edge_field": nn.Linear(hidden_size, len(vocabulary["edge_ids"]) + 1),
                "leaf_node": nn.Linear(hidden_size, MAX_NODES),
                "leaf_class": nn.Linear(hidden_size, len(vocabulary["class_ids"]) + 1),
                "copy": nn.Linear(hidden_size, 1),
                "copy_length": nn.Linear(hidden_size, MAX_LEXEME + 1),
                "lexical": nn.Linear(hidden_size, len(vocabulary["lexeme_ids"]) + 1),
                "gap": nn.Linear(hidden_size, len(vocabulary["gap_ids"]) + 1),
                "node_budget": nn.Linear(hidden_size, MAX_NODES + 1),
            }
        )
        self.parent_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.candidate_node_embedding = nn.Embedding(
            len(vocabulary["node_ids"]) + 1,
            hidden_size,
        )
        self.candidate_scope_embedding = nn.Embedding(
            len(vocabulary["scope_ids"]) + 1,
            hidden_size,
        )
        self.candidate_position_embedding = nn.Embedding(MAX_NODES, hidden_size)
        self.candidate_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.copy_query = nn.Linear(hidden_size, hidden_size, bias=False)

    def encode_prompt(
        self,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_mask.shape != prompt.shape:
            raise ValueError("prompt mask must align with prompt ids")
        return self.prompt_encoder(prompt)

    def forward(
        self,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_chars: torch.Tensor,
        prompt_char_mask: torch.Tensor,
        state: torch.Tensor,
        context_kind: torch.Tensor,
        context_edge: torch.Tensor,
        candidate_kind: torch.Tensor,
        candidate_scope: torch.Tensor,
        candidate_position: torch.Tensor,
        candidate_mask: torch.Tensor,
        history: Mapping[str, torch.Tensor],
        previous: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prompt_state = self.encode_prompt(prompt, prompt_mask)
        previous_state = sum(
            embedding(previous[key])
            for embedding, key in zip(
                self.previous_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        history_state = sum(
            embedding(history[key]).mean(dim=1)
            for embedding, key in zip(
                self.history_embeddings,
                ("action", "node", "edge", "leaf_class", "lexical", "gap"),
                strict=True,
            )
        )
        hidden = self.trunk(
            torch.cat(
                (
                    prompt_state,
                    previous_state
                    + history_state
                    + self.state_projection(state),
                ),
                dim=-1,
            )
        )
        output = {
            name: head(hidden)
            for name, head in self.heads.items()
        }
        node_context = (
            self.context_kind_embedding(context_kind)
            + self.context_edge_embedding(context_edge)
        )
        output["node_kind"] = self.heads["node_kind"](hidden + node_context)
        candidate_state = self.candidate_projection(
            self.candidate_node_embedding(candidate_kind)
            + self.candidate_scope_embedding(candidate_scope)
            + self.candidate_position_embedding(candidate_position)
        )
        parent_logits = torch.einsum(
            "bd,bcd->bc",
            self.parent_query(hidden),
            candidate_state,
        )
        output["parent"] = parent_logits.masked_fill(
            ~candidate_mask,
            -torch.inf,
        )
        positions = torch.arange(prompt_chars.shape[1], device=prompt_chars.device)
        char_states = (
            self.prompt_char_projection(self.prompt_char_embedding(prompt_chars))
            + self.prompt_position_embedding(positions)[None]
        )
        copy_scores = torch.einsum(
            "bd,bld->bl",
            self.copy_query(hidden),
            char_states,
        )
        output["copy_start"] = copy_scores.masked_fill(
            ~prompt_char_mask,
            -torch.inf,
        )
        return output


def _batch_tensors(batch: RealPolicyBatch, device: torch.device) -> dict[str, torch.Tensor]:
    output = {
        name: value.to(device)
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
    }
    output["previous"] = {
        name: value.to(device) for name, value in batch.previous.items()
    }
    output["history"] = {
        name: value.to(device) for name, value in batch.history.items()
    }
    return output


def train_real_corpus_tree_policy(
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    grammars: Mapping[str, ASTTransitionGrammar],
    *,
    device: torch.device,
    seed: int,
    epochs: int = 2,
    batch_size: int = 1024,
) -> tuple[RealCorpusTreePolicy, dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    vocabulary = build_vocabulary(train_rows, grammars)
    train_batch = encode_action_rows(train_rows, grammars, vocabulary)
    validation_batch = encode_action_rows(validation_rows, grammars, vocabulary)
    model = RealCorpusTreePolicy(vocabulary).to(device)
    optimizer = ScheduleFreeAdamW(model.parameters())
    train_data = _batch_tensors(train_batch, device)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    for _epoch in range(epochs):
        model.train()
        optimizer.train()
        order = torch.randperm(len(train_data["action"]), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size].to(device)
            prompt_row = train_data["event_prompt_rows"][index]
            previous = {
                name: train_data["previous"][name][index]
                for name in train_data["previous"]
            }
            action_history = {
                name: train_data["history"][name][index]
                for name in train_data["history"]
            }
            logits = model(
                train_data["prompt_token_ids"][prompt_row],
                train_data["prompt_token_mask"][prompt_row],
                train_data["prompts"][prompt_row],
                train_data["prompt_masks"][prompt_row],
                train_data["state_features"][index],
                train_data["context_kind"][index],
                train_data["context_edge"][index],
                train_data["candidate_kind"][index],
                train_data["candidate_scope"][index],
                train_data["candidate_position"][index],
                train_data["candidate_mask"][index],
                action_history,
                previous,
            )
            action = train_data["action"][index]
            loss = torch.nn.functional.cross_entropy(
                logits["action"],
                action,
            )
            loss = loss + torch.nn.functional.cross_entropy(
                logits["node_budget"],
                train_data["node_budget"][index],
            )
            open_mask = action.eq(REAL_ACTIONS.index("open_node"))
            if open_mask.any():
                loss = loss + torch.nn.functional.cross_entropy(
                    logits["node_kind"][open_mask],
                    train_data["node_kind"][index][open_mask],
                )
            attach_mask = action.eq(REAL_ACTIONS.index("attach_edge"))
            if attach_mask.any():
                loss = (
                    loss
                    + torch.nn.functional.cross_entropy(
                        logits["edge_field"][attach_mask],
                        train_data["edge_field"][index][attach_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["parent"][attach_mask],
                        train_data["candidate_target"][index][attach_mask],
                    )
                )
            leaf_mask = action.eq(REAL_ACTIONS.index("declare_leaf"))
            if leaf_mask.any():
                loss = (
                    loss
                    + torch.nn.functional.cross_entropy(
                        logits["leaf_node"][leaf_mask],
                        train_data["leaf_node"][index][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["leaf_class"][leaf_mask],
                        train_data["leaf_class"][index][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["lexical"][leaf_mask],
                        train_data["lexical"][index][leaf_mask],
                    )
                    + torch.nn.functional.cross_entropy(
                        logits["gap"][leaf_mask],
                        train_data["gap"][index][leaf_mask],
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        logits["copy"][leaf_mask, 0],
                        train_data["copy_label"][index][leaf_mask].float(),
                    )
                )
                copy_mask = (
                    leaf_mask
                    & train_data["copy_label"][index].bool()
                )
                if copy_mask.any():
                    loss = (
                        loss
                        + torch.nn.functional.cross_entropy(
                            logits["copy_start"][copy_mask],
                            train_data["copy_start"][index][copy_mask],
                        )
                        + torch.nn.functional.cross_entropy(
                            logits["copy_length"][copy_mask],
                            train_data["copy_length"][index][copy_mask],
                        )
                    )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(total / max(1, len(order)))
    model.eval()
    optimizer.eval()
    metrics = evaluate_teacher_forced(
        model,
        validation_batch,
        device,
    )
    metrics["loss_history"] = history
    metrics["train_events"] = len(train_batch.action)
    metrics["validation_events"] = len(validation_batch.action)
    return model, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: RealCorpusTreePolicy,
    batch: RealPolicyBatch,
    device: torch.device,
) -> dict[str, object]:
    data = _batch_tensors(batch, device)
    correct = {
        name: 0
        for name in ("action", "node_kind", "edge_field", "parent", "leaf_class")
    }
    total = {name: 0 for name in correct}
    probabilities = []
    budget_correct = 0
    for start in range(0, len(data["action"]), 2048):
        index = torch.arange(
            start,
            min(start + 2048, len(data["action"])),
            device=device,
        )
        prompt_row = data["event_prompt_rows"][index]
        previous = {
            name: data["previous"][name][index]
            for name in data["previous"]
        }
        action_history = {
            name: data["history"][name][index]
            for name in data["history"]
        }
        logits = model(
            data["prompt_token_ids"][prompt_row],
            data["prompt_token_mask"][prompt_row],
            data["prompts"][prompt_row],
            data["prompt_masks"][prompt_row],
            data["state_features"][index],
            data["context_kind"][index],
            data["context_edge"][index],
            data["candidate_kind"][index],
            data["candidate_scope"][index],
            data["candidate_position"][index],
            data["candidate_mask"][index],
            action_history,
            previous,
        )
        action = data["action"][index]
        checks = {
            "action": action,
            "node_kind": data["node_kind"][index],
            "edge_field": data["edge_field"][index],
            "parent": data["candidate_target"][index],
            "leaf_class": data["leaf_class"][index],
        }
        masks = {
            "action": torch.ones_like(action, dtype=torch.bool),
            "node_kind": action.eq(REAL_ACTIONS.index("open_node")),
            "edge_field": action.eq(REAL_ACTIONS.index("attach_edge")),
            "parent": action.eq(REAL_ACTIONS.index("attach_edge")),
            "leaf_class": action.eq(REAL_ACTIONS.index("declare_leaf")),
        }
        for name, target in checks.items():
            active = masks[name]
            if not active.any():
                continue
            predicted = logits[name][active].argmax(dim=-1)
            correct[name] += int(predicted.eq(target[active]).sum())
            total[name] += int(active.sum())
        probabilities.append(torch.softmax(logits["action"], dim=-1).max(dim=-1).values.cpu())
        budget_correct += int(
            logits["node_budget"]
            .argmax(dim=-1)
            .eq(data["node_budget"][index])
            .sum()
        )
    return {
        name: correct[name] / max(1, total[name])
        for name in correct
    } | {
        "action_confidence": float(torch.cat(probabilities).mean()),
        "node_budget_accuracy": budget_correct / max(1, len(data["action"])),
        "target_ast_input": False,
        "repair_or_retry": False,
    }

__all__ = [
    "RealCorpusTreePolicy",
    "evaluate_teacher_forced",
    "train_real_corpus_tree_policy",
]
