"""Supervised legal and illegal AST transition learning."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW

SPECIAL_START = "<start>"
SPECIAL_NONE = "<none>"
INVALID_ROOT = "__invalid_root__"
INVALID_CHILD = "__unknown_child__"
INVALID_FIELD = "__unknown_field__"


@dataclass(frozen=True)
class ASTGrammarExample:
    language: str
    action: str
    parent: str
    edge: str
    child: str
    label: int
    source: str

    def key(self) -> str:
        value = (
            f"{self.language}|{self.action}|{self.parent}|"
            f"{self.edge}|{self.child}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ASTGrammarLegalityHead(nn.Module):
    def __init__(
        self,
        *,
        languages: Sequence[str],
        actions: Sequence[str],
        parents: Sequence[str],
        edges: Sequence[str],
        children: Sequence[str],
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.language_embedding = nn.Embedding(len(languages) + 1, hidden_size)
        self.action_embedding = nn.Embedding(len(actions) + 1, hidden_size)
        self.parent_embedding = nn.Embedding(len(parents) + 1, hidden_size)
        self.edge_embedding = nn.Embedding(len(edges) + 1, hidden_size)
        self.child_embedding = nn.Embedding(len(children) + 1, hidden_size)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        language: torch.Tensor,
        action: torch.Tensor,
        parent: torch.Tensor,
        edge: torch.Tensor,
        child: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self.language_embedding(language)
            + self.action_embedding(action)
            + self.parent_embedding(parent)
            + self.edge_embedding(edge)
            + self.child_embedding(child)
        )
        return self.output(hidden).squeeze(-1)


@dataclass(frozen=True)
class EncodedGrammarExamples:
    columns: dict[str, torch.Tensor]
    labels: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    category_indices: dict[str, dict[str, int]]


def build_grammar_examples(
    grammars: Mapping[str, ASTTransitionGrammar],
    negative_cases: Sequence[Mapping[str, object]],
) -> list[ASTGrammarExample]:
    examples = []
    for language, grammar in sorted(grammars.items()):
        children = sorted({value.child_kind for value in grammar.transitions})
        allowed_children: dict[tuple[str, str], set[str]] = {}
        for transition in grammar.transitions:
            allowed_children.setdefault(
                (transition.parent_kind, transition.edge_field),
                set(),
            ).add(transition.child_kind)
        for root in grammar.legal_root_kinds:
            examples.append(
                ASTGrammarExample(
                    language,
                    "root",
                    SPECIAL_START,
                    "*",
                    root,
                    1,
                    "parser_schema",
                )
            )
            examples.append(
                ASTGrammarExample(
                    language,
                    "root",
                    SPECIAL_START,
                    "*",
                    INVALID_ROOT,
                    0,
                    "negative",
                )
            )
        for index, transition in enumerate(grammar.transitions):
            examples.append(
                ASTGrammarExample(
                    language,
                    "child",
                    transition.parent_kind,
                    transition.edge_field,
                    transition.child_kind,
                    1,
                    "parser_schema_or_train",
                )
            )
            if index % 4:
                continue
            examples.append(
                ASTGrammarExample(
                    language,
                    "child",
                    transition.parent_kind,
                    transition.edge_field,
                    INVALID_CHILD,
                    0,
                    "negative",
                )
            )
            examples.append(
                ASTGrammarExample(
                    language,
                    "child",
                    transition.parent_kind,
                    INVALID_FIELD,
                    transition.child_kind,
                    0,
                    "negative",
                )
            )
            relation_children = allowed_children[
                (transition.parent_kind, transition.edge_field)
            ]
            hard_child = next(
                (
                    value
                    for value in children
                    if value not in relation_children
                ),
                None,
            )
            if hard_child is not None:
                examples.append(
                    ASTGrammarExample(
                        language,
                        "child",
                        transition.parent_kind,
                        transition.edge_field,
                        hard_child,
                        0,
                        "hard_negative",
                    )
                )
        examples.extend(
            (
                ASTGrammarExample(language, "close_scope", "*", "*", "*", 1, "state"),
                ASTGrammarExample(language, "eos", "*", "*", "*", 1, "state"),
                ASTGrammarExample(language, "eos", "*", "*", "*", 0, "negative"),
            )
        )
    for case in negative_cases:
        language = str(case["language"])
        kind = str(case["kind"])
        state = case.get("state", {})
        action = "semantic" if kind == "valid_but_semantically_wrong" else kind
        examples.append(
            ASTGrammarExample(
                language,
                action,
                str(state.get("parent", state.get("scope_id", "*"))),
                str(state.get("field", state.get("observed_field", "*"))),
                str(state.get("observed_child", state.get("child", "*"))),
                0,
                "negative_suite",
            )
        )
    return examples


def encode_grammar_examples(
    examples: Sequence[ASTGrammarExample],
) -> EncodedGrammarExamples:
    categories = {
        name: {
            value
            for example in examples
            if (value := getattr(example, name)) is not None
        }
        for name in ("language", "action", "parent", "edge", "child")
    }
    indices = {
        name: {value: index for index, value in enumerate(sorted(values))}
        for name, values in categories.items()
    }
    columns = {
        name: torch.tensor(
            [indices[name][getattr(example, name)] for example in examples],
            dtype=torch.long,
        )
        for name in ("language", "action", "parent", "edge", "child")
    }
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    validation = torch.tensor(
        [
            int(example.key()[:2], 16) % 5 == 0
            for example in examples
        ],
        dtype=torch.bool,
    )
    train_indices = (~validation).nonzero(as_tuple=False).flatten()
    validation_indices = validation.nonzero(as_tuple=False).flatten()
    return EncodedGrammarExamples(
        columns,
        labels,
        train_indices,
        validation_indices,
        indices,
    )


def encode_examples_with_indices(
    examples: Sequence[ASTGrammarExample],
    indices: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    columns = {
        name: torch.tensor(
            [indices[name][getattr(example, name)] for example in examples],
            dtype=torch.long,
        )
        for name in ("language", "action", "parent", "edge", "child")
    }
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    return columns, labels


def train_grammar_legality_head(
    examples: Sequence[ASTGrammarExample],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[ASTGrammarLegalityHead, dict[str, object]]:
    encoded = encode_grammar_examples(examples)
    model = ASTGrammarLegalityHead(
        languages=range(len(encoded.category_indices["language"])),
        actions=range(len(encoded.category_indices["action"])),
        parents=range(len(encoded.category_indices["parent"])),
        edges=range(len(encoded.category_indices["edge"])),
        children=range(len(encoded.category_indices["child"])),
    ).to(device)
    optimizer = ScheduleFreeAdamW(model.parameters())
    generator = torch.Generator().manual_seed(seed)
    columns = {name: values.to(device) for name, values in encoded.columns.items()}
    labels = encoded.labels.to(device)
    train_indices = encoded.train_indices
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(columns[name][index] for name in columns))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels[index],
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(total / max(1, len(order)))
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        validation = encoded.validation_indices.to(device)
        logits = model(*(columns[name][validation] for name in columns))
        probabilities = torch.sigmoid(logits)
        expected = labels[validation]
        predictions = probabilities.ge(0.5)
        invalid_false_accepts = int(
            (predictions & expected.lt(0.5)).sum()
        )
        metrics = {
            "examples": len(examples),
            "train_examples": len(encoded.train_indices),
            "validation_examples": len(encoded.validation_indices),
            "epochs": epochs,
            "loss_history": history,
            "validation_accuracy": float(
                predictions.eq(expected.bool()).float().mean()
            ),
            "invalid_false_accepts": invalid_false_accepts,
        }
    return model, metrics


__all__ = [
    "ASTGrammarExample",
    "ASTGrammarLegalityHead",
    "build_grammar_examples",
    "encode_examples_with_indices",
    "encode_grammar_examples",
    "train_grammar_legality_head",
]
