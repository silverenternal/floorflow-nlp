"""Typed leaf-class prediction with context copy and generation fallback."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.ast_leaf_classifier import classify_leaf
from floorflow.nlp_flow.engineering.schedule_free import ScheduleFreeAdamW


@dataclass(frozen=True)
class LeafExample:
    language: str
    node_kind: str
    leaf_class: str
    context: tuple[str, ...]
    pointer_label: int | None
    lexical: str


@dataclass(frozen=True)
class LeafTensorBatch:
    node_kind: torch.Tensor
    class_label: torch.Tensor
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    copy_label: torch.Tensor
    pointer_label: torch.Tensor
    lexical_label: torch.Tensor
    class_mask: torch.Tensor


class ASTLeafPointerHead(nn.Module):
    def __init__(
        self,
        *,
        node_kinds: int,
        leaf_classes: int,
        lexemes: int,
        hidden_size: int = 48,
        max_context: int = 128,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_kinds + 1, hidden_size)
        self.context_embedding = nn.Embedding(lexemes + 1, hidden_size)
        self.position_embedding = nn.Embedding(max_context, hidden_size)
        self.node_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.class_head = nn.Linear(hidden_size, leaf_classes)
        self.pointer_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pointer_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.copy_head = nn.Linear(hidden_size, 1)
        self.lexical_head = nn.Linear(hidden_size, lexemes)

    def forward(
        self,
        node_kind: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.node_projection(self.node_embedding(node_kind))
        positions = torch.arange(
            context_ids.shape[1],
            device=context_ids.device,
        )
        context = (
            self.context_embedding(context_ids)
            + self.position_embedding(positions)[None]
        )
        scores = torch.einsum(
            "bd,bld->bl",
            self.pointer_query(state),
            self.pointer_key(context),
        ) / state.shape[-1] ** 0.5
        scores = scores.masked_fill(~context_mask, -torch.inf)
        return {
            "class_logits": self.class_head(state),
            "pointer_logits": scores,
            "copy_logit": self.copy_head(state).squeeze(-1),
            "lexical_logits": self.lexical_head(state),
        }


def _leaves(payload: Mapping[str, object]):
    def visit(node: Mapping[str, object]):
        if not node["children"]:
            yield node
            return
        for child in node["children"]:
            yield from visit(child["node"])

    yield from visit(payload["root"])


def build_leaf_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    max_context: int = 128,
) -> tuple[list[LeafExample], list[LeafExample]]:
    train = []
    validation = []
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            continue
        payload = json.loads(str(row["ast"]))
        leaves = list(_leaves(payload))
        lexemes = [
            str(leaf["lexical"])
            for leaf in leaves
            if isinstance(leaf.get("lexical"), str)
        ]
        output = train if row["split"] == "train" else validation
        for index, leaf in enumerate(leaves):
            lexical = leaf.get("lexical")
            if not isinstance(lexical, str):
                continue
            context = [
                value
                for other_index, value in enumerate(lexemes)
                if other_index != index
            ][:max_context]
            pointer = context.index(lexical) if lexical in context else None
            output.append(
                LeafExample(
                    language=str(row["language"]),
                    node_kind=str(leaf["type"]),
                    leaf_class=classify_leaf(str(leaf["type"]), lexical),
                    context=tuple(context),
                    pointer_label=pointer,
                    lexical=lexical,
                )
            )
    return train, validation


def _category(values):
    ordered = sorted(set(values))
    return ordered, {value: index for index, value in enumerate(ordered)}


def encode_leaf_examples(
    train: Sequence[LeafExample],
    validation: Sequence[LeafExample],
    grammars: Mapping[str, ASTTransitionGrammar],
    *,
    max_context: int = 128,
) -> tuple[LeafTensorBatch, LeafTensorBatch, dict[str, object]]:
    node_names, node_ids = _category(
        f"{value.language}:{value.node_kind}" for value in train
    )
    class_names, class_ids = _category(value.leaf_class for value in train)
    lexeme_names, lexeme_ids = _category(value.lexical for value in train)
    all_class_bits = (1 << len(class_names)) - 1
    allowed_class_bits: dict[tuple[str, str], int] = {}

    def allowed_bits(value: LeafExample) -> int:
        key = (value.language, value.node_kind)
        if key not in allowed_class_bits:
            allowed = grammars[value.language].leaf_classes.get(
                value.node_kind,
                (),
            )
            bits = 0
            for class_name in allowed:
                class_id = class_ids.get(class_name)
                if class_id is not None:
                    bits |= 1 << class_id
            allowed_class_bits[key] = bits or all_class_bits
        return allowed_class_bits[key]

    def encode(examples: Sequence[LeafExample]) -> LeafTensorBatch:
        node_rows = []
        class_rows = []
        context_rows = []
        masks = []
        copy_rows = []
        pointer_rows = []
        lexical_rows = []
        class_mask_bits = []
        for value in examples:
            context = [
                lexeme_ids.get(token, 0)
                for token in value.context[:max_context]
            ]
            padding = max_context - len(context)
            node_rows.append(
                node_ids.get(f"{value.language}:{value.node_kind}", 0)
            )
            class_rows.append(class_ids.get(value.leaf_class, 0))
            context_rows.append(context + [0] * padding)
            masks.append([True] * len(context) + [False] * padding)
            copy_rows.append(value.pointer_label is not None)
            pointer_rows.append(value.pointer_label or 0)
            lexical_rows.append(lexeme_ids.get(value.lexical, 0))
            class_mask_bits.append(allowed_bits(value))
        class_bits = torch.tensor(
            [1 << index for index in range(len(class_names))],
            dtype=torch.long,
        )
        class_mask = (
            torch.tensor(class_mask_bits, dtype=torch.long)[:, None]
            & class_bits[None]
        ).bool()
        return LeafTensorBatch(
            node_kind=torch.tensor(node_rows, dtype=torch.long),
            class_label=torch.tensor(class_rows, dtype=torch.long),
            context_ids=torch.tensor(context_rows, dtype=torch.long),
            context_mask=torch.tensor(masks, dtype=torch.bool),
            copy_label=torch.tensor(copy_rows, dtype=torch.float32),
            pointer_label=torch.tensor(pointer_rows, dtype=torch.long),
            lexical_label=torch.tensor(lexical_rows, dtype=torch.long),
            class_mask=class_mask,
        )

    metadata = {
        "node_kinds": node_ids,
        "leaf_classes": class_ids,
        "lexemes": lexeme_ids,
        "node_names": node_names,
        "class_names": class_names,
        "lexeme_names": lexeme_names,
    }
    return encode(train), encode(validation), metadata


def _batch_to_device(batch: LeafTensorBatch, device: torch.device) -> LeafTensorBatch:
    return LeafTensorBatch(
        **{
            name: value.to(device)
            for name, value in vars(batch).items()
        }
    )


def train_leaf_pointer_head(
    train: Sequence[LeafExample],
    validation: Sequence[LeafExample],
    grammars: Mapping[str, ASTTransitionGrammar],
    *,
    device: torch.device,
    seed: int,
    epochs: int = 3,
    batch_size: int = 512,
    max_context: int = 128,
) -> tuple[ASTLeafPointerHead, dict[str, object]]:
    train_batch, validation_batch, metadata = encode_leaf_examples(
        train,
        validation,
        grammars,
        max_context=max_context,
    )
    model = ASTLeafPointerHead(
        node_kinds=len(metadata["node_names"]),
        leaf_classes=len(metadata["class_names"]),
        lexemes=len(metadata["lexeme_names"]),
        max_context=max_context,
    ).to(device)
    optimizer = ScheduleFreeAdamW(model.parameters())
    generator = torch.Generator().manual_seed(seed)
    train_batch = _batch_to_device(train_batch, device)
    validation_batch = _batch_to_device(validation_batch, device)
    history = []
    for _epoch in range(epochs):
        model.train()
        optimizer.train()
        order = torch.randperm(len(train), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(
                train_batch.node_kind[index],
                train_batch.context_ids[index],
                train_batch.context_mask[index],
            )
            pointer_mask = train_batch.copy_label[index].bool()
            class_logits = result["class_logits"].masked_fill(
                ~train_batch.class_mask[index],
                -torch.inf,
            )
            pointer = (
                result["pointer_logits"].new_zeros(())
                if not pointer_mask.any()
                else torch.nn.functional.cross_entropy(
                    result["pointer_logits"][pointer_mask],
                    train_batch.pointer_label[index][pointer_mask],
                )
            )
            loss = (
                torch.nn.functional.cross_entropy(
                    class_logits,
                    train_batch.class_label[index],
                )
                + torch.nn.functional.cross_entropy(
                    result["lexical_logits"],
                    train_batch.lexical_label[index],
                )
                + 0.5 * pointer
                + 0.2
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    result["copy_logit"],
                    train_batch.copy_label[index],
                )
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(total / max(1, len(train)))
    model.eval()
    optimizer.eval()
    metrics = evaluate_leaf_pointer_head(
        model,
        validation_batch,
        validation,
        metadata,
        grammars,
        device=device,
    )
    metrics["loss_history"] = history
    metrics["train_examples"] = len(train)
    metrics["validation_examples"] = len(validation)
    return model, metrics


@torch.no_grad()
def evaluate_leaf_pointer_head(
    model: ASTLeafPointerHead,
    batch: LeafTensorBatch,
    examples: Sequence[LeafExample],
    metadata: Mapping[str, object],
    grammars: Mapping[str, ASTTransitionGrammar],
    *,
    device: torch.device,
) -> dict[str, object]:
    result = model(
        batch.node_kind.to(device),
        batch.context_ids.to(device),
        batch.context_mask.to(device),
    )
    class_names = metadata["class_names"]
    predicted_classes = result["class_logits"].masked_fill(
        ~batch.class_mask.to(device),
        -torch.inf,
    ).argmax(dim=-1)
    class_accuracy = (
        predicted_classes.eq(batch.class_label).float().mean().item()
    )
    class_valid = []
    for index, example in enumerate(examples):
        predicted = class_names[int(predicted_classes[index].item())]
        allowed = grammars[example.language].leaf_classes.get(
            example.node_kind,
            (),
        )
        class_valid.append(not allowed or predicted in allowed)
    applicable = batch.copy_label.bool()
    pointer_accuracy = 1.0
    if applicable.any():
        pointer_predictions = result["pointer_logits"][
            applicable.to(device)
        ].argmax(dim=-1)
        pointer_accuracy = (
            pointer_predictions.eq(batch.pointer_label[applicable])
            .float()
            .mean()
            .item()
        )
    copy_predictions = result["copy_logit"].ge(0.0)
    generated = result["lexical_logits"].argmax(dim=-1)
    pointer_ids = result["pointer_logits"].argmax(dim=-1)
    lexeme_names = metadata["lexeme_names"]
    predicted_lexemes = []
    for index, example in enumerate(examples):
        if copy_predictions[index] and example.context:
            predicted_lexemes.append(
                example.context[int(pointer_ids[index].item())]
            )
        else:
            predicted_lexemes.append(
                lexeme_names[int(generated[index].item())]
            )
    lexical_accuracy = sum(
        prediction == example.lexical
        for prediction, example in zip(predicted_lexemes, examples, strict=True)
    ) / max(1, len(examples))
    return {
        "class_accuracy": class_accuracy,
        "class_grammar_validity": sum(class_valid) / max(1, len(class_valid)),
        "copy_applicable": int(applicable.sum()),
        "copy_pointer_accuracy": pointer_accuracy,
        "copy_decision_accuracy": (
            copy_predictions.eq(batch.copy_label.bool()).float().mean().item()
        ),
        "lexical_accuracy": lexical_accuracy,
        "parser_repair": False,
        "retry_or_best_of_n": False,
    }


__all__ = [
    "ASTLeafPointerHead",
    "LeafExample",
    "LeafTensorBatch",
    "build_leaf_examples",
    "encode_leaf_examples",
    "evaluate_leaf_pointer_head",
    "train_leaf_pointer_head",
]
