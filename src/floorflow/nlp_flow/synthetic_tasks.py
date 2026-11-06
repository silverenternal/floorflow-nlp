"""Deterministic plan/segment/symbol/constraint tasks for CPU probes."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

TOPICS = ("allocate", "cache", "compile", "parse", "schedule", "sort")
INTENTS = ("explain", "implement", "refactor", "verify")


@dataclass(frozen=True)
class SyntheticExample:
    id: str
    split: str
    source_text: str
    target_text: str
    oracle_plan: dict[str, Any]
    constraints: dict[str, Any]

    def inference_view(self) -> dict[str, str]:
        """Return only fields that are legal before target generation."""

        return {
            "id": self.id,
            "split": self.split,
            "source_text": self.source_text,
        }

    def training_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "constraints": self.constraints,
        }


def _rng(seed: int, split: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{split}:{index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _one(seed: int, split: str, index: int) -> SyntheticExample:
    rng = _rng(seed, split, index)
    intent = rng.choice(INTENTS)
    topic = rng.choice(TOPICS)
    segment_count = rng.randint(2, 8)
    entity_count = segment_count
    entities = tuple(f"e{value}" for value in rng.sample(range(100), entity_count))
    segments = []
    for segment_index in range(segment_count):
        source = entities[segment_index % len(entities)]
        destination = entities[(segment_index + 1) % len(entities)]
        relation = "depends" if segment_index % 2 == 0 else "feeds"
        segments.append((source, relation, destination))

    target_text = " ".join(
        [
            *(token for source, relation, destination in segments for token in ("SEG", source, relation, destination)),
            "END",
        ]
    )
    exact_tokens = len(target_text.split())
    source_text = (
        f"{intent} a {topic} plan. "
        f"Entities: {', '.join(entities)}. "
        f"Relations: {'; '.join(f'{a} {relation} {b}' for a, relation, b in segments)}. "
        f"Keep segment order and emit exactly {exact_tokens} whitespace tokens."
    )
    example_id = hashlib.sha256(f"{split}:{index}:{seed}".encode()).hexdigest()[:20]
    return SyntheticExample(
        id=example_id,
        split=split,
        source_text=source_text,
        target_text=target_text,
        oracle_plan={
            "intent": intent,
            "topic": topic,
            "required_entities": list(entities),
            "entity_relations": [
                {
                    "source": source,
                    "relation": relation,
                    "destination": destination,
                }
                for source, relation, destination in segments
            ],
            "segment_order": list(range(segment_count)),
            "exact_token_budget": exact_tokens,
        },
        constraints={
            "all_required_entities_present": True,
            "relation_direction_preserved": True,
            "segment_order_preserved": True,
            "exact_token_budget": exact_tokens,
            "grammar_valid": True,
        },
    )


def generate_examples(
    *,
    split: str,
    count: int,
    seed: int,
) -> Iterator[SyntheticExample]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if count < 0:
        raise ValueError("count must be non-negative")
    for index in range(count):
        yield _one(seed, split, index)


def verify_synthetic_output(
    example: SyntheticExample,
    output_text: str,
) -> dict[str, bool]:
    """Check all executable constraints without consulting model state."""

    tokens = output_text.split()
    expected_segments = [
        (
            relation["source"],
            relation["relation"],
            relation["destination"],
        )
        for relation in example.oracle_plan["entity_relations"]
    ]
    actual_segments = []
    valid_grammar = bool(tokens) and tokens[-1] == "END"
    cursor = 0
    while valid_grammar and cursor < len(tokens) - 1:
        if cursor + 3 >= len(tokens) or tokens[cursor] != "SEG":
            valid_grammar = False
            break
        actual_segments.append((tokens[cursor + 1], tokens[cursor + 2], tokens[cursor + 3]))
        cursor += 4
    required_entities = set(example.oracle_plan["required_entities"])
    return {
        "all_required_entities_present": required_entities.issubset(set(tokens)),
        "relation_direction_preserved": actual_segments == expected_segments,
        "segment_order_preserved": actual_segments == expected_segments,
        "exact_token_budget_met": len(tokens) == int(example.constraints["exact_token_budget"]),
        "grammar_valid": valid_grammar,
    }


def deterministic_attacks(example: SyntheticExample) -> dict[str, str]:
    tokens = example.target_text.split()
    reverse_relation = list(tokens)
    if len(tokens) >= 4:
        reverse_relation[1], reverse_relation[3] = reverse_relation[3], reverse_relation[1]
    return {
        "truncate_end": " ".join(tokens[:-1]),
        "extra_token": example.target_text + " EXTRA",
        "reverse_first_relation": " ".join(reverse_relation),
        "drop_first_entity": " ".join(
            token
            for index, token in enumerate(tokens)
            if index != 1
        ),
    }


def plan_is_hidden(payload: Mapping[str, Any]) -> bool:
    forbidden = {"oracle_plan", "target_text", "constraints"}
    return not (set(payload) & forbidden)


__all__ = [
    "SyntheticExample",
    "deterministic_attacks",
    "generate_examples",
    "plan_is_hidden",
    "verify_synthetic_output",
]
