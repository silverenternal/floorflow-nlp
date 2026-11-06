"""Native JSON grammar state and typed action heads."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from floorflow.nlp_flow._generated_protocol import JSON_ACTION_NAMES

ACTIONS = JSON_ACTION_NAMES
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
FRAME_KINDS = ("root", "object", "array")
FRAME_PHASES = (
    "value",
    "key_or_end",
    "colon",
    "value_or_end",
    "comma_or_end",
    "complete",
)
GRAMMAR_FEATURE_DIM = len(FRAME_KINDS) + len(FRAME_PHASES) + 3
SCHEMA_KEY_FEATURE_DIM = 64


@dataclass(frozen=True)
class Frame:
    kind: str
    phase: str
    has_value: bool = False


@dataclass(frozen=True)
class JSONGrammarState:
    frames: tuple[Frame, ...] = (Frame("root", "value", False),)

    @property
    def complete(self) -> bool:
        return len(self.frames) == 1 and self.frames[0].phase == "complete"

    @classmethod
    def initial(cls) -> JSONGrammarState:
        return cls()

    def allowed_actions(self) -> tuple[str, ...]:
        frame = self.frames[-1]
        if frame.kind == "root":
            if frame.phase == "value":
                return ("start_object", "start_array", "string", "number", "true", "false", "null")
            return ()
        if frame.kind == "object":
            if frame.phase == "key_or_end":
                return ("key", "end_object")
            if frame.phase == "colon":
                return ("colon",)
            if frame.phase == "value":
                return ("start_object", "start_array", "string", "number", "true", "false", "null")
            if frame.phase == "comma_or_end":
                return ("comma", "end_object")
        if frame.kind == "array":
            if frame.phase == "value_or_end":
                return ("start_object", "start_array", "string", "number", "true", "false", "null", "end_array")
            if frame.phase == "comma_or_end":
                return ("comma", "end_array")
        return ()

    def transition(self, action: str) -> JSONGrammarState:
        if action not in ACTIONS:
            raise ValueError(f"unknown JSON action: {action}")
        if action not in self.allowed_actions():
            raise ValueError(f"invalid JSON transition {action} from {self.frames[-1]}")
        frames = list(self.frames)
        frame = frames[-1]
        if action in {"start_object", "start_array"}:
            kind = "object" if action == "start_object" else "array"
            phase = "key_or_end" if kind == "object" else "value_or_end"
            frames.append(Frame(kind, phase, False))
        elif action in {"string", "number", "true", "false", "null"}:
            frames[-1] = _after_value(frame)
        elif action == "key":
            frames[-1] = Frame("object", "colon", frame.has_value)
        elif action == "colon":
            frames[-1] = Frame("object", "value", True)
        elif action == "comma":
            if frame.kind == "object":
                frames[-1] = Frame("object", "key_or_end", True)
            else:
                frames[-1] = Frame("array", "value_or_end", True)
        elif action == "end_object":
            if frame.kind != "object":
                raise ValueError("end_object outside object")
            frames.pop()
            frames[-1] = _after_value(frames[-1])
        elif action == "end_array":
            if frame.kind != "array":
                raise ValueError("end_array outside array")
            frames.pop()
            frames[-1] = _after_value(frames[-1])
        return JSONGrammarState(tuple(frames))


@dataclass(frozen=True)
class SchemaNode:
    kind: str
    required: tuple[str, ...] = ()
    properties: Mapping[str, SchemaNode] = field(default_factory=dict)
    items: SchemaNode | None = None
    enum_values: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    max_items: int | None = 16


@dataclass(frozen=True)
class SchemaGrammarState:
    grammar: JSONGrammarState
    schemas: tuple[SchemaNode, ...]
    seen: tuple[frozenset[str], ...]
    pending_key: str | None = None

    @classmethod
    def initial(cls, schema: SchemaNode) -> SchemaGrammarState:
        return cls(JSONGrammarState.initial(), (schema,), (frozenset(),))

    @property
    def complete(self) -> bool:
        return self.grammar.complete

    def allowed_actions(self) -> tuple[str, ...]:
        actions = self.grammar.allowed_actions()
        current = self.schemas[-1]
        frame = self.grammar.frames[-1]
        if frame.kind == "object" and frame.phase == "key_or_end" and frame.has_value:
            actions = tuple(action for action in actions if action != "end_object")
        if frame.kind == "array" and frame.phase == "value_or_end" and frame.has_value:
            actions = tuple(action for action in actions if action != "end_array")
        if (
            frame.kind == "array"
            and frame.phase == "value_or_end"
            and current.max_items is not None
            and sum(value.startswith("@item:") for value in self.seen[-1]) >= current.max_items
        ):
            actions = tuple(action for action in actions if action == "end_array")
        if (
            frame.kind == "array"
            and frame.phase == "comma_or_end"
            and current.max_items is not None
            and sum(value.startswith("@item:") for value in self.seen[-1]) >= current.max_items
        ):
            actions = tuple(action for action in actions if action != "comma")
        if (
            frame.kind == "object"
            and frame.phase == "comma_or_end"
            and set(current.properties).issubset(self.seen[-1])
        ):
            actions = tuple(action for action in actions if action != "comma")
        allowed = []
        for action in actions:
            if action == "end_object":
                if set(current.required) - set(self.seen[-1]):
                    continue
            elif action == "key":
                if set(current.properties).issubset(self.seen[-1]):
                    continue
            elif action in {"start_object", "start_array"}:
                if frame.kind not in {"root", "object", "array"}:
                    continue
                if frame.kind != "root":
                    expected = self._child_schema()
                    actual = "object" if action == "start_object" else "array"
                    if expected.kind != actual:
                        continue
            elif action in {"string", "number", "true", "false", "null"}:
                expected = (
                    self._child_schema().kind
                    if frame.kind in {"object", "array"}
                    else current.kind
                )
                actual = "boolean" if action in {"true", "false"} else action
                if not (actual == expected or (actual == "number" and expected == "integer")):
                    continue
            allowed.append(action)
        return tuple(allowed)

    def allowed_keys(self) -> tuple[str, ...]:
        current = self.schemas[-1]
        frame = self.grammar.frames[-1]
        if current.kind != "object" or frame.phase != "key_or_end":
            return ()
        return tuple(
            key
            for key in current.properties
            if key not in self.seen[-1]
        )

    def _child_schema(self) -> SchemaNode:
        current = self.schemas[-1]
        if current.kind == "object":
            if self.pending_key is None or self.pending_key not in current.properties:
                raise ValueError("object value has no valid schema property")
            return current.properties[self.pending_key]
        if current.kind == "array" and current.items is not None:
            return current.items
        raise ValueError("current schema cannot contain a nested value")

    def transition(
        self,
        action: str,
        *,
        key: str | None = None,
    ) -> SchemaGrammarState:
        current = self.schemas[-1]
        if action == "key":
            if current.kind != "object" or key is None:
                raise ValueError("key action requires an object schema and key")
            if key not in self.allowed_keys():
                if key in self.seen[-1]:
                    raise ValueError(f"duplicate schema key: {key}")
                raise ValueError(f"schema does not allow property: {key}")
        if action in {"string", "number", "true", "false", "null"}:
            expected = (
                self._child_schema().kind
                if current.kind in {"object", "array"}
                else current.kind
            )
            actual = "boolean" if action in {"true", "false"} else action
            if not (actual == expected or (actual == "number" and expected == "integer")):
                raise ValueError(f"schema expected {expected}, got {actual}")
        if action in {"start_object", "start_array"}:
            expected = current if self.grammar.frames[-1].kind == "root" else self._child_schema()
            actual = "object" if action == "start_object" else "array"
            if expected.kind != actual:
                raise ValueError(f"schema expected {expected.kind}, got {actual}")
        if action == "end_object":
            missing = set(current.required) - set(self.seen[-1])
            if missing:
                raise ValueError(f"schema missing required fields: {sorted(missing)}")
        if action not in self.allowed_actions():
            raise ValueError(f"invalid schema JSON transition {action}")
        grammar = self.grammar.transition(action)
        schemas = list(self.schemas)
        seen = list(self.seen)
        pending = self.pending_key
        if action == "key":
            pending = key
        elif action == "colon":
            pending = self.pending_key
        elif action in {"start_object", "start_array"}:
            child = (
                current
                if self.grammar.frames[-1].kind == "root"
                else self._child_schema()
            )
            if current.kind == "object" and pending is not None:
                seen[-1] = seen[-1] | {pending}
                if child.kind != ("object" if action == "start_object" else "array"):
                    raise ValueError(
                        f"schema expected {child.kind} for nested value"
                    )
            schemas.append(child)
            seen.append(frozenset())
            pending = None
        elif action in {"string", "number", "true", "false", "null"}:
            expected = (
                self._child_schema().kind
                if current.kind in {"object", "array"}
                else current.kind
            )
            actual = "boolean" if action in {"true", "false"} else action
            if not (actual == expected or (actual == "number" and expected == "integer")):
                raise ValueError(f"schema expected {expected}, got {actual}")
            if current.kind == "object" and pending is not None:
                seen[-1] = seen[-1] | {pending}
                pending = None
            elif current.kind == "array":
                count = sum(value.startswith("@item:") for value in seen[-1])
                seen[-1] = seen[-1] | {f"@item:{count}"}
        elif action in {"end_object", "end_array"}:
            expected_kind = "object" if action == "end_object" else "array"
            if current.kind != expected_kind:
                raise ValueError("schema type mismatch on close")
            missing = set(current.required) - set(seen[-1])
            if missing:
                raise ValueError(f"schema missing required fields: {sorted(missing)}")
            if len(schemas) > 1:
                schemas.pop()
                seen.pop()
                parent = schemas[-1]
                if parent.kind == "array":
                    count = sum(value.startswith("@item:") for value in seen[-1])
                    seen[-1] = seen[-1] | {f"@item:{count}"}
            pending = None
        elif action == "comma":
            pending = None
        return SchemaGrammarState(grammar, tuple(schemas), tuple(seen), pending)


def _after_value(frame: Frame) -> Frame:
    if frame.kind == "root":
        return Frame("root", "complete", True)
    if frame.kind == "object":
        return Frame("object", "comma_or_end", True)
    if frame.kind == "array":
        return Frame("array", "comma_or_end", True)
    raise ValueError(f"unsupported JSON frame: {frame}")


class JSONActionHead(nn.Module):
    """Action logits with invalid grammar actions structurally masked."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, len(ACTIONS))
        self.state_proj = nn.Linear(GRAMMAR_FEATURE_DIM, len(ACTIONS), bias=False)
        self.step_embedding = nn.Embedding(64, len(ACTIONS))
        nn.init.zeros_(self.step_embedding.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        states: list[JSONGrammarState | SchemaGrammarState],
        step_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.proj(hidden)
        if logits.ndim != 2 or logits.shape[0] != len(states):
            raise ValueError("JSON action head expects [batch, hidden] and batch states")
        mask = torch.full_like(logits, float("-inf"))
        features = []
        for row, state in enumerate(states):
            grammar = state.grammar if isinstance(state, SchemaGrammarState) else state
            frame = grammar.frames[-1]
            feature = torch.zeros(
                GRAMMAR_FEATURE_DIM,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            feature[FRAME_KINDS.index(frame.kind)] = 1.0
            phase_offset = len(FRAME_KINDS)
            feature[phase_offset + FRAME_PHASES.index(frame.phase)] = 1.0
            offset = phase_offset + len(FRAME_PHASES)
            feature[offset] = min(len(grammar.frames), 4) / 4.0
            feature[offset + 1] = float(frame.has_value)
            feature[offset + 2] = float(grammar.complete)
            features.append(feature)
            allowed = state.allowed_actions()
            if allowed:
                indices = [ACTION_INDEX[action] for action in allowed]
                mask[row, indices] = 0.0
        if step_ids is None:
            step_ids = torch.zeros(
                logits.shape[0],
                dtype=torch.long,
                device=logits.device,
            )
        if step_ids.shape != (logits.shape[0],):
            raise ValueError("step_ids must be [batch]")
        if step_ids.min() < 0 or step_ids.max() >= self.step_embedding.num_embeddings:
            raise ValueError("step id is outside the action-position range")
        state_logits = (
            logits
            + self.state_proj(torch.stack(features))
            + self.step_embedding(step_ids)
        )
        return state_logits + mask


class JSONTypedHeads(nn.Module):
    """Eight independently trainable JSON structural/type heads."""

    NAMES = (
        "object_open_close",
        "property_key",
        "array_open_close",
        "string_value",
        "number_value",
        "boolean_value",
        "null_value",
        "schema_reference",
    )

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, 2) for name in self.NAMES}
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: self.heads[name](hidden) for name in self.NAMES}


def schema_key_features(key: str) -> torch.Tensor:
    """Deterministic character n-gram features for a schema property name."""

    if not key:
        raise ValueError("schema key cannot be empty")
    values = torch.zeros(SCHEMA_KEY_FEATURE_DIM)
    padded = f"<{key}>"
    for width in (1, 2, 3):
        for index in range(max(1, len(padded) - width + 1)):
            gram = padded[index : index + width]
            digest = hashlib.blake2b(
                gram.encode("utf-8"),
                digest_size=8,
            ).digest()
            bucket = int.from_bytes(digest[:4], "little")
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket % SCHEMA_KEY_FEATURE_DIM] += sign
    return values / math.sqrt(max(1.0, float(values.square().sum())))


class JSONSchemaKeyHead(nn.Module):
    """Score only schema-allowed property names for the current object frame."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.query = nn.Linear(hidden_size, SCHEMA_KEY_FEATURE_DIM)
        self.key_projection = nn.Linear(
            SCHEMA_KEY_FEATURE_DIM,
            SCHEMA_KEY_FEATURE_DIM,
            bias=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        states: list[SchemaGrammarState],
    ) -> tuple[torch.Tensor, ...]:
        if hidden.ndim != 2 or hidden.shape[0] != len(states):
            raise ValueError("schema key head expects [batch, hidden] and states")
        query = self.query(self.norm(hidden))
        output = []
        for row, state in enumerate(states):
            keys = state.allowed_keys()
            if not keys:
                output.append(hidden.new_empty(0))
                continue
            features = torch.stack(
                [schema_key_features(key) for key in keys]
            ).to(hidden.device, hidden.dtype)
            output.append(
                (query[row] * self.key_projection(features)).sum(dim=-1)
            )
        return tuple(output)


def schema_key_loss(
    logits: tuple[torch.Tensor, ...],
    states: list[SchemaGrammarState],
    target_keys: list[str | None],
) -> torch.Tensor:
    if len(logits) != len(states) or len(states) != len(target_keys):
        raise ValueError("schema key logits, states, and targets must align")
    losses = []
    for row, (row_logits, state, target) in enumerate(
        zip(logits, states, target_keys)
    ):
        if target is None:
            continue
        keys = state.allowed_keys()
        if target not in keys:
            raise ValueError(f"row {row} target key is not schema-allowed: {target}")
        losses.append(
            torch.nn.functional.cross_entropy(
                row_logits[None],
                torch.tensor([keys.index(target)], device=row_logits.device),
            )
        )
    if not losses:
        raise ValueError("schema key targets have no active rows")
    return torch.stack(losses).mean()


def validate_json_action_sequence(actions: list[str]) -> JSONGrammarState:
    state = JSONGrammarState.initial()
    for action in actions:
        state = state.transition(action)
    if not state.complete:
        raise ValueError("JSON action sequence is incomplete")
    return state


def json_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[-1] != len(ACTIONS):
        raise ValueError("logits must be [batch, steps, actions]")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must be [batch, steps]")
    active = (
        torch.ones_like(targets, dtype=torch.bool)
        if mask is None
        else mask.bool()
    )
    if active.shape != targets.shape:
        raise ValueError("mask must match targets")
    for row in range(targets.shape[0]):
        state = JSONGrammarState.initial()
        for step in range(targets.shape[1]):
            if not bool(active[row, step]):
                continue
            action = ACTIONS[int(targets[row, step])]
            state = state.transition(action)
    if not active.any():
        raise ValueError("JSON action mask has no active positions")
    return torch.nn.functional.cross_entropy(
        logits[active],
        targets[active],
    )


__all__ = [
    "ACTIONS",
    "ACTION_INDEX",
    "FRAME_KINDS",
    "FRAME_PHASES",
    "GRAMMAR_FEATURE_DIM",
    "SCHEMA_KEY_FEATURE_DIM",
    "Frame",
    "JSONActionHead",
    "JSONGrammarState",
    "JSONSchemaKeyHead",
    "JSONTypedHeads",
    "SchemaGrammarState",
    "SchemaNode",
    "json_action_loss",
    "schema_key_features",
    "schema_key_loss",
    "validate_json_action_sequence",
]
