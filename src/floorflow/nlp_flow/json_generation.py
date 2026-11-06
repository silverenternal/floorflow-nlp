"""Native grammar-constrained JSON generation from the compact decoder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from tokenizers import Tokenizer

from floorflow.nlp_flow.model.json_grammar import (
    ACTION_INDEX,
    JSONGrammarState,
    SchemaGrammarState,
)
from floorflow.nlp_flow.native_json_contract import (
    NativeJSONDecision,
    commit_native_json_decisions,
    output_schema,
)
from floorflow.nlp_flow.structured_generation import (
    StructuredGenerationResult,
)
from floorflow.nlp_flow.typed_tree import TypedTreeNode

OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "ast": ("language", "nodes", "edges"),
    "patch": ("files", "summary"),
    "tool_action": ("tool", "arguments"),
    "diagnostic": ("diagnosis", "recommended_action", "evidence"),
    "engineering_result": ("status", "tests", "measurements"),
}
VALUE_ACTIONS = (
    "start_object",
    "start_array",
    "string",
    "number",
    "true",
    "false",
    "null",
)


@dataclass(frozen=True)
class GeneratedJSON:
    text: str
    value: dict[str, Any]
    actions: tuple[str, ...]
    invalid_transition_probability: float
    typed_tree: TypedTreeNode
    structured: StructuredGenerationResult
    decisions: tuple[NativeJSONDecision, ...]


def gold_action_sequence(kind: str) -> tuple[str, ...]:
    if kind not in OUTPUT_FIELDS:
        raise ValueError(f"unknown JSON output kind: {kind}")
    actions = ["start_object", "key", "colon", "string"]
    actions.extend(["comma", "key", "colon", "string"])
    actions.extend(["comma", "key", "colon", "start_object"])
    fields = OUTPUT_FIELDS[kind]
    for index, _field in enumerate(fields):
        actions.extend(["key", "colon", "string"])
        if index + 1 < len(fields):
            actions.append("comma")
    actions.extend(["end_object", "end_object"])
    return tuple(actions)


def validate_action_sequence(kind: str, actions: tuple[str, ...]) -> None:
    state = JSONGrammarState.initial()
    for action in actions:
        state = state.transition(action)
    if not state.complete:
        raise ValueError(f"{kind} action sequence is incomplete")


def _value_for_action(
    action: str,
    *,
    field: str,
    schema,
    typed,
    hidden: torch.Tensor,
) -> Any:
    string_variants = ("generated", "diagnostic", "validated", "model-output")
    string_index = 0
    for name in (
        "object_open_close",
        "property_key",
        "array_open_close",
        "string_value",
        "number_value",
    ):
        string_index = (
            string_index * 2
            + int(typed[name][0, 1] > typed[name][0, 0])
        )
    if action == "start_object":
        return {}
    if action == "start_array":
        return []
    if action == "string":
        if schema.enum_values:
            return schema.enum_values[0]
        return f"{field}:{string_variants[string_index % len(string_variants)]}"
    if action == "number":
        sign = int(typed["number_value"][0, 1] > typed["number_value"][0, 0])
        raw = float(sign) + float(hidden[0, :2].sum().abs() * 1.0e-6)
        if schema.kind == "integer":
            lower = int(schema.minimum or 0)
            upper = int(schema.maximum if schema.maximum is not None else lower + 100)
            return min(upper, max(lower, int(raw * 100) + lower))
        if schema.minimum is not None:
            raw = max(raw, schema.minimum)
        if schema.maximum is not None:
            raw = min(raw, schema.maximum)
        return raw
    if action in {"true", "false"}:
        return action == "true"
    if action == "null":
        return None
    raise ValueError(f"action cannot materialize a scalar value: {action}")


class NativeJSONGenerator:
    """Generate required schema fields with grammar-masked model choices."""

    def __init__(self, model: torch.nn.Module, tokenizer: Tokenizer) -> None:
        if getattr(model, "json_head", None) is None:
            raise ValueError("model must enable the native JSON head")
        if getattr(model, "json_schema_key_head", None) is None:
            raise ValueError("model must enable the native schema-key head")
        self.model = model
        self.tokenizer = tokenizer

    def _hidden(self, prompt: str, max_tokens: int) -> torch.Tensor:
        context_limit = int(self.model.config.max_seq_len) - 2
        max_tokens = min(int(max_tokens), context_limit)
        if max_tokens <= 0:
            raise ValueError("model context must leave room for special tokens")
        ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).ids
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        if bos is None or eos is None:
            raise ValueError("tokenizer needs <bos> and <eos>")
        ids = [bos, *ids[-max_tokens:], eos]
        tensor = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        return self.model(tensor).hidden_states[:, -1]

    def _choose_action(
        self,
        hidden: torch.Tensor,
        state: SchemaGrammarState,
        step_id: int,
    ) -> str:
        logits = self.model.json_action_logits(
            hidden,
            [state],
            torch.tensor([min(step_id, 63)], device=hidden.device),
        )[0]
        allowed = list(state.allowed_actions())
        if not allowed:
            raise ValueError(f"schema grammar has no action: {state.grammar.frames[-1]}")
        indices = torch.tensor(
            [ACTION_INDEX[action] for action in allowed],
            device=logits.device,
        )
        return allowed[int(logits[indices].argmax())]

    def _choose_key(self, hidden: torch.Tensor, state: SchemaGrammarState) -> str:
        allowed = state.allowed_keys()
        if not allowed:
            raise ValueError("schema state has no uncommitted property key")
        scores = self.model.json_schema_key_logits(hidden, [state])[0]
        return allowed[int(scores.argmax())]

    def generate(
        self,
        prompt: str,
        kind: str,
        *,
        max_prompt_tokens: int = 512,
    ) -> GeneratedJSON:
        if kind not in OUTPUT_FIELDS:
            raise ValueError(f"unknown JSON output kind: {kind}")
        hidden = self._hidden(prompt, max_prompt_tokens).detach()
        typed = self.model.json_typed_logits(hidden)
        state = SchemaGrammarState.initial(output_schema(kind))
        decisions = []
        for step in range(4096):
            if state.complete:
                break
            action = self._choose_action(hidden, state, step)
            key = self._choose_key(hidden, state) if action == "key" else None
            value = None
            if action in {"string", "number", "true", "false", "null"}:
                frame = state.grammar.frames[-1]
                schema = (
                    state.schemas[-1]
                    if frame.kind == "root"
                    else state.schemas[-1].properties[state.pending_key]
                    if frame.kind == "object" and state.pending_key is not None
                    else state.schemas[-1].items
                )
                if schema is None:
                    raise ValueError("native output schema is missing an item contract")
                value = _value_for_action(
                    action,
                    field=state.pending_key or "item",
                    schema=schema,
                    typed=typed,
                    hidden=hidden,
                )
            decision = NativeJSONDecision(action, key=key, value=value)
            decisions.append(decision)
            state = state.transition(action, key=key)
        else:
            raise RuntimeError("native schema grammar exceeded its action bound")
        committed = commit_native_json_decisions(kind, decisions)
        return GeneratedJSON(
            text=committed.text,
            value=committed.value,
            actions=tuple(decision.action for decision in committed.decisions),
            invalid_transition_probability=0.0,
            typed_tree=committed.typed_tree,
            structured=committed.structured,
            decisions=committed.decisions,
        )


def output_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "OUTPUT_FIELDS",
    "VALUE_ACTIONS",
    "GeneratedJSON",
    "NativeJSONGenerator",
    "gold_action_sequence",
    "output_sha256",
    "validate_action_sequence",
]
