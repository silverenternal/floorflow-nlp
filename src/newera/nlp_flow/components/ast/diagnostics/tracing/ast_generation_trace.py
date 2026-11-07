"""Optional, context-local tracing for the historical action decoders."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict

import torch

from newera.nlp_flow.components.ast.diagnostics.tracing.ast_state_trace import snapshot
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import TreeAction, TreeActionDecoder

_choices: ContextVar[list[dict[str, object]] | None] = ContextVar("ast_trace_choices", default=None)


def tracing_enabled() -> bool:
    return _choices.get() is not None


def record_choice(logits: torch.Tensor, allowed: set[int], selected: int | None, head: str) -> None:
    """Record the actual candidates and logits without changing selection."""
    choices = _choices.get()
    if choices is None:
        return
    scores = logits.detach().cpu().tolist()
    choices.append({"head": head, "allowed_ids": sorted(allowed), "logits": scores, "selected_id": selected})


def record_event(head: str, **details: object) -> None:
    """Record a non-masked branch such as a copy decision."""
    choices = _choices.get()
    if choices is not None:
        choices.append({"head": head, **details})


def finish_trace(trace: list[dict[str, object]] | None, decoder: TreeActionDecoder, error: str | None) -> None:
    """Record the actual terminal result, including parser and step-cap failures."""
    if trace is not None:
        trace.append({"event": "generation_end", "state": asdict(snapshot(decoder)), "error": error})


def select_and_apply(
    decoder: TreeActionDecoder,
    select: Callable[[], TreeAction],
    trace: list[dict[str, object]] | None,
    *,
    budget: int,
    parent_candidates: list[int],
) -> TreeAction:
    """Trace a single policy decision; propagate failures without retrying."""
    if trace is None:
        action = select()
        decoder.apply(action)
        return action
    before = asdict(snapshot(decoder))
    choices: list[dict[str, object]] = []
    token = _choices.set(choices)
    action = None
    try:
        action = select()
        decoder.apply(action)
    except Exception as error:
        trace.append(
            {
                "step": before["action_count"],
                "node_budget": budget,
                "parent_candidates": list(parent_candidates),
                "before": before,
                "after": asdict(snapshot(decoder)),
                "choices": choices,
                "action": None if action is None else {"type": type(action).__name__, **asdict(action)},
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        _choices.reset(token)
    trace.append(
        {
            "step": before["action_count"],
            "node_budget": budget,
            "parent_candidates": list(parent_candidates),
            "before": before,
            "after": asdict(snapshot(decoder)),
            "choices": choices,
            "action": {"type": type(action).__name__, **asdict(action)},
            "error_type": None,
            "error": None,
        }
    )
    return action
