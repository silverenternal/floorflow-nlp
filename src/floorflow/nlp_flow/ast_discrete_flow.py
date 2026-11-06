"""AST-aware discrete corruption and transition interfaces."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from floorflow.nlp_flow.ast_grammar import ASTTransitionGrammar
from floorflow.nlp_flow.tree_action_decoder import (
    AttachEdgeAction,
    CloseScopeAction,
    DeclareLeafAction,
    OpenNodeAction,
    OpenScopeAction,
    TreeAction,
    TreeActionDecoder,
)

TOKEN_DIMENSIONS = (
    "action_type",
    "node_kind",
    "edge_field",
    "scope_state",
    "leaf_class",
)
MASK = "<mask>"


@dataclass(frozen=True)
class ASTDiscreteToken:
    action_index: int
    dimension: str
    value: str
    masked: bool = False


@dataclass(frozen=True)
class ASTDiscreteProposal:
    tokens: tuple[ASTDiscreteToken, ...]
    target_free: bool = True


@dataclass(frozen=True)
class ASTHybridStep:
    continuous_state: torch.Tensor
    discrete: ASTDiscreteProposal
    step: int


def extract_discrete_tokens(actions: Sequence[TreeAction]) -> list[ASTDiscreteToken]:
    tokens = []
    for index, action in enumerate(actions):
        if isinstance(action, AttachEdgeAction):
            action_type = "attach_edge"
        elif isinstance(action, OpenNodeAction):
            action_type = "open_node"
        elif isinstance(action, OpenScopeAction):
            action_type = "open_scope"
        elif isinstance(action, DeclareLeafAction):
            action_type = "declare_leaf"
        elif isinstance(action, CloseScopeAction):
            action_type = "close_scope"
        else:
            action_type = "terminate"
        tokens.append(ASTDiscreteToken(index, "action_type", action_type))
        if isinstance(action, OpenNodeAction):
            tokens.append(ASTDiscreteToken(index, "node_kind", action.node_kind))
        if isinstance(action, AttachEdgeAction):
            tokens.append(ASTDiscreteToken(index, "edge_field", action.edge_field))
        if isinstance(action, (OpenScopeAction, CloseScopeAction)):
            tokens.append(
                ASTDiscreteToken(
                    index,
                    "scope_state",
                    "open" if isinstance(action, OpenScopeAction) else "close",
                )
            )
        if isinstance(action, DeclareLeafAction):
            tokens.append(
                ASTDiscreteToken(index, "leaf_class", action.leaf_class)
            )
    return tokens


def corruption_schedule(
    tokens: Sequence[ASTDiscreteToken],
    *,
    mask_fraction: float,
    seed: int,
) -> list[ASTDiscreteToken]:
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError("mask_fraction must be in [0, 1]")
    indices = list(range(len(tokens)))
    random.Random(seed).shuffle(indices)
    selected = set(indices[: round(len(indices) * mask_fraction)])
    return [
        ASTDiscreteToken(
            token.action_index,
            token.dimension,
            MASK if index in selected else token.value,
            index in selected,
        )
        for index, token in enumerate(tokens)
    ]


def legal_action_candidates(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
) -> dict[str, tuple[str, ...]]:
    """Return legal structural categories without any target AST."""
    candidates = {
        "action_type": [],
        "node_kind": [],
        "edge_field": [],
        "scope_state": [],
        "leaf_class": [],
    }
    if decoder.expected_scope is not None:
        candidates["action_type"] = ["open_scope"]
        candidates["scope_state"] = ["open"]
        return candidates
    if not decoder.nodes:
        candidates["action_type"] = ["open_node"]
        candidates["node_kind"] = list(grammar.legal_root_kinds)
        return candidates
    if decoder.pending_edge is not None:
        pending = decoder.pending_edge
        candidates["action_type"] = ["open_node"]
        candidates["node_kind"] = list(
            decoder.state.allowed_child_kinds(
                parent_index=pending.parent_index,
                edge_field=pending.edge_field,
            )
        )
        return candidates
    actions = ["attach_edge"]
    for index, node in enumerate(decoder.nodes):
        if node.scope_id not in decoder.state.open_scopes:
            continue
        edges = sorted(
            {
                transition.edge_field
                for transition in grammar.transitions
                if transition.parent_kind == node.kind
            }
        )
        if edges:
            candidates["edge_field"] = edges
            break
    leaves = [
        index
        for index, node in enumerate(decoder.nodes)
        if not node.children
        and node.leaf_class is None
        and grammar.leaf_classes.get(node.kind)
    ]
    if leaves:
        actions.append("declare_leaf")
        candidates["leaf_class"] = sorted(
            {
                value
                for index in leaves
                for value in grammar.leaf_classes[decoder.nodes[index].kind]
            }
        )
    if decoder.state.open_scopes:
        actions.append("close_scope")
        candidates["scope_state"] = ["close"]
    undeclared = any(
        not node.children and node.leaf_class is None
        for node in decoder.nodes
    )
    if decoder.state.eos_ready and not undeclared:
        actions.append("terminate")
    candidates["action_type"] = actions
    return candidates


def sample_legal_dimension(
    values: Sequence[str],
    *,
    allowed: Sequence[str],
    scores: Mapping[str, float],
    generator: torch.Generator | None = None,
) -> str:
    if not allowed:
        raise ValueError("legal token set is empty")
    allowed_values = [value for value in allowed if value in values]
    if not allowed_values:
        raise ValueError("allowed tokens are outside the score vocabulary")
    logits = torch.tensor(
        [float(scores.get(value, 0.0)) for value in allowed_values]
    )
    probabilities = torch.softmax(logits, dim=-1)
    index = int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
    )
    return allowed_values[index]


def sample_legal_discrete_proposal(
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
    logits: Mapping[str, Mapping[str, float]],
    *,
    generator: torch.Generator | None = None,
    step: int = 0,
) -> ASTDiscreteProposal:
    """Sample only grammar-legal tokens; no target AST enters this interface."""
    legal = legal_action_candidates(decoder, grammar)
    action_values = (
        "attach_edge",
        "open_node",
        "open_scope",
        "declare_leaf",
        "close_scope",
        "terminate",
    )
    action = sample_legal_dimension(
        action_values,
        allowed=legal["action_type"],
        scores=logits.get("action_type", {}),
        generator=generator,
    )
    tokens = [ASTDiscreteToken(step, "action_type", action)]
    if action == "open_node":
        node_kind = sample_legal_dimension(
            tuple(legal["node_kind"]),
            allowed=legal["node_kind"],
            scores=logits.get("node_kind", {}),
            generator=generator,
        )
        tokens.append(ASTDiscreteToken(step, "node_kind", node_kind))
    elif action == "attach_edge":
        edge = sample_legal_dimension(
            tuple(legal["edge_field"]),
            allowed=legal["edge_field"],
            scores=logits.get("edge_field", {}),
            generator=generator,
        )
        tokens.append(ASTDiscreteToken(step, "edge_field", edge))
    elif action == "open_scope":
        tokens.append(ASTDiscreteToken(step, "scope_state", "open"))
    elif action == "close_scope":
        tokens.append(ASTDiscreteToken(step, "scope_state", "close"))
    elif action == "declare_leaf":
        leaf_class = sample_legal_dimension(
            tuple(legal["leaf_class"]),
            allowed=legal["leaf_class"],
            scores=logits.get("leaf_class", {}),
            generator=generator,
        )
        tokens.append(ASTDiscreteToken(step, "leaf_class", leaf_class))
    return ASTDiscreteProposal(tuple(tokens))


def hybrid_denoising_step(
    continuous_state: torch.Tensor,
    velocity: torch.Tensor,
    decoder: TreeActionDecoder,
    grammar: ASTTransitionGrammar,
    logits: Mapping[str, Mapping[str, float]],
    *,
    step: int,
    generator: torch.Generator | None = None,
) -> ASTHybridStep:
    """Interface combining continuous structural flow and legal discrete actions."""
    if continuous_state.shape != velocity.shape:
        raise ValueError("continuous state and velocity must align")
    if not continuous_state.is_floating_point():
        raise TypeError("continuous AST state must be floating point")
    proposal = sample_legal_discrete_proposal(
        decoder,
        grammar,
        logits,
        generator=generator,
        step=step,
    )
    return ASTHybridStep(
        continuous_state=continuous_state + velocity,
        discrete=proposal,
        step=step,
    )


def contract_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "MASK",
    "TOKEN_DIMENSIONS",
    "ASTDiscreteProposal",
    "ASTDiscreteToken",
    "ASTHybridStep",
    "contract_fingerprint",
    "corruption_schedule",
    "extract_discrete_tokens",
    "hybrid_denoising_step",
    "legal_action_candidates",
    "sample_legal_dimension",
    "sample_legal_discrete_proposal",
]
