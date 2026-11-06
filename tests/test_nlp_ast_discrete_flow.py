from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_discrete_flow import (
    MASK,
    TOKEN_DIMENSIONS,
    corruption_schedule,
    extract_discrete_tokens,
    hybrid_denoising_step,
    legal_action_candidates,
    sample_legal_discrete_proposal,
)
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.tree_action_decoder import (
    OpenNodeAction,
    TreeActionDecoder,
    trace_teacher_actions,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/nlp_ast_transition_grammar_v1.json"


def test_discrete_corruption_covers_every_ast_action_dimension() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    payload = json.loads(
        serialize_ast_graph(
            parse_ast_graph("value = 1\n", "python")
        )
    )
    tokens = extract_discrete_tokens(trace_teacher_actions(grammar, payload))
    corrupted = corruption_schedule(tokens, mask_fraction=0.25, seed=3)

    assert {token.dimension for token in tokens} == set(TOKEN_DIMENSIONS)
    assert any(token.value == MASK for token in corrupted)
    assert not any(
        token.masked and token.dimension == "scope_state"
        and token.value not in {MASK, "open", "close"}
        for token in corrupted
    )


def test_legal_action_candidates_are_derived_without_target_ast() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    decoder = TreeActionDecoder(grammar)
    initial = legal_action_candidates(decoder, grammar)

    assert initial["action_type"] == ["open_node"]
    assert initial["node_kind"] == ["module"]

    decoder.apply(OpenNodeAction("module"))
    scoped = legal_action_candidates(decoder, grammar)
    assert scoped["action_type"] == ["open_scope"]


def test_discrete_and_hybrid_interfaces_are_target_free() -> None:
    grammar = load_ast_grammar(GRAMMAR, "python")
    decoder = TreeActionDecoder(grammar)
    proposal = sample_legal_discrete_proposal(
        decoder,
        grammar,
        {
            "action_type": {"open_node": 10.0},
            "node_kind": {"module": 10.0},
        },
        generator=torch.Generator().manual_seed(4),
    )
    hybrid = hybrid_denoising_step(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        decoder,
        grammar,
        {
            "action_type": {"open_node": 10.0},
            "node_kind": {"module": 10.0},
        },
        step=0,
    )

    assert proposal.target_free
    assert [(token.dimension, token.value) for token in proposal.tokens] == [
        ("action_type", "open_node"),
        ("node_kind", "module"),
    ]
    assert torch.equal(hybrid.continuous_state, torch.ones(2, 3))
    assert not any(
        "target" in name.lower()
        for name in inspect.signature(sample_legal_discrete_proposal).parameters
    )
