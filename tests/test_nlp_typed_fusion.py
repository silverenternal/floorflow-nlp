from __future__ import annotations

import torch

from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    TypedChannelFusion,
    TypedChannelMemoryEncoder,
)


def tiny_config(fusion: bool) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        channel_size=8,
        enable_typed_channels=True,
        enable_typed_fusion=fusion,
    )


def test_typed_fusion_is_identity_at_initialization() -> None:
    channels = TypedChannelMemoryEncoder(32, 8)
    fusion = TypedChannelFusion(32)
    hidden = torch.randn(2, 6, 32)
    memory = channels(hidden)

    output = fusion(hidden, memory)

    assert torch.equal(output.hidden_states, hidden)
    assert "slots_hidden_to_memory_gate" in output.diagnostics


def test_typed_fusion_gates_and_heads_receive_gradients() -> None:
    channels = TypedChannelMemoryEncoder(32, 8)
    fusion = TypedChannelFusion(32)
    hidden = torch.randn(2, 6, 32)
    memory = channels(hidden)
    output = fusion(hidden, memory)

    loss = output.hidden_states.square().mean()
    loss.backward()

    assert fusion.cross_attention.gate.grad is not None
    assert fusion.router.gate.grad is not None
    assert fusion.slot_mixer.gate.grad is not None
    assert fusion.cross_attention.q.weight.grad is not None
    assert fusion.router.route.q_proj.weight.grad is not None


def test_fused_model_matches_unfused_control_at_initialization() -> None:
    torch.manual_seed(7)
    control = CompactNLPDecoder(tiny_config(False)).eval()
    torch.manual_seed(7)
    fused = CompactNLPDecoder(tiny_config(True)).eval()
    ids = torch.randint(0, 128, (2, 7))

    with torch.no_grad():
        left = control(ids).token_logits
        right = fused(ids).token_logits

    assert torch.allclose(left, right, atol=1e-6)


def test_fusion_stack_stays_compact() -> None:
    fusion = TypedChannelFusion(160)

    assert count_unique_trainable_parameters(fusion) < 100_000
