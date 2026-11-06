from __future__ import annotations

import torch

from floorflow.nlp_flow.engineering import (
    audit_parameter_budget,
    count_unique_trainable_parameters,
    load_nlp_checkpoint,
    restore_nlp_checkpoint,
    save_nlp_checkpoint,
)
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def tiny_config() -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        channel_size=8,
        parameter_cap=2_300_000,
        enable_typed_channels=True,
    )


def test_compact_decoder_shapes_tied_parameters_and_typed_heads() -> None:
    model = CompactNLPDecoder(tiny_config())
    ids = torch.randint(0, 128, (2, 12))

    output = model(ids)

    assert output.hidden_states.shape == (2, 12, 32)
    assert output.token_logits.shape == (2, 12, 128)
    assert set(output.typed_channels) == {
        "syntax",
        "symbol",
        "discourse",
        "execution",
        "engineering",
    }
    assert all(value.shape == (2, 12, 8) for value in output.typed_channels.values())
    assert count_unique_trainable_parameters(model) <= 2_300_000


def test_default_model_matches_p2m_budget_with_cfm_headroom() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(enable_typed_channels=True, enable_typed_fusion=True)
    )
    parameters = count_unique_trainable_parameters(model)
    report = audit_parameter_budget(parameters, 46_000_000)

    assert report.passed
    assert 2_300_000 - parameters >= 40_000


def test_control_without_typed_channels_has_no_channel_branch() -> None:
    config = tiny_config()
    control_config = NLPModelConfig(
        **{**config.to_dict(), "enable_typed_channels": False}
    )
    model = CompactNLPDecoder(control_config)

    output = model(torch.randint(0, 128, (2, 5)))

    assert output.typed_channels == {}
    assert output.typed_memory is None


def test_causal_attention_prevents_future_leakage() -> None:
    model = CompactNLPDecoder(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 9, 8]])

    with torch.no_grad():
        left = model(first).hidden_states
        right = model(second).hidden_states

    assert torch.allclose(left[:, :3], right[:, :3], atol=1e-6)


def test_padded_keys_do_not_change_valid_prefix() -> None:
    model = CompactNLPDecoder(tiny_config()).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    first = torch.tensor([[1, 2, 3, 0, 0]])
    second = torch.tensor([[1, 2, 3, 7, 6]])

    with torch.no_grad():
        left = model(first, attention_mask=mask).hidden_states
        right = model(second, attention_mask=mask).hidden_states

    assert torch.allclose(left[:, :3], right[:, :3], atol=1e-6)


def test_channel_heads_are_separately_trainable() -> None:
    model = CompactNLPDecoder(tiny_config())
    output = model(torch.randint(0, 128, (2, 6)))
    loss = output.typed_memory.global_memory.square().mean()
    loss.backward()

    assert model.embedding.weight.grad is not None
    for name in output.typed_channels:
        weight = model.typed_memory.down[name].weight
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()


def test_compact_control_can_train_and_round_trip_checkpoint(tmp_path) -> None:
    model = CompactNLPDecoder(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 128, (2, 8))
    target = torch.randint(0, 128, (2, 8))
    output = model(ids)
    loss = torch.nn.functional.cross_entropy(
        output.token_logits.reshape(-1, 128),
        target.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "control.pt"
    save_nlp_checkpoint(
        checkpoint,
        step=1,
        samples_seen=2,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        spec_sha256="control",
        data_manifest_sha256="data",
        tokenizer_sha256="tokenizer",
        config=tiny_config().to_dict(),
        device="cpu",
    )

    restored = CompactNLPDecoder(tiny_config())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_nlp_checkpoint(checkpoint, expected_spec_sha256="control")
    state = restore_nlp_checkpoint(
        payload,
        model=restored,
        optimizer=restored_optimizer,
        device="cpu",
    )

    assert state == {"step": 1, "samples_seen": 2}
    for left, right in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(left, right)


def test_tied_logits_start_near_uniform_scale() -> None:
    torch.manual_seed(5)
    model = CompactNLPDecoder(tiny_config()).eval()
    logits = model(torch.zeros(1, 4, dtype=torch.long)).token_logits

    assert logits.std().item() < 1.0
    assert logits.abs().mean().item() < 2.0
