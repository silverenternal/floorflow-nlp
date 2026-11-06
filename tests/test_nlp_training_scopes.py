from __future__ import annotations

import torch

from floorflow.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    ScopeController,
    frozen_parameter_delta,
    parameter_snapshot,
)


def build_model() -> CompactNLPDecoder:
    return CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=128,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            channel_size=8,
            enable_typed_channels=True,
            enable_typed_fusion=True,
            enable_two_stage_cfm=True,
            enable_json_action_head=True,
            enable_expert_factory=True,
        )
    )


def test_staged_scopes_select_expected_prefixes() -> None:
    expected = {
        "S0-base": {"embedding", "blocks", "final_norm"},
        "S1-channels": {"typed_memory"},
        "S2-cross-attention": {"typed_fusion"},
        "S3-cfm": {"cfm"},
    }
    for scope_name, prefixes in expected.items():
        model = build_model()
        report = ScopeController.from_name(scope_name).apply(model)
        assert set(report["prefixes"]) == prefixes


def test_s4_excludes_factory_and_s5_includes_it() -> None:
    s4 = build_model()
    ScopeController.from_name("S4-joint").apply(s4)
    assert not any(
        parameter.requires_grad
        for name, parameter in s4.named_parameters()
        if name.startswith("expert_factory.")
    )

    s5 = build_model()
    ScopeController.from_name("S5-factory").apply(s5)
    assert all(
        parameter.requires_grad
        for name, parameter in s5.named_parameters()
        if name.startswith("expert_factory.")
    )


def test_frozen_parameters_have_zero_delta_after_s2_step() -> None:
    model = build_model()
    ScopeController.from_name("S2-cross-attention").apply(model)
    frozen = parameter_snapshot(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    output = model(torch.randint(0, 128, (2, 5)))
    loss = output.token_logits.square().mean()
    loss.backward()
    optimizer.step()

    deltas = frozen_parameter_delta(model, frozen)

    assert deltas
    assert max(deltas.values()) == 0.0
