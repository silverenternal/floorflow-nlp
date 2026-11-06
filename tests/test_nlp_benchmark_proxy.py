from __future__ import annotations

import torch

from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.benchmark_proxy import (
    REGRESSION_PROXY_TARGETS,
    SCALAR_PROXY_TARGETS,
    TOKEN_PROXY_TARGETS,
    BenchmarkProxyConfig,
    BenchmarkProxyObjective,
)


def test_all_scalar_and_token_proxy_losses_are_finite_and_reachable() -> None:
    model = BenchmarkProxyObjective(
        BenchmarkProxyConfig(hidden_size=32)
    )
    hidden = torch.randn(2, 7, 32)
    pooled = hidden.mean(dim=1)
    scalar_targets = {
        name: torch.randint(0, 2, (2,))
        for name in SCALAR_PROXY_TARGETS
    }
    token_targets = {
        name: torch.randint(0, 2, (2, 7))
        for name in TOKEN_PROXY_TARGETS
    }
    regression_targets = {
        name: torch.rand(2) * 2.0
        for name in REGRESSION_PROXY_TARGETS
    }
    losses = {
        **model.scalar_losses(pooled, scalar_targets),
        **model.regression_losses(pooled, regression_targets),
        **model.token_losses(hidden, token_targets),
        "L_test_output": model.test_output_loss(
            pooled,
            torch.tensor([0, 3]),
        ),
        "L_test_output_token_sequence": model.test_output_token_loss(
            hidden,
            torch.randint(0, 64, (2, 7)),
        ),
        "L_multiple_choice": model.multiple_choice_loss(
            pooled,
            torch.randn(2, 4, 32),
            torch.tensor([0, 3]),
        ),
        "L_instruction_constraints": model.constraint_loss(
            pooled,
            torch.randint(0, 2, (2, 16)).float(),
            torch.ones(2, 16, dtype=torch.bool),
        ),
    }
    total = model.total_loss(losses)

    assert torch.isfinite(total)
    total.backward()
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("retrieval_project.")
        and parameter.grad is None
    ]
    assert missing_gradients == []


def test_retrieval_and_metamorphic_proxies_have_expected_properties() -> None:
    model = BenchmarkProxyObjective(
        BenchmarkProxyConfig(hidden_size=32, retrieval_dim=8)
    )
    query = torch.randn(3, 32)
    candidates = torch.randn(3, 5, 32)

    loss = model.retrieval_loss(
        query,
        candidates,
        torch.tensor([0, 2, 4]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.retrieval_project.weight.grad is not None
    logits = torch.randn(2, 7, 13)
    assert torch.isclose(
        model.metamorphic_consistency_loss(logits, logits),
        torch.tensor(0.0),
        atol=1.0e-6,
    )
    assert model.metamorphic_consistency_loss(
        logits,
        logits + torch.randn_like(logits),
    ).item() > 0.0


def test_proxy_heads_remain_bounded() -> None:
    model = BenchmarkProxyObjective(
        BenchmarkProxyConfig(hidden_size=320)
    )
    parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    assert parameters < 100_000


def test_compact_model_can_enable_benchmark_proxy_heads() -> None:
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=128,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            parameter_cap=1_000_000,
            enable_benchmark_proxy=True,
            proxy_test_output_classes=8,
            proxy_max_constraints=4,
            proxy_retrieval_dim=8,
        )
    )

    assert model.benchmark_proxy is not None
    assert model.benchmark_proxy.config.test_output_classes == 8
    output = model(torch.randint(0, 128, (2, 8)))
    assert output.benchmark_proxy is not None
    assert output.benchmark_proxy.test_output_logits.shape == (2, 8)
    assert output.benchmark_proxy.test_output_token_logits.shape == (2, 8, 8)
    assert set(output.benchmark_proxy.scalar_logits) == set(
        SCALAR_PROXY_TARGETS
    )
    assert set(output.benchmark_proxy.regression_logits) == set(
        REGRESSION_PROXY_TARGETS
    )
