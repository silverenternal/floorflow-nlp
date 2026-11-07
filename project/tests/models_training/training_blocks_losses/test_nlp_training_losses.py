from __future__ import annotations

from contextlib import nullcontext

import torch

from newera.nlp_flow.components.research.training.objectives.training_losses import batch_loss
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig


def test_joint_loss_reaches_conversation_and_proxy_heads() -> None:
    torch.manual_seed(307)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            parameter_cap=2_000_000,
            enable_turn_embeddings=True,
            enable_conversation_control=True,
            enable_benchmark_proxy=True,
            proxy_test_output_classes=8,
            proxy_max_constraints=4,
            proxy_retrieval_dim=8,
            enable_emergence_objective=True,
            emergence_basis_count=16,
            emergence_active_basis=4,
        )
    )
    batch_size, sequence = 2, 10
    input_ids = torch.randint(0, 64, (batch_size, sequence))
    labels = input_ids.clone()
    labels[:, :5] = -100
    validity = torch.zeros(batch_size, sequence, 3)
    validity[:, 5:, 0] = 1.0
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(batch_size, sequence, dtype=torch.bool),
        "turn_ids": torch.tensor(
            [[0] * 5 + [1] * 5, [0] * 5 + [1] * 5]
        ),
        "continue_targets": torch.tensor(
            [[0.0] * 5 + [1.0] * 5, [0.0] * 5 + [1.0] * 5]
        ),
        "turn_boundary_targets": torch.full(
            (batch_size, sequence),
            -100,
            dtype=torch.long,
        ),
        "reply_mode_targets": torch.full(
            (batch_size, sequence),
            -100,
            dtype=torch.long,
        ),
        "language_targets": torch.full(
            (batch_size, sequence),
            -100,
            dtype=torch.long,
        ),
        "validity_targets": validity,
        "validity_mask": torch.ones(batch_size, sequence, 3, dtype=torch.bool),
        "proxy_execution_python": torch.tensor([1.0, 0.0]),
        "surface_valid": torch.ones(batch_size, sequence, dtype=torch.bool),
        "execution_valid": torch.cat(
            (
                torch.ones(batch_size, sequence // 2, dtype=torch.long),
                torch.zeros(batch_size, sequence // 2, dtype=torch.long),
            ),
            dim=1,
        ),
    }
    batch["turn_boundary_targets"][:, -2] = 1
    batch["reply_mode_targets"][:, 5:] = 1
    batch["language_targets"][:, 5:] = 1

    loss, _, _, emergence_metrics = batch_loss(
        model,
        batch,
        device=torch.device("cpu"),
        autocast=nullcontext(),
        conversation_control_weight=0.5,
        benchmark_proxy_weight=0.5,
        emergence_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert emergence_metrics is not None
    assert "emergence_score" in emergence_metrics
    assert model.emergence.encoder.weight.grad is not None
    assert model.conversation_head.continue_head.weight.grad is not None
    assert model.benchmark_proxy.scalar_heads["execution_python"][-1].weight.grad is not None


def test_language_loss_ignores_packed_segment_boundaries_and_padding() -> None:
    torch.manual_seed(41)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            max_seq_len=8,
            parameter_cap=2_000_000,
            enable_neuron_graph=True,
        )
    ).eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.tensor([[True, True, True, True, True, False]]),
        "segment_ids": torch.tensor([[0, 0, 1, 1, 1, -1]]),
        "neuron_node_ids": torch.tensor([[10, 11, 12, 13, 14, -1]]),
    }
    base, *_ = batch_loss(
        model, batch, device=torch.device("cpu"), autocast=nullcontext()
    )
    changed = {**batch, "labels": batch["labels"].clone()}
    changed["labels"][0, 2] = 17  # first token of segment 1
    changed["labels"][0, 5] = 21  # padding token
    modified, *_ = batch_loss(
        model, changed, device=torch.device("cpu"), autocast=nullcontext()
    )
    assert torch.allclose(base, modified)


def test_proxy_registry_wires_retrieval_class_and_hidden_gradients() -> None:
    torch.manual_seed(719)
    model = CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            parameter_cap=2_000_000,
            enable_benchmark_proxy=True,
            proxy_test_output_classes=8,
            proxy_retrieval_dim=8,
            proxy_max_constraints=4,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8))
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones(2, 8, dtype=torch.bool),
        "proxy_retrieval_candidates": torch.randn(2, 3, 32),
        "proxy_retrieval_target": torch.tensor([0, 2]),
        "proxy_test_output_class": torch.tensor([1, 7]),
    }

    loss, *_ = batch_loss(
        model,
        batch,
        device=torch.device("cpu"),
        autocast=nullcontext(),
        benchmark_proxy_weight=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.benchmark_proxy.retrieval_project.weight.grad is not None
    assert model.benchmark_proxy.test_output_head.weight.grad is not None
    assert model.embedding.weight.grad is not None
